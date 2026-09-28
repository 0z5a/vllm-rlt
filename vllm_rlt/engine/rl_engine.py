"""Synchronous fixed-budget rollouts with selected-token probabilities and publication."""

from collections.abc import Mapping
from dataclasses import dataclass, replace

import torch

from vllm_rlt.config import CacheConfig, ExecutionConfig, SchedulerConfig
from vllm_rlt.engine.llm_engine import LLMEngine
from vllm_rlt.request import Request
from vllm_rlt.sampling_params import SamplingParams
from vllm_rlt.worker.model_runner import ModelRunner


class RLModelRunner(ModelRunner):
    selected_log_probs: dict[str, list[torch.Tensor]]
    phase: str

    def _prefill_tokens(self, ids, positions, tokens):
        self.phase = "prefill"
        hidden = super()._prefill_tokens(ids, positions, tokens)
        self.phase = "decode"
        return hidden

    def _sample_tensor(self, logits: torch.Tensor, request: Request) -> torch.Tensor:
        token = super()._sample_tensor(logits, request)
        # This API admits temperature > 0 and unfiltered sampling only.
        logprob = (logits.float() / request.sampling_params.temperature).log_softmax(-1)[token]
        self.selected_log_probs.setdefault(request.request_id, []).append(logprob.clone())
        return token


@dataclass(frozen=True)
class RLCompletion:
    token_ids: list[int]
    log_probs: list[float]
    exit_depths: list[int]
    policy_version: int
    loop_budget: int
    finish_reason: str


class RLEngine(LLMEngine):
    """One rollout batch at a time; no cross-version/depth KV reuse.

    Prefix caching, asynchronous execution and speculative decoding are excluded
    by construction. Physical KV pages are reused only after every request finishes.
    """

    model_runner: RLModelRunner

    def __init__(
        self,
        model,
        *,
        num_blocks: int = 4096,
        attention_backend: str = "triton",
        max_num_seqs: int = 8,
        max_num_batched_tokens: int = 128,
    ):
        super().__init__(
            model,
            cache_config=CacheConfig(num_blocks=num_blocks, layout="last_exited"),
            scheduler_config=SchedulerConfig(
                max_num_seqs=max_num_seqs, max_num_batched_tokens=max_num_batched_tokens
            ),
            execution_config=ExecutionConfig(),
            attention_backend=attention_backend,
            model_runner_class=RLModelRunner,
        )
        self.model_runner.selected_log_probs = {}
        self.model_runner.phase = "decode"
        self.block_tokens = {"prefill": 0, "decode": 0}
        for layer in model.model.layers:
            layer.register_forward_pre_hook(self._record_block_tokens)
        self.policy_version = 0
        self.ready = True
        self.batch_index = 0

    def add_request(
        self,
        request_id: str,
        prompt_token_ids: list[int],
        sampling_params: SamplingParams | None = None,
        *,
        trace_id=None,
    ):
        if not self.ready:
            raise RuntimeError("Weight publication did not complete")
        params = sampling_params or SamplingParams()
        if (
            params.temperature <= 0
            or params.top_k != -1
            or params.top_p != 1
            or params.min_loops != self.model.config.total_ut_steps
            or params.max_loops != self.model.config.total_ut_steps
            or params.exit_threshold != 1
        ):
            raise ValueError("RL engine requires a uniform fixed depth and unfiltered sampling")
        return super().add_request(request_id, prompt_token_ids, params, trace_id=trace_id)

    def _record_block_tokens(self, module, inputs) -> None:
        self.block_tokens[self.model_runner.phase] += inputs[0].shape[0]

    def set_loop_budget(self, loops: int) -> None:
        if self.has_unfinished_requests():
            raise RuntimeError("Finish the active rollout before changing depth")
        if type(loops) is not int or not 1 <= loops <= self.cache_manager.max_loops:
            raise ValueError("Unsupported loop budget")
        self.model.config = replace(self.model.config, total_ut_steps=loops)

    @torch.no_grad()
    def publish(self, weights: Mapping[str, torch.Tensor], version: int) -> None:
        if self.has_unfinished_requests() or version <= self.policy_version:
            raise ValueError("Publication requires an idle engine and a newer version")
        target = dict(self.model.named_parameters())
        if target.keys() != weights.keys():
            raise ValueError("Publication must include all physical Ouro parameters")
        for name, parameter in target.items():
            if parameter.shape != weights[name].shape:
                raise ValueError(f"Shape mismatch: {name}")
        self.ready = False
        for name, parameter in target.items():
            parameter.copy_(weights[name])
        device = next(self.model.parameters()).device
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        self.policy_version = version
        self.ready = True

    def generate_batch(
        self,
        prompts: list[list[int]],
        *,
        loops: int,
        max_tokens: int,
        seed: int,
        temperature: float = 1.0,
    ) -> list[RLCompletion]:
        if not self.ready:
            raise RuntimeError("Weight publication did not complete")
        if temperature <= 0:
            raise ValueError("RL requires a positive sampling temperature")
        self.set_loop_budget(loops)
        self.block_tokens = {"prefill": 0, "decode": 0}
        ids = [f"rl-{self.batch_index}-{index}" for index in range(len(prompts))]
        self.batch_index += 1
        for index, (request_id, prompt) in enumerate(zip(ids, prompts, strict=True)):
            self.add_request(
                request_id,
                prompt,
                SamplingParams(
                    max_tokens=max_tokens,
                    temperature=temperature,
                    seed=seed + index,
                    min_loops=loops,
                    max_loops=loops,
                ),
            )
        results = {}
        while self.has_unfinished_requests():
            for output in self.step():
                if output.finished:
                    assert output.finish_reason is not None
                    logs = self.model_runner.selected_log_probs.pop(output.request_id)
                    results[output.request_id] = RLCompletion(
                        output.token_ids,
                        torch.stack(logs).tolist(),
                        output.exit_depths,
                        self.policy_version,
                        loops,
                        output.finish_reason,
                    )
        return [results[request_id] for request_id in ids]
