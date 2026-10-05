"""Decoded FP8 reference and full-depth HrmText native KV/Graph checks."""

from pathlib import Path
from typing import Literal

import torch

from loopquant.fused_fp8 import RowwiseFP8Linear
from loopquant.hrm_fp8 import quantize_hrm_fp8
from loopquant.hrm_fp8_export import export_hrm_fp8, load_hrm_fp8
from loopquant.int4_export import core_projections
from loopquant.quality import native_continuation_nll
from loopquant.quantizers import ScaleLayout
from tests.loopquant_int4_checks import int4_reference_model
from tests.loopquant_serving_checks import ReferenceLinear, RowContext
from vllm_rlt import CacheConfig, ExecutionConfig, ExitConfig, SamplingParams, SchedulerConfig
from vllm_rlt.engine.llm_engine import LLMEngine
from vllm_rlt.models.hrm_text import HrmTextForCausalLM


class ReferenceHrmText(HrmTextForCausalLM):
    def quantize(
        self,
        scales: dict[str, torch.Tensor],
        layout: ScaleLayout,
        mode: Literal["static", "dynamic"],
    ) -> None:
        self.context = RowContext()
        for name, layer in core_projections(self).items():
            parent, field = name.rsplit(".", 1)
            self.get_submodule(parent).add_module(
                field, ReferenceLinear(layer, scales[name], layout, mode, self.context)
            )

    def recurrent_prepared(self, hidden, batch, cache, *, compute_gate=True):
        self.context.loop_ids = batch.loop_ids
        return super().recurrent_prepared(hidden, batch, cache, compute_gate=compute_gate)


@torch.inference_mode()
def check_hrm_fp8(folder: Path, policy: str, graphs: bool, compact: bool) -> dict:
    torch.manual_seed(37)
    native = int4_reference_model("hrm_text").to(device="cuda", dtype=torch.bfloat16).eval()
    reference = ReferenceHrmText(native.config).to(device="cuda", dtype=torch.bfloat16).eval()
    reference.load_state_dict(native.state_dict())
    mode = "dynamic" if policy == "DYN" else "static"
    layout = ScaleLayout(2, {"DYN": (), "SH1": (), "ST2": (1,), "LOOP": (1,)}[policy])
    scales = {
        name: torch.linspace(0.005, 0.04, layout.stages, device="cuda")
        for name in core_projections(native)
    }
    protected = {
        name: value.clone()
        for name, value in native.state_dict().items()
        if name not in {key + ".weight" for key in core_projections(native)}
    }
    reference.quantize(scales, layout, mode)
    quantize_hrm_fp8(native, scales, layout, mode, max_rows=8)
    export_hrm_fp8(native, folder, model_revision="a" * 40)
    restored = load_hrm_fp8(folder, torch.device("cuda"), max_rows=8)
    left, right = native.state_dict(), restored.state_dict()
    assert left.keys() == right.keys()
    assert all(
        torch.equal(left[name].view(torch.uint8), right[name].view(torch.uint8)) for name in left
    )
    del native, left, right
    native = restored
    assert dict(native.named_parameters()).keys() == protected.keys()
    assert all(torch.equal(native.state_dict()[name], value) for name, value in protected.items())
    assert native.model.rotary_emb.inv_freq.dtype == torch.float32
    assert torch.equal(native.model.rotary_emb.inv_freq, reference.model.rotary_emb.inv_freq)
    weights = {
        name: layer.packed_weight.data_ptr()
        for name, layer in native.named_modules()
        if isinstance(layer, RowwiseFP8Linear)
    }
    assert len(weights) == len(set(weights.values())) == 16
    common = dict(
        attention_backend="triton",
        cache_config=CacheConfig(128, 4, compact_last_exited=compact),
        scheduler_config=SchedulerConfig(max_num_seqs=3, max_num_batched_tokens=8),
        exit_config=ExitConfig(mode="ouro_delayed"),
    )
    engines = [
        LLMEngine(reference, **common),
        LLMEngine(
            native,
            execution_config=ExecutionConfig(
                cuda_graphs=graphs,
                static_buffers=True,
                pad_to_power_of_two=True,
                async_scheduling=True,
                multi_stream=True,
            ),
            **common,
        ),
    ]
    tokens = torch.tensor([4, 7, 3, 9, 2, 8, 11], device="cuda")
    likelihood = []
    for prefix in (1, 3):
        scores = [
            native_continuation_nll(engine.model, tokens, prefix, 2, engine.cache_manager)
            for engine in engines
        ]
        assert scores[0][1] == scores[1][1] == len(tokens) - prefix
        torch.testing.assert_close(scores[0][0], scores[1][0], atol=0.02, rtol=0.02)
        assert all(engine.cache_manager.num_used_blocks == 0 for engine in engines)
        likelihood.append(
            dict(
                prefix=prefix,
                targets=scores[0][1],
                reference_nll=scores[0][0],
                native_nll=scores[1][0],
            )
        )
    results = []
    for engine in engines:
        rounds = []
        for _ in range(2):
            for index, prompt in enumerate(([4, 7], [9, 3, 2], [8])):
                engine.add_request(
                    str(index),
                    prompt,
                    SamplingParams(
                        max_tokens=4,
                        min_loops=2,
                        max_loops=2,
                        ignore_eos=True,
                    ),
                )
            finished = {}
            while engine.has_unfinished_requests():
                for output in engine.step():
                    if output.finished:
                        finished[output.request_id] = (output.token_ids, output.exit_depths)
            assert len(finished) == 3 and engine.cache_manager.num_used_blocks == 0
            assert all(depths == [2] * 4 for _, depths in finished.values())
            rounds.append(finished)
        results.append(rounds)
    assert results[0] == results[1]
    assert weights == {
        name: layer.packed_weight.data_ptr()
        for name, layer in native.named_modules()
        if isinstance(layer, RowwiseFP8Linear)
    }
    graph = engines[1].model_runner.graphs
    if graphs:
        assert graph.captures > 0 and graph.replays > graph.captures
    return dict(
        policy=policy,
        graphs=graphs,
        compact=compact,
        asynchronous=True,
        encoded_reload_exact=True,
        physical_packed_weights=len(weights),
        repeated_requests=6,
        full_depth=2,
        likelihood=likelihood,
        protected_parameters_unchanged=True,
        low_applications_per_outer_cycle=3,
        kv_after_drain=0,
        captures=graph.captures if graphs else 0,
        replays=graph.replays if graphs else 0,
    )
