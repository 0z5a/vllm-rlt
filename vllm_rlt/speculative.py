"""Single-request fixed-depth greedy token speculation for LAST_EXITED Ouro."""

from collections.abc import Callable
from dataclasses import dataclass

import torch

from vllm_rlt.core.kv_cache_manager import KVCacheManager


def online_ngram(history: list[int], limit: int, max_order: int = 4) -> list[int]:
    """Propose only continuations already present in committed history."""
    draft = []
    for _ in range(limit):
        context = history + draft
        match = None
        for width in range(min(max_order, len(context)), 0, -1):
            suffix = context[-width:]
            for start in range(len(history) - width - 1, -1, -1):
                if history[start : start + width] == suffix:
                    match = history[start + width]
                    break
            if match is not None:
                break
        if match is None:
            break
        draft.append(match)
    return draft


@dataclass(frozen=True)
class GreedyResult:
    token_ids: list[int]
    exit_depths: list[int]
    accepted_lengths: list[int]
    drafted_tokens: int
    target_cycles: int


class FixedDepthGreedy:
    """Verify a short proposed suffix in one target traversal, then commit its KV prefix."""

    def __init__(self, model, cache: KVCacheManager, *, prefill_chunk_size: int = 128):
        if cache.layout != "last_exited":
            raise ValueError("fixed-depth speculation requires last_exited KV")
        self.model = model.eval()
        self.cache = cache
        self.prefill_chunk_size = prefill_chunk_size
        self._next_request = 0

    def _forward(self, request_id: str, start: int, tokens: list[int]) -> torch.Tensor:
        positions = list(range(start, start + len(tokens)))
        hidden = self.model.prelude(torch.tensor(tokens, device=self.cache.device))
        for depth in range(self.model.config.total_ut_steps):
            hidden, _ = self.model.recurrent(
                hidden,
                [request_id] * len(tokens),
                [depth] * len(tokens),
                positions,
                self.cache,
                compute_gate=False,
            )
        return self.model.coda(hidden)

    @torch.inference_mode()
    def generate(
        self,
        prompt: list[int],
        max_tokens: int,
        *,
        gamma: int = 0,
        proposer: Callable[[list[int], int], list[int]] = online_ngram,
        ignore_eos: bool = False,
    ) -> GreedyResult:
        if not prompt or max_tokens < 1 or gamma < 0:
            raise ValueError("prompt and output budget must be positive; gamma must be nonnegative")
        capacity = len(prompt) + max_tokens - 1
        request_id = f"fixed-greedy-{self._next_request}"
        self._next_request += 1
        if not self.cache.allocate(request_id, capacity):
            raise MemoryError("insufficient KV blocks for fixed-depth generation")
        eos = self.model.config.eos_token_id
        eos_ids = set(eos if isinstance(eos, (tuple, list)) else (eos,))
        accepted_lengths = []
        drafted_tokens = target_cycles = 0
        try:
            for start in range(0, len(prompt), self.prefill_chunk_size):
                chunk = prompt[start : start + self.prefill_chunk_size]
                logits = self._forward(request_id, start, chunk)
            output = [int(logits[-1].argmax().item())]
            position = len(prompt)
            while len(output) < max_tokens and (ignore_eos or output[-1] not in eos_ids):
                remaining = max_tokens - len(output)
                draft = proposer(prompt + output, min(gamma, remaining - 1)) if gamma else []
                if len(draft) > min(gamma, remaining - 1):
                    raise ValueError("proposer exceeded the requested draft length")
                if not draft:
                    token = int(self._forward(request_id, position, output[-1:])[-1].argmax())
                    position += 1
                    output.append(token)
                    target_cycles += 1
                    continue
                fork_id = f"{request_id}:cycle-{target_cycles}"
                try:
                    forked = self.cache.fork_prefix(
                        request_id, fork_id, position, position + len(draft) + 1
                    )
                    if not forked:
                        token = int(self._forward(request_id, position, output[-1:])[-1].argmax())
                        position += 1
                        output.append(token)
                    else:
                        predictions = self._forward(fork_id, position, output[-1:] + draft)
                        predictions = predictions.argmax(dim=-1).tolist()
                        accepted = 0
                        for candidate, target in zip(draft, predictions):
                            if candidate != target:
                                break
                            accepted += 1
                            if candidate in eos_ids and not ignore_eos:
                                break
                        emitted = draft[:accepted]
                        if ignore_eos or not emitted or emitted[-1] not in eos_ids:
                            emitted.append(predictions[accepted])
                        self.cache.commit_fork(request_id, fork_id, position + accepted + 1)
                        position += accepted + 1
                        output.extend(emitted)
                        accepted_lengths.append(accepted)
                        drafted_tokens += len(draft)
                finally:
                    self.cache.free(fork_id)
                target_cycles += 1
            return GreedyResult(
                output,
                [self.model.config.total_ut_steps] * len(output),
                accepted_lengths,
                drafted_tokens,
                target_cycles,
            )
        finally:
            self.cache.free(request_id)
