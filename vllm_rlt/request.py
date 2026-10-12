from dataclasses import dataclass, field
from enum import Enum

import torch

from vllm_rlt.sampling_params import SamplingParams


class Stage(str, Enum):
    WAITING = "waiting"
    RECEIVING = "receiving"
    PREFILL = "prefill"
    PREFILLING_PRELUDE = "prefilling_prelude"
    PREFILLING_RECURRENT = "prefilling_recurrent"
    PRELUDE = "prelude"
    RECURRENT = "recurrent"
    SPECULATIVE = "speculative"
    CODA = "coda"
    FINISHED = "finished"


class FinishReason(str, Enum):
    STOP = "stop"
    LENGTH = "length"
    ABORT = "abort"


@dataclass
class Request:
    request_id: str
    prompt_token_ids: list[int]
    sampling_params: SamplingParams
    stage: Stage = Stage.WAITING
    generated_token_ids: list[int] = field(default_factory=list)
    exit_depths: list[int] = field(default_factory=list)
    # Scheduling advances before coda's CPU output delivery.
    num_output_placeholders: int = 0
    num_speculative_rounds: int = 0
    input_token_tensor: torch.Tensor | None = field(default=None, repr=False)
    num_prefilled_tokens: int = 0
    # Split prefill. [num_prefilled_tokens, num_preluded_tokens) is the staged
    # chunk: its prelude output is resident and it advances one depth per batch.
    num_preluded_tokens: int = 0
    # Start of the resident chunk. It stays fixed while the final depth advances
    # a partial chunk and is used for runner pool row addressing.
    prefill_chunk_start: int = 0
    # Split prefill only: per-depth written frontiers, see prefill_boundaries.
    running_depth_boundary: list[list[int]] = field(default_factory=list, repr=False)
    # Host progress; in async mode an event confirms submitted GPU work is done.
    loops_done: int = 0
    pending_exit_depth: int | None = None
    admission_bypasses: int = 0
    remaining_probability: float = 1.0
    hidden_state: torch.Tensor | None = field(default=None, repr=False)
    finish_reason: FinishReason | None = None
    exit_trace: tuple[int, ...] = field(default=(), repr=False)

    @staticmethod
    def prefill_boundaries(
        num_tokens: int, total_depths: int, *, start: int = 0
    ) -> list[list[int]]:
        """Per-depth written frontiers of a freshly admitted split-prefill prompt.

        ``running_depth_boundary[d] == [X[d + 1], X[d])`` holds the prompt tokens
        that have completed exactly ``d`` depths, where ``X[d]`` counts the tokens
        that completed at least ``d`` depths. A new request has ``X[0]`` at the
        full prompt and every deeper frontier at zero. Advancing a run
        ``[left, right)`` from depth ``d`` to ``d + 1`` sets ``X[d + 1] = right``:
        the run leaves ``boundary[d]`` and extends the right edge of
        ``boundary[d + 1]``. A run may start only at ``X[d + 1]`` so no depth
        frontier ever has a hole. Every interval is empty once prefill completes,
        and ``num_prefilled_tokens`` then holds the finished prefix.
        """
        if num_tokens < 1 or total_depths < 1 or not 0 <= start <= num_tokens:
            raise ValueError(
                "prefill boundaries require positive sizes and start in [0, num_tokens]"
            )
        return [[start, num_tokens]] + [[start, start] for _ in range(total_depths - 1)]

    def prefill_depth(self, token_index: int) -> int:
        """Completed recurrence depth of the prompt token at ``token_index``."""
        for depth, (left, right) in enumerate(self.running_depth_boundary):
            if left <= token_index < right:
                return depth
        raise RuntimeError(
            f"request {self.request_id!r} has no prefill depth for token {token_index}"
        )

    @property
    def num_scheduled_outputs(self) -> int:
        return len(self.generated_token_ids) + self.num_output_placeholders

    @property
    def position(self) -> int:
        return len(self.prompt_token_ids) + self.num_scheduled_outputs - 1

    @property
    def input_token_id(self) -> int:
        return self.generated_token_ids[-1]


@dataclass(frozen=True)
class RequestOutput:
    request_id: str
    prompt_token_ids: list[int]
    token_ids: list[int]
    exit_depths: list[int]
    finished: bool
    finish_reason: str | None = None
    text: str = ""

    @classmethod
    def from_request(cls, request: Request):
        return cls(
            request_id=request.request_id,
            prompt_token_ids=list(request.prompt_token_ids),
            token_ids=list(request.generated_token_ids),
            exit_depths=list(request.exit_depths),
            finished=request.stage == Stage.FINISHED,
            finish_reason=request.finish_reason.value
            if request.finish_reason is not None
            else None,
        )
