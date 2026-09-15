from dataclasses import dataclass, field
from enum import Enum

import torch

from vllm_lt.sampling_params import SamplingParams


class Stage(str, Enum):
    WAITING = "waiting"
    PREFILL = "prefill"
    PRELUDE = "prelude"
    RECURRENT = "recurrent"
    CODA = "coda"
    FINISHED = "finished"


@dataclass
class Request:
    request_id: str
    prompt_token_ids: list[int]
    sampling_params: SamplingParams
    stage: Stage = Stage.WAITING
    generated_token_ids: list[int] = field(default_factory=list)
    exit_depths: list[int] = field(default_factory=list)
    num_prefilled_tokens: int = 0
    # Host progress; in async mode an event confirms submitted GPU work is done.
    loops_done: int = 0
    pending_exit_depth: int | None = None
    admission_bypasses: int = 0
    remaining_probability: float = 1.0
    hidden_state: torch.Tensor | None = field(default=None, repr=False)
    generator: torch.Generator | None = field(default=None, repr=False)
    finish_reason: str | None = None

    @property
    def position(self) -> int:
        return len(self.prompt_token_ids) + len(self.generated_token_ids) - 1

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
            finish_reason=request.finish_reason,
        )
