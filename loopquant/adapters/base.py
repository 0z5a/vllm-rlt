"""Shared quantized projections and explicit-loop statistics for dense adapters."""

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

from ..quantizers import FP8FakeLinear
from ..stats import ActivationStats


@dataclass
class LoopOutput:
    logits: torch.Tensor
    states: list[torch.Tensor]


class QuantizedProjections(nn.Module):
    def __init__(self, attention_backend: Literal["eager", "sdpa"] = "eager") -> None:
        super().__init__()
        if attention_backend not in ("eager", "sdpa"):
            raise ValueError("reference attention must be explicitly eager or sdpa")
        self.attention_backend = attention_backend
        self.quantized = nn.ModuleDict()
        self.statistics: dict[tuple[str, int], ActivationStats] = {}
        self.collect = False

    def _linear(
        self, name: str, module: nn.Linear, values: torch.Tensor, loop: int, valid: torch.Tensor
    ) -> torch.Tensor:
        if self.collect:
            if (name, loop) not in self.statistics:
                self.statistics[name, loop] = ActivationStats()
            self.statistics[name, loop].update(values.flatten(0, 1), valid.flatten())
        key = name.replace(".", "__")
        if key in self.quantized:
            loops = torch.full(values.shape[:-1], loop, device=values.device, dtype=torch.long)
            return self.quantized[key](values, loops)
        return module(values)

    def attach(self, name: str, quantized: FP8FakeLinear) -> None:
        self.quantized[name.replace(".", "__")] = quantized
