"""Bounded, reproducible activation summaries keyed by explicit loop metadata."""

import math
from dataclasses import dataclass, field

import torch


@dataclass
class ActivationStats:
    sample_limit: int = 4096
    seed: int = 17
    rows: int = 0
    elements: int = 0
    sum_squares: float = 0
    amax: float = 0
    samples: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    priorities: torch.Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.float64))
    channel_sum_squares: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    generator: torch.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.sample_limit < 1:
            raise ValueError("sample limit must be positive")
        self.generator = torch.Generator().manual_seed(self.seed)

    def update(self, values: torch.Tensor, valid: torch.Tensor) -> None:
        x = values.detach()[valid].float()
        if x.numel() == 0:
            return
        if not torch.isfinite(x).all():
            raise ValueError("nonfinite calibration activation")
        energy = x.square().sum(0).cpu().double()
        if self.channel_sum_squares.numel() == 0:
            self.channel_sum_squares = torch.zeros_like(energy)
        self.channel_sum_squares += energy
        self.rows += x.shape[0]
        self.elements += x.numel()
        self.sum_squares += float(energy.sum())
        self.amax = max(self.amax, float(x.abs().max()))
        # Uniform random priorities give every observed element the same inclusion chance.
        # Transfer at most sample_limit observations per update from the GPU.
        priority = torch.rand(x.numel(), generator=self.generator, dtype=torch.float64)
        selected = priority.topk(min(self.sample_limit, x.numel())).indices
        samples = x.flatten()[selected.to(x.device)].abs().cpu()
        priorities = torch.cat((self.priorities, priority[selected]))
        samples = torch.cat((self.samples, samples))
        keep = priorities.topk(min(self.sample_limit, len(priorities))).indices
        self.priorities, self.samples = priorities[keep], samples[keep]

    def summary(self) -> dict[str, int | float | list[float]]:
        if not self.elements:
            raise ValueError("cannot summarize an empty activation stream")
        return {
            "rows": self.rows,
            "elements": self.elements,
            "rms": math.sqrt(self.sum_squares / self.elements),
            "amax": self.amax,
            "p99": float(self.samples.quantile(0.99)),
            "p999": float(self.samples.quantile(0.999)),
            "sample_count": self.samples.numel(),
            "channel_energy": (self.channel_sum_squares / self.rows).tolist(),
        }
