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
    samples: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    priorities: torch.Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.float64))
    channel_sum_squares: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    generator: torch.Generator = field(init=False, repr=False)
    _peak: torch.Tensor = field(default_factory=lambda: torch.zeros(()), repr=False)
    _finite: torch.Tensor = field(default_factory=lambda: torch.tensor(True), repr=False)

    def __post_init__(self) -> None:
        if self.sample_limit < 1:
            raise ValueError("sample limit must be positive")
        self.generator = torch.Generator().manual_seed(self.seed)

    def update(self, values: torch.Tensor, valid: torch.Tensor) -> None:
        self._update(values.detach()[valid].float())

    def update_rows(self, values: torch.Tensor, indices: torch.Tensor) -> None:
        """Use prepared row indices, avoiding a CUDA boolean-index size query."""
        self._update(values.detach().index_select(0, indices).float())

    def _update(self, x: torch.Tensor) -> None:
        if x.numel() == 0:
            return
        energy = x.square().sum(0).double()
        if not self.elements:
            self.channel_sum_squares = torch.zeros_like(energy)
            self._peak = self._peak.to(x.device)
            self._finite = self._finite.to(x.device)
            self.samples = self.samples.to(x.device)
            self.priorities = self.priorities.to(x.device)
            self.generator = torch.Generator(device=x.device).manual_seed(self.seed)
        self._finite.logical_and_(torch.isfinite(x).all())
        if x.device.type == "cpu":
            self.validate()
        self.channel_sum_squares += energy
        self.rows += x.shape[0]
        self.elements += x.numel()
        torch.maximum(self._peak, x.abs().max(), out=self._peak)
        # Uniform random priorities give every observed element the same inclusion chance.
        # Device-local RNG streams are reproducible on that device, not across CPU/CUDA.
        priority = torch.rand(
            x.numel(), generator=self.generator, device=x.device, dtype=torch.float64
        )
        selected = priority.topk(min(self.sample_limit, x.numel())).indices
        samples = x.flatten()[selected].abs()
        priorities = torch.cat((self.priorities, priority[selected]))
        samples = torch.cat((self.samples, samples))
        keep = priorities.topk(min(self.sample_limit, len(priorities))).indices
        self.priorities, self.samples = priorities[keep], samples[keep]

    def validate(self) -> None:
        if not bool(self._finite):
            raise ValueError("nonfinite calibration activation")

    @property
    def amax(self) -> float:
        self.validate()
        return float(self._peak)

    @property
    def sum_squares(self) -> float:
        self.validate()
        return float(self.channel_sum_squares.sum())

    def summary(self) -> dict[str, int | float | str | list[float]]:
        if not self.elements:
            raise ValueError("cannot summarize an empty activation stream")
        self.validate()
        samples = self.samples.cpu()
        return {
            "rows": self.rows,
            "elements": self.elements,
            "rms": math.sqrt(self.sum_squares / self.elements),
            "amax": self.amax,
            "p99": float(samples.quantile(0.99)),
            "p999": float(samples.quantile(0.999)),
            "sample_count": self.samples.numel(),
            "channel_energy": (self.channel_sum_squares / self.rows).cpu().tolist(),
            "sample_seed": self.seed,
            "sample_device": str(self.samples.device),
        }
