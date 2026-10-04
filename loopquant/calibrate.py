"""Calibration policies change scale binding only; weight bytes are held fixed."""

from collections import defaultdict

import torch
from torch import nn

from .quantizers import ScaleLayout
from .stats import ActivationStats


def activation_scales(
    statistics: dict[tuple[str, int], ActivationStats],
    layout: ScaleLayout,
    percentile: float,
) -> dict[str, torch.Tensor]:
    if not 0 < percentile <= 100:
        raise ValueError("percentile must be in (0, 100]")
    grouped: dict[str, dict[int, list[ActivationStats]]] = defaultdict(lambda: defaultdict(list))
    for (module, loop), stats in statistics.items():
        if loop not in range(layout.max_loops):
            raise ValueError("observed loop outside registered range")
        stage = sum(loop >= boundary for boundary in layout.boundaries)
        grouped[module][stage].append(stats)
    result = {}
    for module, stages in grouped.items():
        if set(stages) != set(range(layout.stages)):
            raise ValueError(f"missing calibration stage for {module}")
        values = []
        for stage in range(layout.stages):
            if percentile == 100:
                bound = max(stats.amax for stats in stages[stage])
            else:
                # Each loop gets an equal reservoir budget; this is explicitly loop-balanced.
                sample = torch.cat([stats.samples for stats in stages[stage]])
                bound = float(sample.quantile(percentile / 100))
            values.append(max(bound / 448, torch.finfo(torch.float32).tiny))
        result[module] = torch.tensor(values, dtype=torch.float32)
    return result


@torch.no_grad()
def fold_equalization(norm: nn.Module, consumers: list[nn.Linear], diagonal: torch.Tensor) -> None:
    """Caller supplies every consumer of the same norm; D is shared across loops."""
    if diagonal.ndim != 1 or not torch.isfinite(diagonal).all() or not (diagonal > 0).all():
        raise ValueError("equalization requires a positive finite channel diagonal")
    parameters = dict(norm.named_parameters(recurse=False))
    weight = parameters["weight"]
    if weight.shape != diagonal.shape or not consumers:
        raise ValueError("norm gain and all consumer channels must match")
    if any(layer.in_features != len(diagonal) for layer in consumers):
        raise ValueError("consumer input channels must match the norm")
    weight.div_(diagonal.to(weight))
    for layer in consumers:
        layer.weight.mul_(diagonal.to(layer.weight))
