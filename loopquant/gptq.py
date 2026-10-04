"""Shared-module GPTQ with fixed signed-INT4 groups and explicit loop row counts.

Uses the error-compensation algorithm of Frantar et al., arXiv:2210.17323.
This is numerical packing, not a native W4A16 execution backend.
"""

import math
from dataclasses import dataclass, field

import torch
from torch.nn import functional as F


@dataclass
class HessianStats:
    """One physical module's row-weighted Gram matrix; padding is excluded."""

    gram: torch.Tensor
    loop_rows: dict[int, int] = field(default_factory=dict)
    loop_energy: dict[int, float] = field(default_factory=dict)

    @classmethod
    def create(cls, channels: int, device: torch.device) -> "HessianStats":
        return cls(torch.zeros(channels, channels, device=device, dtype=torch.float32))

    @torch.no_grad()
    def add(self, values: torch.Tensor, valid: torch.Tensor, loop: int) -> None:
        rows = values.reshape(-1, values.shape[-1])[valid.flatten()].float()
        if rows.shape[1] != self.gram.shape[0] or loop < 0:
            raise ValueError("activation channels or loop index do not match")
        self.gram.addmm_(rows.T, rows)
        self.loop_rows[loop] = self.loop_rows.get(loop, 0) + rows.shape[0]
        self.loop_energy[loop] = self.loop_energy.get(loop, 0.0) + float(rows.square().sum())

    def matrix(self) -> torch.Tensor:
        count = sum(self.loop_rows.values())
        if not count:
            raise ValueError("Hessian requires at least one valid activation row")
        return self.gram * (2 / count)


@torch.no_grad()
def gptq_int4(
    weight: torch.Tensor,
    hessian: torch.Tensor,
    *,
    group_size: int = 128,
    block_size: int = 128,
    damping: float = 0.01,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Static groups, original column order, one packed matrix for all loops.

    Group scales are fixed from the original weights, matching RTN's signed
    [-7,7] encoding. Only error compensation changes. Diagonal damping is a
    fixed fraction of the mean diagonal; failed factorization is not hidden.
    """
    if weight.ndim != 2 or group_size < 2 or group_size % 2 or block_size < 1:
        raise ValueError("expected a matrix, positive block size, and even group size")
    outputs, channels = weight.shape
    if hessian.shape != (channels, channels) or not 0 < damping < 1:
        raise ValueError("Hessian shape or damping outside the registered range")
    working = weight.float().clone()
    metric = hessian.float().clone()
    dead = metric.diagonal() == 0
    working[:, dead] = 0
    metric[dead, dead] = 1
    metric.diagonal().add_(damping * metric.diagonal().mean())
    inverse = torch.cholesky_inverse(torch.linalg.cholesky(metric))
    factor = torch.linalg.cholesky(inverse, upper=True)

    groups = math.ceil(channels / group_size)
    grouped = F.pad(working, (0, groups * group_size - channels)).reshape(
        outputs, groups, group_size
    )
    scales = grouped.abs().amax(-1).clamp_min(torch.finfo(torch.float32).tiny) / 7
    codes = torch.zeros(outputs, groups * group_size, dtype=torch.int8, device=weight.device)
    for start in range(0, channels, block_size):
        end = min(start + block_size, channels)
        block = working[:, start:end].clone()
        errors = torch.empty_like(block)
        for offset, column in enumerate(range(start, end)):
            scale = scales[:, column // group_size]
            code = (block[:, offset] / scale).round().clamp(-7, 7)
            codes[:, column] = code.to(torch.int8)
            error = (block[:, offset] - code * scale) / factor[column, column]
            errors[:, offset] = error
            block[:, offset:] -= error[:, None] * factor[column, column:end]
        working[:, end:] -= errors @ factor[start:end, end:]
    nibbles = codes.to(torch.uint8) & 15
    return nibbles[:, 0::2] | (nibbles[:, 1::2] << 4), scales
