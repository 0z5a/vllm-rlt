"""Signed groupwise INT4 through Torch's installed BF16 tinygemm kernel.

Its execution ABI rounds group scales to BF16 and uses unsigned high-nibble-first
codes, unlike our signed low-nibble-first portable checkpoint representation.
"""

import torch
from torch import nn
from torch.nn import functional as F

from .quantizers import int4_pack


def tinygemm_inputs(
    packed: torch.Tensor, scales: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    if packed.ndim != 2 or packed.dtype != torch.uint8 or scales.ndim != 2:
        raise ValueError("expected portable INT4 bytes and per-output group scales")
    if packed.shape[0] != scales.shape[0] or packed.device != scales.device:
        raise ValueError("weight codes and scales must agree on outputs and device")
    offset = packed ^ 0x88
    unsigned = (offset << 4) | (offset >> 4)
    rounded = scales.to(torch.bfloat16)
    if not bool(torch.isfinite(rounded).all() & (rounded > 0).all()):
        raise ValueError("native BF16 group scales must be finite and positive")
    parameters = torch.stack((rounded, torch.zeros_like(rounded)), -1).transpose(0, 1).contiguous()
    return unsigned.contiguous(), parameters


class NativeInt4Linear(nn.Module):
    def __init__(
        self, packed: torch.Tensor, scales: torch.Tensor, columns: int, group_size: int = 128
    ) -> None:
        super().__init__()
        if packed.device.type != "cuda" or group_size not in (32, 64, 128, 256):
            raise ValueError("native INT4 requires CUDA and a supported group size")
        padded = scales.shape[1] * group_size
        if packed.shape != (scales.shape[0], padded // 2) or not 0 < columns <= padded:
            raise ValueError("portable weight bytes do not match the group layout")
        if packed.shape[0] % 8:
            raise ValueError("native INT4 output width must be divisible by eight")
        source, parameters = tinygemm_inputs(packed, scales)
        self.register_buffer("packed_weight", torch._convert_weight_to_int4pack(source, 2))
        self.register_buffer("scales_and_zeros", parameters)
        self.columns, self.padded_columns, self.group_size = columns, padded, group_size

    @classmethod
    def from_linear(cls, linear: nn.Linear, group_size: int = 128) -> "NativeInt4Linear":
        if linear.bias is not None:
            raise ValueError("native INT4 supports bias-free core projections")
        packed, scales = int4_pack(linear.weight.detach(), group_size)
        return cls(packed, scales, linear.in_features, group_size)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.dtype != torch.bfloat16 or values.shape[-1] != self.columns:
            raise ValueError("native INT4 expects BF16 inputs with the registered width")
        rows = values.reshape(-1, self.columns)
        if self.columns != self.padded_columns:
            rows = F.pad(rows, (0, self.padded_columns - self.columns))
        result = torch._weight_int4pack_mm(
            rows.contiguous(), self.packed_weight, self.group_size, self.scales_and_zeros
        )
        return result.reshape(*values.shape[:-1], -1)
