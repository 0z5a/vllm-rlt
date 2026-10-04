"""Reference encodings and deterministic fake quantization; never a speed backend."""

import math
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ScaleLayout:
    """Absolute zero-based loop intervals; boundaries start the next stage."""

    max_loops: int
    boundaries: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.max_loops < 1 or tuple(sorted(set(self.boundaries))) != self.boundaries:
            raise ValueError("loop count must be positive and boundaries strictly increasing")
        if self.boundaries and not 0 < self.boundaries[0] <= self.boundaries[-1] < self.max_loops:
            raise ValueError("boundaries must be inside the loop range")

    @property
    def stages(self) -> int:
        return len(self.boundaries) + 1

    def indices(self, loop_ids: torch.Tensor) -> torch.Tensor:
        edges = torch.tensor(self.boundaries, device=loop_ids.device, dtype=loop_ids.dtype)
        return torch.bucketize(loop_ids, edges, right=True)


def fp8_encode(values: torch.Tensor, dequant_scale: torch.Tensor) -> torch.Tensor:
    return (values.float() / dequant_scale).clamp(-448, 448).to(torch.float8_e4m3fn)


def fp8_fake_quant(values: torch.Tensor, dequant_scale: torch.Tensor) -> torch.Tensor:
    scaled = values.float() / dequant_scale
    clipped = scaled.clamp(-448, 448)
    encoded = clipped.to(torch.float8_e4m3fn).float()
    rounded = clipped + (encoded - clipped).detach()
    return (rounded * dequant_scale).to(values.dtype)


def int4_pack(weight: torch.Tensor, group_size: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric signed INT4, low nibble first; last input group is zero padded."""
    if weight.ndim != 2 or group_size < 2 or group_size % 2:
        raise ValueError("expected a matrix and a positive even group size")
    rows, columns = weight.shape
    groups = math.ceil(columns / group_size)
    padded = nn.functional.pad(weight.float(), (0, groups * group_size - columns))
    grouped = padded.reshape(rows, groups, group_size)
    scale = grouped.abs().amax(-1).clamp_min(torch.finfo(torch.float32).tiny) / 7
    quantized = (grouped / scale.unsqueeze(-1)).round().clamp(-7, 7).to(torch.int8)
    nibbles = quantized.reshape(rows, -1).to(torch.uint8) & 15
    return nibbles[:, 0::2] | (nibbles[:, 1::2] << 4), scale


def int4_unpack(packed: torch.Tensor, scale: torch.Tensor, columns: int) -> torch.Tensor:
    nibbles = torch.stack((packed & 15, packed >> 4), -1).flatten(1).to(torch.int8)
    signed = torch.where(nibbles >= 8, nibbles - 16, nibbles)
    grouped = signed.reshape(scale.shape[0], scale.shape[1], -1)
    return (grouped.float() * scale.unsqueeze(-1)).flatten(1)[:, :columns]


class FP8FakeLinear(nn.Module):
    """One immutable FP8 weight matrix; scales may vary by loop for Q0 diagnosis."""

    def __init__(self, linear: nn.Linear, scales: torch.Tensor, layout: ScaleLayout) -> None:
        super().__init__()
        if linear.bias is not None or scales.shape != (layout.stages,):
            raise ValueError("expected bias-free linear and one activation scale per stage")
        if not torch.isfinite(scales).all() or not (scales > 0).all():
            raise ValueError("scales must be finite and positive")
        weight = linear.weight.detach().float()
        weight_scale = weight.abs().amax().clamp_min(torch.finfo(torch.float32).tiny) / 448
        self.register_buffer("packed_weight", fp8_encode(weight, weight_scale))
        self.register_buffer("weight_scale", weight_scale)
        # Training/reference storage only; never serialize a second weight payload.
        # Reuse it across recurrence so autograd does not retain R decoded copies.
        self.register_buffer(
            "reference_weight", self.packed_weight.float() * self.weight_scale, persistent=False
        )
        self.register_load_state_dict_post_hook(self._refresh_reference)
        self.log_scale = nn.Parameter(scales.detach().float().log())
        self.layout = layout

    def _refresh_reference(self, module: nn.Module, incompatible_keys: object) -> None:
        self.reference_weight.copy_(self.packed_weight.float() * self.weight_scale)

    def forward(self, values: torch.Tensor, loop_ids: torch.Tensor) -> torch.Tensor:
        scale = self.log_scale.exp()[self.layout.indices(loop_ids)].unsqueeze(-1)
        # Decode in FP32 so the reference does not insert BF16 rounding before
        # the native FP8 GEMM's accumulation and final output conversion.
        activation = fp8_fake_quant(values.float(), scale)
        return nn.functional.linear(activation, self.reference_weight).to(values.dtype)
