"""Explicit FP8 scratch buffers and fused producers for native rowwise GEMM."""

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

from .quantizers import ScaleLayout, fp8_encode


@dataclass
class FP8Workspace:
    """Caller-owned scratch; consume all GEMMs before reusing it on the stream."""

    packed: torch.Tensor
    row_scales: torch.Tensor

    @classmethod
    def allocate(cls, rows: int, columns: int, device: torch.device) -> "FP8Workspace":
        padded = (rows + 15) // 16 * 16
        return cls(
            torch.empty(padded, columns, device=device, dtype=torch.float8_e4m3fn),
            torch.empty(padded, 1, device=device, dtype=torch.float32),
        )


class FP8InputQuantizer(nn.Module):
    """One launch for pad, optional RMSNorm, scale selection/reduction and E4M3 cast.

    Mixed-loop selection consumes explicit GPU loop IDs. The caller owns their
    validity and graph replay updates; no Python forward-count scale selection.
    """

    def __init__(
        self,
        scales: torch.Tensor,
        layout: ScaleLayout,
        mode: Literal["static", "dynamic"],
        *,
        norm_epsilon: float = 1e-6,
    ) -> None:
        super().__init__()
        if mode not in ("static", "dynamic") or scales.shape != (layout.stages,):
            raise ValueError("expected static/dynamic mode and one scale per stage")
        if scales.device.type != "cuda" or not bool(
            torch.isfinite(scales).all() & (scales > 0).all()
        ):
            raise ValueError("native activation scales must be finite positive CUDA values")
        self.register_buffer("scales", scales.detach().float().clone())
        self.layout, self.mode, self.norm_epsilon = layout, mode, norm_epsilon

    def forward(
        self,
        values: torch.Tensor,
        workspace: FP8Workspace,
        *,
        norm_weight: torch.Tensor | None = None,
        loop_ids: torch.Tensor | None = None,
    ) -> FP8Workspace:
        from .fp8_kernels import quantize_rows

        rows, columns = values.shape
        if values.dtype not in (torch.bfloat16, torch.float16) or values.stride(-1) != 1:
            raise ValueError("quantization expects packed BF16/FP16 rows")
        if workspace.packed.shape != ((rows + 15) // 16 * 16, columns):
            raise ValueError("scratch shape must match the registered padded row bucket")
        if self.mode == "static" and self.layout.stages > 1 and loop_ids is None:
            raise ValueError("staged scales require the prepared GPU row loop IDs")
        quantize_rows[(workspace.packed.shape[0],)](
            values,
            norm_weight if norm_weight is not None else values,
            loop_ids if loop_ids is not None else values,
            self.scales,
            workspace.packed,
            workspace.row_scales,
            rows,
            columns,
            values.stride(0),
            self.mode == "dynamic",
            norm_weight is not None,
            self.norm_epsilon,
            self.layout.boundaries,
            1 << (columns - 1).bit_length(),
            num_warps=4 if columns <= 4096 else 8,
            enable_fp_fusion=False,
        )
        return workspace


class RowwiseFP8Linear(nn.Module):
    """One column-major packed W; row scales support DYN and static loop policies."""

    def __init__(self, linear: nn.Linear) -> None:
        super().__init__()
        if linear.weight.device.type != "cuda" or linear.bias is not None:
            raise ValueError("native FP8 requires CUDA and bias-free weights")
        if linear.in_features % 16 or linear.out_features % 16:
            raise ValueError("native FP8 requires K and N divisible by 16")
        weight = linear.weight.detach().float()
        scale = weight.abs().amax().clamp_min(torch.finfo(torch.float32).tiny) / 448
        self.register_buffer("packed_weight", fp8_encode(weight, scale).t())
        # CUTLASS rowwise ABI requires a contiguous N-vector, even for equal values.
        self.register_buffer("weight_scales", scale.expand(1, linear.out_features).contiguous())

    def forward(
        self, activation: FP8Workspace, rows: int, *, dtype: torch.dtype = torch.bfloat16
    ) -> torch.Tensor:
        return torch._scaled_mm(
            activation.packed,
            self.packed_weight,
            scale_a=activation.row_scales,
            scale_b=self.weight_scales,
            out_dtype=dtype,
            use_fast_accum=False,
        )[:rows]
