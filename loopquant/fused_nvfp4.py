"""Fixed-global NVFP4 producer with dynamic or shared fixed block16 scales."""

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class NVFP4Workspace:
    packed: torch.Tensor
    swizzled_scales: torch.Tensor

    @classmethod
    def allocate(cls, rows: int, columns: int, device: torch.device) -> "NVFP4Workspace":
        if rows < 1 or columns % 64:
            raise ValueError("native NVFP4 requires positive rows and K divisible by64")
        return cls(
            torch.empty(rows, columns // 2, dtype=torch.uint8, device=device),
            torch.empty(
                ((rows + 127) // 128 * 128) * (columns // 16),
                dtype=torch.float8_e4m3fn,
                device=device,
            ),
        )


class NVFP4InputQuantizer(nn.Module):
    """One producer launch; fixed scales are registered once, without per-call reads."""

    def __init__(
        self,
        global_scale: torch.Tensor,
        columns: int,
        *,
        block_scales: torch.Tensor | None = None,
        norm_epsilon: float = 1e-6,
    ) -> None:
        super().__init__()
        if columns < 64 or columns % 64 or global_scale.device.type != "cuda":
            raise ValueError("native NVFP4 requires CUDA and K divisible by64")
        if global_scale.numel() != 1 or not bool(
            torch.isfinite(global_scale).all() & (global_scale > 0).all()
        ):
            raise ValueError("expected a finite positive global dequantization scale")
        if block_scales is not None:
            if (
                block_scales.shape != (columns // 16,)
                or block_scales.dtype != torch.float8_e4m3fn
                or block_scales.device != global_scale.device
                or not bool(
                    torch.isfinite(block_scales.float()).all() & (block_scales.float() > 0).all()
                )
            ):
                raise ValueError("fixed block scales must be positive CUDA E4M3 values per block16")
            block_scales = block_scales.detach().clone().contiguous()
        self.register_buffer("global_scale", global_scale.detach().float().reshape(()).clone())
        self.register_buffer("block_scales", block_scales)
        self.columns, self.norm_epsilon = columns, norm_epsilon

    def forward(
        self,
        values: torch.Tensor,
        workspace: NVFP4Workspace,
        *,
        norm_weight: torch.Tensor | None = None,
    ) -> NVFP4Workspace:
        from .nvfp4_kernels import quantize_nvfp4_rows

        rows, columns = values.shape
        if (
            values.dtype not in (torch.bfloat16, torch.float16)
            or values.stride(-1) != 1
            or columns != self.columns
            or workspace.packed.shape != (rows, columns // 2)
        ):
            raise ValueError("producer inputs must match the registered native row bucket")
        quantize_nvfp4_rows[(((rows + 127) // 128) * 128,)](
            values,
            norm_weight if norm_weight is not None else values,
            self.global_scale,
            self.block_scales if self.block_scales is not None else values,
            workspace.packed,
            workspace.swizzled_scales,
            rows,
            columns,
            values.stride(0),
            norm_weight is not None,
            self.block_scales is not None,
            self.norm_epsilon,
            1 << (columns - 1).bit_length(),
            num_warps=4 if columns <= 4096 else 8,
            enable_fp_fusion=False,
        )
        return workspace
