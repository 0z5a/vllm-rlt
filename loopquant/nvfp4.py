"""NVFP4 reference encoding: E2M1 payload, block16 E4M3 scales and FP32 global scale.

These ordinary Torch operations establish numerical/ABI semantics only. They
are not a fused deployment quantizer or a measured performance baseline.
"""

from dataclasses import dataclass

import torch
from torch.nn import functional as F


def pack_e2m1(values: torch.Tensor) -> torch.Tensor:
    """Round to nearest even E2M1, saturate to six, and pack low nibble first."""
    if values.ndim != 2 or values.shape[1] % 2:
        raise ValueError("E2M1 packing requires an even-width matrix")
    magnitude = values.float().abs().contiguous()
    edges = values.new_tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], dtype=torch.float32)
    code = torch.bucketize(magnitude, edges).to(torch.uint8)
    edge = edges[code.long().clamp_max(6)]
    code += ((magnitude == edge) & (code % 2 == 1)).to(torch.uint8)
    code |= torch.signbit(values).to(torch.uint8) << 3
    return (code[:, 0::2] | (code[:, 1::2] << 4)).view(torch.float4_e2m1fn_x2)


def unpack_e2m1(packed: torch.Tensor) -> torch.Tensor:
    raw = packed.view(torch.uint8)
    code = torch.stack((raw & 15, raw >> 4), -1).flatten(1)
    magnitude = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device=packed.device)
    value = magnitude[(code & 7).long()]
    return torch.where(code >= 8, -value, value)


def swizzle_block_scales(scales: torch.Tensor) -> torch.Tensor:
    """cuBLAS SWIZZLE_32_4_4, including every required padded scale slot."""
    rows, blocks = scales.shape
    padded_rows, padded_blocks = (rows + 127) // 128 * 128, (blocks + 3) // 4 * 4
    padded = torch.zeros(padded_rows, padded_blocks, device=scales.device, dtype=scales.dtype)
    padded[:rows, :blocks] = scales
    return (
        padded.reshape(padded_rows // 128, 4, 32, padded_blocks // 4, 4)
        .permute(0, 3, 2, 1, 4)
        .contiguous()
        .flatten()
    )


@dataclass(frozen=True)
class NVFP4Tensor:
    packed: torch.Tensor
    block_scales: torch.Tensor
    global_scale: torch.Tensor
    columns: int

    def dequantize(self) -> torch.Tensor:
        values = unpack_e2m1(self.packed)
        return (
            values * self.block_scales.float().repeat_interleave(16, dim=1) * self.global_scale
        )[:, : self.columns]


def quantize_nvfp4(
    values: torch.Tensor,
    *,
    global_scale: torch.Tensor | None = None,
    block_scales: torch.Tensor | None = None,
) -> NVFP4Tensor:
    """Optional fixed scales separate static-global and static-local experiments."""
    if values.ndim != 2 or not torch.isfinite(values).all():
        raise ValueError("NVFP4 reference expects a finite matrix")
    columns = values.shape[1]
    padded = F.pad(values.float(), (0, (-columns) % 16))
    blocks = padded.reshape(values.shape[0], -1, 16)
    if global_scale is None:
        global_scale = blocks.abs().amax().clamp_min(1e-12) / (6 * 448)
    if global_scale.numel() != 1 or not torch.isfinite(global_scale).all() or global_scale <= 0:
        raise ValueError("global dequant scale must be a finite positive scalar")
    if block_scales is None:
        local = (blocks.abs().amax(-1) / (6 * global_scale)).clamp(2**-9, 448)
        block_scales = local.to(torch.float8_e4m3fn)
    if block_scales.shape != blocks.shape[:2] or block_scales.dtype != torch.float8_e4m3fn:
        raise ValueError("local scales must be one E4M3 value per row and block16")
    decoded_scales = block_scales.float() * global_scale
    if not torch.isfinite(decoded_scales).all() or not (decoded_scales > 0).all():
        raise ValueError("local dequant scales must be finite and positive")
    scaled = (blocks / decoded_scales.unsqueeze(-1)).flatten(1)
    return NVFP4Tensor(pack_e2m1(scaled), block_scales, global_scale.float(), columns)
