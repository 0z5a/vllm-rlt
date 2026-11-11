# SPDX-License-Identifier: Apache-2.0
"""Shared rotary position embedding (RoPE) layers across models."""

import torch
from torch import nn

from vllm_rlt.kernels import single_launch_ops


def rotate_half(value: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dimensions of the input tensor."""
    first, second = value.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Applies Rotary Position Embedding (RoPE) to Query and Key states.

    CUDA inference on packed ``[rows, heads, head_dim]`` states in a 16-bit
    dtype runs as one kernel launch with bit-identical results.
    """
    if q.is_cuda and q.ndim == 3 and k.ndim == 3 and not torch.is_grad_enabled():
        ops = single_launch_ops()
        rows, heads, dim = q.shape
        tail = (dim, 1)
        if (
            ops
            and q.dtype in ops.ROUNDED
            and q.dtype == k.dtype == cos.dtype == sin.dtype
            and k.shape[0] == rows
            and k.shape[2] == dim
            and cos.shape == sin.shape
            and cos.shape in ((rows, 1, dim), (rows, dim))
            and cos.is_contiguous()
            and sin.is_contiguous()
            and q.stride()[1:] == tail
            and k.stride()[1:] == tail
            and q.device == k.device == cos.device == sin.device
            and ops.rope_supported(heads, k.shape[1], dim)
        ):
            return ops.rope_qk(q, k, cos, sin, q_heads=heads, k_heads=k.shape[1])
    q_embed = q * cos + rotate_half(q) * sin
    k_embed = k * cos + rotate_half(k) * sin
    return q_embed, k_embed


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding with precision-preserving buffer management.

    Accepts `head_dim` and `rope_theta` directly without depending on a specific
    model configuration dataclass.
    """

    def __init__(self, head_dim: int, rope_theta: float = 10000.0) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.rope_theta = rope_theta
        self.register_buffer("inv_freq", self.frequencies(), persistent=False)

    def frequencies(self, device: torch.device | str | None = None) -> torch.Tensor:
        return 1.0 / (
            self.rope_theta
            ** (
                torch.arange(0, self.head_dim, 2, dtype=torch.float32, device=device)
                / self.head_dim
            )
        )

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse=recurse)
        # Module.to(dtype) otherwise rounds these constants before forward's
        # float32 cast, changing long-context rotary angles irreversibly.
        self.inv_freq = self.frequencies(device=self.inv_freq.device)
        return self

    def forward(
        self, hidden: torch.Tensor, positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Force FP32 calculation to maintain accuracy regardless of model dtype
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            angles = positions.float().unsqueeze(-1) * self.inv_freq.float()
            angles = torch.cat((angles, angles), dim=-1)
        return (
            angles.cos().to(hidden.dtype).unsqueeze(1),
            angles.sin().to(hidden.dtype).unsqueeze(1),
        )
