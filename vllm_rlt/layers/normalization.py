# SPDX-License-Identifier: Apache-2.0
"""Shared normalization layers across models."""

import torch
from torch import nn

_FUSED = None


def _fused_rms_norm():
    """The single-launch CUDA operator, or ``False`` where Triton is unavailable."""
    global _FUSED
    if _FUSED is None:
        try:
            from vllm_rlt.kernels import rmsnorm
        except ImportError:
            _FUSED = False
        else:
            _FUSED = rmsnorm
    return _FUSED


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Retains FP32 precision during reduction to preserve numerical stability
    across all input activation dtypes (BF16/FP16/FP32).

    Inference on CUDA runs as one kernel launch (``kernels.rmsnorm``); the
    composite operators below are the reference and serve every other case.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        weight = self.weight
        if (
            hidden.is_cuda
            and hidden.dtype == weight.dtype
            and hidden.device == weight.device
            and not torch.is_grad_enabled()
            and hidden.is_contiguous()
        ):
            fused = _fused_rms_norm()
            if (
                fused
                and hidden.ndim
                and hidden.dtype in fused.DTYPES
                and 0 < hidden.shape[-1] <= fused.MAX_WIDTH
            ):
                return fused.rms_norm(hidden, weight, self.variance_epsilon)
        return self.reference(hidden)

    def reference(self, hidden: torch.Tensor) -> torch.Tensor:
        """The composite operators: every device and dtype, and under autograd."""
        value = hidden.float()
        value = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + self.variance_epsilon)
        return self.weight * value.to(hidden.dtype)
