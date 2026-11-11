# SPDX-License-Identifier: Apache-2.0
"""RMSNorm as one kernel launch.

The composite module issues eight operators per call and a looped block calls
it four times, so without CUDA graphs the normalization alone is most of the
operator launches of a decode round. One program normalizes one row.
"""

import torch
import triton
import triton.language as tl

from vllm_rlt.kernels.sparse_decode import _launch

# One program loads its whole row; wider rows keep the composite operators.
MAX_WIDTH = 16384
DTYPES = frozenset((torch.float32, torch.float16, torch.bfloat16))


@triton.jit
def _rms_norm_kernel(X, W, OUT, eps, H: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    x = tl.load(X + row * H + cols, mask=cols < H, other=0).to(tl.float32)
    scale = tl.rsqrt(tl.sum(x * x, axis=0) / H + eps)
    # As the module: normalize in FP32, round to the activation dtype, then scale.
    value = (x * scale).to(X.dtype.element_ty).to(tl.float32)
    weight = tl.load(W + cols, mask=cols < H, other=0).to(tl.float32)
    # The product of two activation-dtype values is exact in FP32, so this
    # rounds once, as the module's multiply in the activation dtype does.
    tl.store(OUT + row * H + cols, (weight * value).to(OUT.dtype.element_ty), mask=cols < H)


def rms_norm(hidden: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """``weight * (hidden / rms(hidden))`` over the last dimension, in new storage.

    ``hidden`` is contiguous, on the CUDA device of ``weight`` and of its dtype
    (one of ``DTYPES``), at most ``MAX_WIDTH`` wide. The mean of squares is
    reduced in FP32 in a different order than the composite module, so results
    agree with it to the last bits of the activation dtype, not bit for bit.
    """
    out = torch.empty_like(hidden)
    width = hidden.shape[-1]
    rows = hidden.numel() // width
    if rows:
        _launch(
            _rms_norm_kernel,
            (rows,),
            hidden,
            weight,
            out,
            eps,
            H=width,
            BLOCK=triton.next_power_of_2(width),
            num_warps=8,
        )
    return out
