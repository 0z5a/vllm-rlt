"""Fuse the RMSNorm epilogue while retaining PyTorch's variance reduction."""

import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_epilogue(
    X,
    WEIGHT,
    VARIANCE,
    OUT,
    ROW_STRIDE: tl.constexpr,
    WIDTH: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    columns = tl.arange(0, BLOCK)
    value = tl.load(X + row * ROW_STRIDE + columns, columns < WIDTH, other=0).to(tl.float32)
    variance = tl.load(VARIANCE + row)
    # Preserve both rounding boundaries: FP32 normalization, then BF16 before
    # multiplying by the learned weight. Combining the reduction changes logits
    # after many recurrent steps, even when the sampled tokens still agree.
    normalized = (value * tl.rsqrt(variance + EPS)).to(X.dtype.element_ty).to(tl.float32)
    weight = tl.load(WEIGHT + columns, columns < WIDTH, other=0).to(tl.float32)
    tl.store(OUT + row * WIDTH + columns, normalized * weight, columns < WIDTH)


def rmsnorm_epilogue(hidden, weight, eps):
    variance = hidden.float().square().mean(-1)
    output = torch.empty_like(hidden, memory_format=torch.contiguous_format)
    _rmsnorm_epilogue[(hidden.shape[0],)](
        hidden,
        weight,
        variance,
        output,
        hidden.stride(0),
        hidden.shape[1],
        eps,
        triton.next_power_of_2(hidden.shape[1]),
        num_warps=4,
        enable_fp_fusion=False,
    )
    return output
