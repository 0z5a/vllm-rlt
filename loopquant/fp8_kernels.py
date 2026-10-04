"""Fused row quantization; GEMM remains the installed Torch/CUTLASS operation."""

import triton
import triton.language as tl


@triton.jit
def quantize_rows(
    X,
    Gain,
    LoopIds,
    Scales,
    Packed,
    RowScales,
    ROWS: tl.constexpr,
    K: tl.constexpr,
    STRIDE: tl.constexpr,
    DYNAMIC: tl.constexpr,
    NORM: tl.constexpr,
    EPS: tl.constexpr,
    BOUNDARIES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    values = tl.load(X + row * STRIDE + cols, (row < ROWS) & (cols < K), 0)
    dtype: tl.constexpr = values.dtype
    x = values.to(tl.float32)
    if NORM:
        inverse = tl.rsqrt(tl.sum(x * x, 0) / K + EPS)
        normalized = (x * inverse).to(dtype).to(tl.float32)
        gain = tl.load(Gain + cols, cols < K, 0).to(tl.float32)
        # Match the protected BF16 norm boundary, including its two roundings.
        x = (normalized * gain).to(dtype).to(tl.float32)
    if DYNAMIC:
        # Torch's scalar division uses the rounded FP32 reciprocal product.
        scale = tl.maximum(tl.max(tl.abs(x), 0), 1.0e-12) * (1.0 / 448.0)
    else:
        stage = tl.full((), 0, tl.int32)
        if len(BOUNDARIES):
            loop = tl.load(LoopIds + row, row < ROWS, 0)
            for boundary in tl.static_range(len(BOUNDARIES)):
                stage += (loop >= BOUNDARIES[boundary]).to(tl.int32)
        scale = tl.load(Scales + stage)
    encoded = tl.minimum(tl.maximum(tl.div_rn(x, scale), -448.0), 448.0)
    tl.store(Packed + row * K + cols, encoded, cols < K)
    tl.store(RowScales + row, scale)
