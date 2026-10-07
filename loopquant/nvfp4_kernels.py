"""Fused NVFP4 producer; the GEMM stays in the installed Torch backend."""

import triton
import triton.language as tl


@triton.jit
def quantize_nvfp4_rows(
    X,
    Gain,
    GlobalScale,
    FixedBlocks,
    Packed,
    Swizzled,
    ROWS: tl.constexpr,
    K: tl.constexpr,
    STRIDE: tl.constexpr,
    NORM: tl.constexpr,
    STATIC_BLOCKS: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    columns = tl.arange(0, BLOCK)
    values = tl.load(X + row * STRIDE + columns, (row < ROWS) & (columns < K), 0)
    dtype: tl.constexpr = values.dtype
    x = values.to(tl.float32)
    if NORM:
        inverse = tl.rsqrt(tl.sum(x * x, 0) / K + EPS)
        normalized = (x * inverse).to(dtype).to(tl.float32)
        gain = tl.load(Gain + columns, columns < K, 0).to(tl.float32)
        x = (normalized * gain).to(dtype).to(tl.float32)
    blocks = tl.reshape(x, (BLOCK // 16, 16))
    block_ids = tl.arange(0, BLOCK // 16)
    global_scale = tl.load(GlobalScale)
    if STATIC_BLOCKS:
        scales = tl.load(FixedBlocks + block_ids, block_ids < K // 16, 1)
    else:
        local = tl.div_rn(tl.max(tl.abs(blocks), 1), 6.0 * global_scale)
        local = tl.minimum(tl.maximum(local, 2.0**-9), 448.0)
        scales = local.to(Swizzled.dtype.element_ty)
    decoded = scales.to(tl.float32) * global_scale
    scaled = tl.reshape(tl.div_rn(blocks, decoded[:, None]), (BLOCK,))
    magnitude = tl.abs(scaled)
    code = tl.full((BLOCK,), 0, tl.int32)
    for edge in tl.static_range(7):
        threshold = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)[edge]
        code += (magnitude > threshold).to(tl.int32)
        if edge % 2:
            code += (magnitude == threshold).to(tl.int32)
    sign = (scaled.to(tl.int32, bitcast=True) >> 31) & 1
    code |= sign << 3
    pairs = tl.reshape(code, (BLOCK // 2, 2))
    byte = tl.sum(pairs * (1 << (4 * tl.arange(0, 2)))[None, :], 1)
    byte_ids = tl.arange(0, BLOCK // 2)
    tl.store(Packed + row * (K // 2) + byte_ids, byte, (row < ROWS) & (byte_ids < K // 2))
    padded_blocks: tl.constexpr = ((K // 16 + 3) // 4) * 4
    tile = (row // 128) * (padded_blocks // 4) + block_ids // 4
    offset = tile * 512 + (row % 32) * 16 + ((row % 128) // 32) * 4 + block_ids % 4
    tl.store(Swizzled + offset, tl.where(row < ROWS, scales, 0), block_ids < padded_blocks)
