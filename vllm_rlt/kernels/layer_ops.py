# SPDX-License-Identifier: Apache-2.0
"""Decoder-layer operators as single kernel launches.

Without CUDA graphs a decode round is bound by the number of launches the host
issues, not by the work they do. Each operator here replaces a run of composite
operators with one program per row:

- ``rope_qk``: split-half rotary embedding of the query and key heads;
- ``kv_store``: one token's keys and values into one layer of the paged cache;
- ``rms_norm_add``: ``residual + RMSNorm(x)`` and, optionally, the RMSNorm of
  that sum for the block that reads it next;
- ``silu_mul``: ``silu(gate) * up`` over the two halves of one projection.

The composite operators in ``vllm_rlt.layers`` and the models stay the
reference. In the 16-bit dtypes every product and sum below is rounded exactly
where the composite operator rounds it; ``rms_norm_add`` shares the reduction
of ``kernels.rmsnorm``.
"""

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from vllm_rlt.kernels.rmsnorm import DTYPES, MAX_WIDTH, rms_norm
from vllm_rlt.kernels.sparse_decode import _launch

# The 16-bit dtypes, where every product is rounded before the sum that
# follows it, as the composite operators round it.
ROUNDED = frozenset((torch.float16, torch.bfloat16))
# One program holds all heads of a row: [heads, head_dim] elements.
MAX_ROPE_TILE = 8192
_STORE_TILE = 1024
_SILU_TILE = 1024

__all__ = [
    "DTYPES",
    "MAX_ROPE_TILE",
    "MAX_WIDTH",
    "ROUNDED",
    "kv_store",
    "kv_store_supported",
    "rms_norm",
    "rms_norm_add",
    "rope_qk",
    "rope_supported",
    "silu_mul",
]


@triton.jit
def _rounded(x, dtype: tl.constexpr):
    """``x.to(dtype).to(float32)`` that the compiler cannot fold away.

    A plain round trip lets LLVM shrink ``round(a * b) + c`` to 16-bit operators
    and contract them into one FMA, which drops the rounding of the product.
    """
    if dtype == tl.bfloat16:
        value = tl.inline_asm_elementwise(
            "cvt.rn.bf16.f32 $0, $1;", "=h,r", [x], dtype=tl.bfloat16, is_pure=True, pack=1
        )
    elif dtype == tl.float16:
        value = tl.inline_asm_elementwise(
            "cvt.rn.f16.f32 $0, $1;", "=h,r", [x], dtype=tl.float16, is_pure=True, pack=1
        )
    else:
        value = x
    return value.to(tl.float32)


@triton.jit
def _rope_kernel(
    Q,
    K,
    COS,
    SIN,
    Q_OUT,
    K_OUT,
    q_stride,
    k_stride,
    k_offset,
    angle_stride,
    Q_HEADS: tl.constexpr,
    K_HEADS: tl.constexpr,
    D: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    dims = tl.arange(0, BLOCK_D)
    heads = tl.arange(0, BLOCK_H)
    half = D // 2
    valid = dims < D
    cos = tl.load(COS + row * angle_stride + dims, mask=valid, other=0).to(tl.float32)
    sin = tl.load(SIN + row * angle_stride + dims, mask=valid, other=0).to(tl.float32)
    # rotate_half is (-second, first): fold its sign into the sine, which is exact.
    sin = tl.where(dims < half, -sin, sin)
    partner = tl.where(dims < half, dims + half, dims - half)
    here = heads[:, None] * D + dims[None, :]
    there = heads[:, None] * D + partner[None, :]

    q_mask = (heads[:, None] < Q_HEADS) & valid[None, :]
    q = tl.load(Q + row * q_stride + here, mask=q_mask, other=0).to(tl.float32)
    q_rot = tl.load(Q + row * q_stride + there, mask=q_mask, other=0).to(tl.float32)
    q_cos = _rounded(q * cos[None, :], Q.dtype.element_ty)
    q_sin = _rounded(q_rot * sin[None, :], Q.dtype.element_ty)
    tl.store(
        Q_OUT + row * (Q_HEADS * D) + here, (q_cos + q_sin).to(Q.dtype.element_ty), mask=q_mask
    )

    k_mask = (heads[:, None] < K_HEADS) & valid[None, :]
    k = tl.load(K + row * k_stride + k_offset + here, mask=k_mask, other=0).to(tl.float32)
    k_rot = tl.load(K + row * k_stride + k_offset + there, mask=k_mask, other=0).to(tl.float32)
    k_cos = _rounded(k * cos[None, :], K.dtype.element_ty)
    k_sin = _rounded(k_rot * sin[None, :], K.dtype.element_ty)
    tl.store(
        K_OUT + row * (K_HEADS * D) + here, (k_cos + k_sin).to(K.dtype.element_ty), mask=k_mask
    )


def rope_supported(q_heads: int, k_heads: int, head_dim: int) -> bool:
    """Whether one program can hold a row's heads."""
    heads = triton.next_power_of_2(max(q_heads, k_heads, 1))
    return 0 < head_dim and heads * triton.next_power_of_2(head_dim) <= MAX_ROPE_TILE


def rope_qk(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    q_heads: int,
    k_heads: int,
    k_offset: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``x * cos + rotate_half(x) * sin`` for the query and key heads of each row.

    Row ``i`` of ``q`` holds ``q_heads`` contiguous heads of ``head_dim`` values
    and row ``i`` of ``k`` holds ``k_heads`` of them from element ``k_offset``;
    both may be the same projection output. ``cos`` and ``sin`` hold one
    ``head_dim`` row per input row. All four share a 16-bit dtype (``ROUNDED``)
    and a CUDA device, and ``rope_supported`` holds. Returns new
    ``[rows, heads, head_dim]`` tensors, bit for bit the composite result.
    """
    rows, head_dim = q.shape[0], cos.shape[-1]
    q_out = torch.empty((rows, q_heads, head_dim), device=q.device, dtype=q.dtype)
    k_out = torch.empty((rows, k_heads, head_dim), device=q.device, dtype=q.dtype)
    if rows:
        _launch(
            _rope_kernel,
            (rows,),
            q,
            k,
            cos,
            sin,
            q_out,
            k_out,
            q.stride(0),
            k.stride(0),
            k_offset,
            cos.stride(0),
            Q_HEADS=q_heads,
            K_HEADS=k_heads,
            D=head_dim,
            BLOCK_H=triton.next_power_of_2(max(q_heads, k_heads)),
            BLOCK_D=triton.next_power_of_2(head_dim),
            num_warps=4,
        )
    return q_out, k_out


@triton.jit
def _kv_store_kernel(
    K,
    V,
    KEYS,
    VALUES,
    BLOCKS,
    OFFSETS,
    k_stride,
    v_stride,
    layer_offset,
    block_stride,
    token_stride,
    WIDTH: tl.constexpr,
    TILE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * TILE + tl.arange(0, TILE)
    mask = cols < WIDTH
    block = tl.load(BLOCKS + row).to(tl.int64)
    offset = tl.load(OFFSETS + row).to(tl.int64)
    address = block * block_stride + layer_offset + offset * token_stride + cols
    tl.store(KEYS + address, tl.load(K + row * k_stride + cols, mask=mask), mask=mask)
    tl.store(VALUES + address, tl.load(V + row * v_stride + cols, mask=mask), mask=mask)


def kv_store(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    layer: int,
    k: torch.Tensor,
    v: torch.Tensor,
    blocks: torch.Tensor,
    offsets: torch.Tensor,
    rows: int,
) -> None:
    """``cache[blocks, layer, offsets] = x[:rows]`` for the keys and the values.

    The caches are ``[block, layer, token, head, dim]`` with equal strides and a
    contiguous ``[head, dim]`` tail; ``k`` and ``v`` are ``[>= rows, head, dim]``
    of the cache dtype with the same tail; ``blocks`` and ``offsets`` are
    contiguous integer vectors that address each destination once.
    """
    if rows:
        width = k.shape[1] * k.shape[2]
        _launch(
            _kv_store_kernel,
            (rows, triton.cdiv(width, _STORE_TILE)),
            k,
            v,
            key_cache,
            value_cache,
            blocks,
            offsets,
            k.stride(0),
            v.stride(0),
            layer * key_cache.stride(1),
            key_cache.stride(0),
            key_cache.stride(2),
            WIDTH=width,
            TILE=_STORE_TILE,
            num_warps=4,
        )


def kv_store_supported(
    key_cache: torch.Tensor, value_cache: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> bool:
    """The layout ``kv_store`` addresses: equal cache strides and contiguous tails."""
    tail = (k.shape[2], 1)
    return (
        key_cache.stride() == value_cache.stride()
        and key_cache.stride()[3:] == tail
        and k.stride()[1:] == tail
        and v.stride()[1:] == tail
    )


@triton.jit
def _norm_add_kernel(
    X,
    W,
    R,
    OUT,
    W_NEXT,
    OUT_NEXT,
    eps,
    eps_next,
    x_stride,
    H: tl.constexpr,
    BLOCK: tl.constexpr,
    NEXT: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    mask = cols < H
    x = tl.load(X + row * x_stride + cols, mask=mask, other=0).to(tl.float32)
    scale = tl.rsqrt(tl.sum(x * x, axis=0) / H + eps)
    # As kernels.rmsnorm: normalize in FP32, round, scale, round.
    value = (x * scale).to(X.dtype.element_ty).to(tl.float32)
    weight = tl.load(W + cols, mask=mask, other=0).to(tl.float32)
    normed = _rounded(weight * value, OUT.dtype.element_ty)
    residual = tl.load(R + row * H + cols, mask=mask, other=0).to(tl.float32)
    total = (residual + normed).to(OUT.dtype.element_ty)
    tl.store(OUT + row * H + cols, total, mask=mask)
    if NEXT:
        y = total.to(tl.float32)
        scale_next = tl.rsqrt(tl.sum(y * y, axis=0) / H + eps_next)
        value_next = (y * scale_next).to(OUT.dtype.element_ty).to(tl.float32)
        weight_next = tl.load(W_NEXT + cols, mask=mask, other=0).to(tl.float32)
        tl.store(
            OUT_NEXT + row * H + cols,
            (weight_next * value_next).to(OUT.dtype.element_ty),
            mask=mask,
        )


def rms_norm_add(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    residual: torch.Tensor,
    next_weight: torch.Tensor | None = None,
    next_eps: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """``total = residual + RMSNorm(x)`` and, with ``next_weight``, ``RMSNorm(total)``.

    ``x`` and ``residual`` are ``[rows, width]`` of one dtype in ``DTYPES`` on the
    CUDA device of the weights, ``residual`` contiguous, at most ``MAX_WIDTH``
    wide. Each result owns new storage. In the 16-bit dtypes both results equal
    ``kernels.rmsnorm.rms_norm`` composed with the dtype's own addition.
    """
    total = torch.empty_like(residual)
    following = None if next_weight is None else torch.empty_like(residual)
    rows, width = residual.shape
    if rows:
        _launch(
            _norm_add_kernel,
            (rows,),
            x,
            weight,
            residual,
            total,
            weight if next_weight is None else next_weight,
            total if following is None else following,
            eps,
            next_eps,
            x.stride(0),
            H=width,
            BLOCK=triton.next_power_of_2(width),
            NEXT=next_weight is not None,
            num_warps=8,
        )
    return total, following


@triton.jit
def _silu_mul_kernel(GATE_UP, OUT, stride, WIDTH: tl.constexpr, TILE: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * TILE + tl.arange(0, TILE)
    mask = cols < WIDTH
    gate = tl.load(GATE_UP + row * stride + cols, mask=mask, other=0).to(tl.float32)
    up = tl.load(GATE_UP + row * stride + WIDTH + cols, mask=mask, other=0).to(tl.float32)
    # As aten::silu: x / (1 + expf(-x)) in FP32 with the accurate exponential
    # and an IEEE division, rounded before the product.
    silu = _rounded(tl.math.div_rn(gate, 1.0 + libdevice.exp(-gate)), OUT.dtype.element_ty)
    tl.store(OUT + row * WIDTH + cols, (silu * up).to(OUT.dtype.element_ty), mask=mask)


def silu_mul(gate_up: torch.Tensor) -> torch.Tensor:
    """``silu(gate) * up`` where row ``i`` of ``gate_up`` is ``[gate_i, up_i]``.

    ``gate_up`` is ``[rows, 2 * width]`` of a dtype in ``DTYPES`` on CUDA with a
    contiguous last dimension. Returns a new ``[rows, width]`` tensor.
    """
    rows, width = gate_up.shape[0], gate_up.shape[1] // 2
    out = torch.empty((rows, width), device=gate_up.device, dtype=gate_up.dtype)
    if rows and width:
        _launch(
            _silu_mul_kernel,
            (rows, triton.cdiv(width, _SILU_TILE)),
            gate_up,
            out,
            gate_up.stride(0),
            WIDTH=width,
            TILE=_SILU_TILE,
            num_warps=4,
        )
    return out
