"""Paged GQA kernel using bounded-memory online softmax.

One Triton program computes one query head. Unlike a fixed-depth batch, every
query row receives its own physical block table. This is a correctness-first
kernel without a performance claim or autotuning.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _paged_attention_kernel(
    Q,
    K,
    V,
    TABLES,
    LENGTHS,
    OUT,
    EXITS,
    DEPTH_TABLES,
    QUERY_DEPTHS,
    ALIAS_STARTS,
    RECORD_MAP_POINTERS,
    RECORD_MAP_WIDTHS,
    q_batch_stride: tl.constexpr,
    q_head_stride: tl.constexpr,
    q_dim_stride: tl.constexpr,
    k_block_stride: tl.constexpr,
    k_token_stride: tl.constexpr,
    k_head_stride: tl.constexpr,
    k_dim_stride: tl.constexpr,
    v_block_stride: tl.constexpr,
    v_token_stride: tl.constexpr,
    v_head_stride: tl.constexpr,
    v_dim_stride: tl.constexpr,
    table_stride: tl.constexpr,
    depth_row_stride: tl.constexpr,
    depth_stride: tl.constexpr,
    ALIAS: tl.constexpr,
    RECORDS: tl.constexpr,
    MAX_LOOPS: tl.constexpr,
    out_batch_stride: tl.constexpr,
    out_head_stride: tl.constexpr,
    out_dim_stride: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    HEAD_GROUPS: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    row = tl.program_id(0)
    head = tl.program_id(1)
    kv_head = head // HEAD_GROUPS
    length = tl.load(LENGTHS + row)
    dims = tl.arange(0, BLOCK_D)
    query = tl.load(
        Q + row * q_batch_stride + head * q_head_stride + dims * q_dim_stride,
        mask=dims < HEAD_DIM,
        other=0,
    ).to(tl.float32)
    maximum = -float("inf")
    normalizer = 0.0
    accumulator = tl.zeros((BLOCK_D,), dtype=tl.float32)
    for start in range(0, length, BLOCK_T):
        positions = start + tl.arange(0, BLOCK_T)
        valid_tokens = positions < length
        if RECORDS:
            pointer = tl.load(RECORD_MAP_POINTERS + row).to(tl.pointer_type(tl.int32))
            width = tl.load(RECORD_MAP_WIDTHS + row).to(tl.int64)
            query_depth = tl.load(QUERY_DEPTHS + row)
            exit_depths = tl.load(
                pointer + MAX_LOOPS * width + positions, mask=valid_tokens, other=-1
            )
            sources = tl.where(exit_depths >= 0, tl.minimum(query_depth, exit_depths), query_depth)
            records = tl.load(pointer + sources * width + positions, mask=valid_tokens, other=0).to(
                tl.int64
            )
            blocks, offsets = records // PAGE_SIZE, records % PAGE_SIZE
        elif ALIAS:
            if start + BLOCK_T <= tl.load(ALIAS_STARTS + row):
                blocks = tl.load(
                    TABLES + row * table_stride + positions // PAGE_SIZE,
                    mask=valid_tokens,
                    other=0,
                ).to(tl.int64)
            else:
                query_depth = tl.load(QUERY_DEPTHS + row)
                first_blocks = tl.load(
                    DEPTH_TABLES + row * depth_row_stride + positions // PAGE_SIZE,
                    mask=valid_tokens,
                    other=0,
                ).to(tl.int64)
                exit_depths = tl.load(
                    EXITS + first_blocks * PAGE_SIZE + positions % PAGE_SIZE,
                    mask=valid_tokens,
                    other=-1,
                )
                sources = tl.where(
                    exit_depths >= 0, tl.minimum(query_depth, exit_depths), query_depth
                )
                blocks = tl.load(
                    DEPTH_TABLES
                    + row * depth_row_stride
                    + sources * depth_stride
                    + positions // PAGE_SIZE,
                    mask=valid_tokens,
                    other=0,
                ).to(tl.int64)
        else:
            blocks = tl.load(
                TABLES + row * table_stride + positions // PAGE_SIZE,
                mask=valid_tokens,
                other=0,
            ).to(tl.int64)
        # Physical IDs fit in int32, but ID * block_stride can exceed 2**31
        # elements in a large KV pool. Promote BEFORE multiplying, for K and V.
        if not RECORDS:
            offsets = positions % PAGE_SIZE
        keys = tl.load(
            K
            + blocks[:, None] * k_block_stride
            + offsets[:, None] * k_token_stride
            + kv_head * k_head_stride
            + dims[None, :] * k_dim_stride,
            mask=valid_tokens[:, None] & (dims[None, :] < HEAD_DIM),
            other=0,
        ).to(tl.float32)
        scores = tl.sum(keys * query[None, :], axis=1) * SCALE
        scores = tl.where(valid_tokens, scores, -float("inf"))
        next_maximum = tl.maximum(maximum, tl.max(scores, axis=0))
        correction = tl.exp(maximum - next_maximum)
        weights = tl.exp(scores - next_maximum)
        values = tl.load(
            V
            + blocks[:, None] * v_block_stride
            + offsets[:, None] * v_token_stride
            + kv_head * v_head_stride
            + dims[None, :] * v_dim_stride,
            mask=valid_tokens[:, None] & (dims[None, :] < HEAD_DIM),
            other=0,
        ).to(tl.float32)
        accumulator = accumulator * correction + tl.sum(weights[:, None] * values, axis=0)
        normalizer = normalizer * correction + tl.sum(weights, axis=0)
        maximum = next_maximum
    tl.store(
        OUT + row * out_batch_stride + head * out_head_stride + dims * out_dim_stride,
        accumulator / tl.maximum(normalizer, 1.0e-30),
        mask=dims < HEAD_DIM,
    )


def paged_attention(
    q,
    key_cache,
    value_cache,
    block_tables,
    context_lengths,
    *,
    source_depths=None,
    depth_block_tables=None,
    query_depths=None,
    alias_starts=None,
    record_map_pointers=None,
    record_map_widths=None,
    max_loops=0,
):
    """Launch over [batch row, query head]; inputs are validated by the manager."""
    output = torch.empty_like(q)
    if q.shape[0] == 0:
        return output
    _paged_attention_kernel[(q.shape[0], q.shape[1])](
        q,
        key_cache,
        value_cache,
        block_tables,
        context_lengths,
        output,
        source_depths if source_depths is not None else block_tables,
        depth_block_tables if depth_block_tables is not None else block_tables,
        query_depths if query_depths is not None else context_lengths,
        alias_starts if alias_starts is not None else context_lengths,
        record_map_pointers if record_map_pointers is not None else block_tables,
        record_map_widths if record_map_widths is not None else context_lengths,
        *q.stride(),
        *key_cache.stride(),
        *value_cache.stride(),
        block_tables.stride(0),
        depth_block_tables.stride(0) if depth_block_tables is not None else 0,
        depth_block_tables.stride(1) if depth_block_tables is not None else 0,
        source_depths is not None,
        record_map_pointers is not None,
        max_loops,
        *output.stride(),
        HEAD_DIM=q.shape[-1],
        HEAD_GROUPS=q.shape[1] // key_cache.shape[2],
        PAGE_SIZE=key_cache.shape[1],
        SCALE=q.shape[-1] ** -0.5,
        BLOCK_D=triton.next_power_of_2(q.shape[-1]),
        BLOCK_T=32,
        num_warps=4,
    )
    return output
