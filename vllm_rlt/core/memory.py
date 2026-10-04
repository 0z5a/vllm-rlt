"""Startup KV budget: explicit overrides or CUDA peak profiling plus headroom."""

import math
from dataclasses import replace

import torch

from vllm_rlt.core.kv_cache_manager import KVCacheManager
from vllm_rlt.core.stage_kv_cache import StageKVCacheManager
from vllm_rlt.models.huginn import HuginnForCausalLM
from vllm_rlt.worker.buffers import execution_buffer_bytes


def budget_blocks(available_bytes, bytes_per_block):
    blocks = int(available_bytes) // bytes_per_block
    if blocks < 1:
        raise ValueError(
            "no memory remains for KV blocks; reduce execution limits or memory reserve"
        )
    return blocks


def make_cache_manager(model, cache, num_blocks: int, backend: str) -> KVCacheManager:
    parameter, config = next(model.parameters()), model.config
    arguments = dict(
        num_layers=config.num_hidden_layers,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        max_loops=config.total_ut_steps,
        num_blocks=num_blocks,
        layout=cache.layout,
        block_size=cache.block_size,
        device=parameter.device,
        dtype=parameter.dtype,
        backend=backend,
        enable_prefix_caching=cache.enable_prefix_caching,
        incremental_allocation=cache.incremental_allocation,
        watermark_ratio=cache.watermark_ratio,
        recurrent_layers=model.recurrent_kv_layers
        if isinstance(model, HuginnForCausalLM)
        else None,
    )
    if cache.stage_aware:
        if not isinstance(model, HuginnForCausalLM):
            raise ValueError("stage-aware KV requires Huginn's explicit layer groups")
        return StageKVCacheManager(**arguments, groups=model.kv_groups)
    return KVCacheManager(**arguments)


@torch.inference_mode()
def plan_cache(model, cache, scheduler, execution, backend):
    parameter = next(model.parameters())
    config, device = model.config, parameter.device
    per_block = (
        2
        * config.num_hidden_layers
        * cache.block_size
        * config.num_key_value_heads
        * config.head_dim
        * parameter.element_size()
    )
    unit_blocks, unit_bytes = 1, per_block
    if cache.stage_aware:
        if not isinstance(model, HuginnForCausalLM):
            raise ValueError("stage-aware KV requires Huginn's explicit layer groups")
        unit_blocks = config.total_ut_steps
        planes = sum(group.planes for group in model.kv_groups)
        unit_bytes = per_block // config.num_hidden_layers * planes
        per_block = unit_bytes / unit_blocks
    unit_info = {"bytes_per_block": per_block}
    if cache.stage_aware:
        unit_info.update(bytes_per_page_bundle=unit_bytes, blocks_per_bundle=unit_blocks)
    if cache.num_blocks is not None:
        return cache.num_blocks // unit_blocks * unit_blocks, {"source": "blocks", **unit_info}
    if cache.kv_cache_memory_bytes is not None:
        blocks = budget_blocks(cache.kv_cache_memory_bytes, unit_bytes) * unit_blocks
        return blocks, {"source": "bytes", **unit_info}
    if device.type != "cuda":
        return 256 // unit_blocks * unit_blocks, {"source": "cpu_default", **unit_info}
    torch.cuda.synchronize(device)
    # Return unused allocator segments before measuring driver-free memory.
    # In particular, a previous engine may have left its KV pool cached here.
    # Live allocations and non-releasable segments remain reserved.
    with torch.cuda.device(device):
        torch.cuda.empty_cache()
    free_before, total = torch.cuda.mem_get_info(device)
    reserved = torch.cuda.memory_reserved(device)
    allocated = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    count = min(
        scheduler.max_num_batched_tokens, scheduler.max_num_seqs * config.max_position_embeddings
    )
    if execution.pad_to_power_of_two:
        count = 1 << (count - 1).bit_length()
    lengths = [
        min(config.max_position_embeddings, count - start)
        for start in range(0, count, config.max_position_embeddings)
    ]
    probe = make_cache_manager(
        model,
        replace(
            cache,
            layout="last_exited",
            enable_prefix_caching=False,
            incremental_allocation=False,
            watermark_ratio=0.0,
        ),
        sum(math.ceil(length / cache.block_size) for length in lengths) * config.total_ut_steps,
        backend,
    )
    ids, positions = [], []
    for index, length in enumerate(lengths):
        rid = f"profile-{index}"
        probe.allocate(rid, length)
        ids.extend([rid] * length)
        positions.extend(range(length))
    tokens = torch.zeros(count, dtype=torch.long, device=device)
    if getattr(model, "requires_boundary_kv", False):
        boundary = probe._prepare_batch(ids, [0] * count, positions)
        hidden = model.prelude_prepared(tokens, boundary, probe)
    else:
        hidden = model.prelude(tokens)
    for depth in range(config.total_ut_steps):
        hidden, _ = model.recurrent(hidden, ids, [depth] * count, positions, probe)
    logits = (
        model.coda_prepared(hidden, boundary, probe)
        if getattr(model, "requires_boundary_kv", False)
        else model.coda(hidden)
    )
    torch.cuda.synchronize(device)
    peak = max(0, torch.cuda.max_memory_allocated(device) - allocated)
    del hidden, logits, probe
    # Profiling includes a conservative temporary KV allocation. Also budget for
    # long-context reference attention (Triton's online softmax has bounded scratch).
    context_scratch = 0
    if backend == "torch":
        context_scratch = (
            config.max_position_embeddings
            * config.num_attention_heads
            * (8 * config.head_dim + 8)
            * 4
        )
    buffers = execution_buffer_bytes(config, scheduler, cache, execution, parameter.element_size())
    # Dynamic metadata at the largest supported context is small but not zero.
    metadata = (
        scheduler.max_num_batched_tokens
        * math.ceil(config.max_position_embeddings / cache.block_size)
        * 4
    )
    budget = min(free_before, int(total * cache.gpu_memory_utilization) - reserved)
    concurrent_peak = peak * (2 if execution.async_scheduling and execution.multi_stream else 1)
    graph_reserve = execution.cuda_graph_memory_reserve_bytes if execution.cuda_graphs else 0
    budget -= (
        concurrent_peak
        + context_scratch
        + buffers
        + metadata
        + cache.memory_reserve_bytes
        + graph_reserve
    )
    # More blocks than all concurrent full-context requests can use only waste
    # memory, particularly for tiny-model GPU tests with very small pages.
    useful_blocks = (
        math.ceil(config.max_position_embeddings / cache.block_size)
        * (config.total_ut_steps if cache.layout == "last_exited" else 1)
        * scheduler.max_num_seqs
    )
    blocks = min(budget_blocks(budget, unit_bytes) * unit_blocks, useful_blocks)
    return blocks, dict(
        source="cuda_profile",
        **unit_info,
        profile_peak_bytes=peak,
        concurrent_peak_bytes=concurrent_peak,
        max_useful_blocks=useful_blocks,
        cuda_graph_reserve_bytes=graph_reserve,
        static_buffer_bytes=buffers,
        context_scratch_bytes=context_scratch,
        kv_budget_bytes=int(budget),
    )
