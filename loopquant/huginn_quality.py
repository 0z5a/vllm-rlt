"""Native Huginn likelihood with boundary KV and an explicit initial-state seed."""

import torch
from torch.nn import functional as F

from vllm_rlt.core.kv_cache_manager import KVCacheManager
from vllm_rlt.models.huginn import HuginnForCausalLM


@torch.inference_mode()
def native_huginn_window_nll(
    model: HuginnForCausalLM,
    tokens: torch.Tensor,
    loops: int,
    cache: KVCacheManager,
    *,
    seed: int,
) -> tuple[float, int]:
    """Score a causal window through prelude/core/coda using the actual native KV."""
    if tokens.ndim != 1 or len(tokens) < 2 or not 1 <= loops <= model.config.mean_recurrence:
        raise ValueError("expected a token window and a supported fixed recurrence")
    if cache.recurrent_layers != model.recurrent_kv_layers:
        raise ValueError("Huginn requires its declared boundary and recurrent KV layers")
    if cache.num_used_blocks or not cache.allocate("quality", len(tokens)):
        raise ValueError("quality evaluation requires an empty cache with enough capacity")
    try:
        requests, positions = ["quality"] * len(tokens), list(range(len(tokens)))
        boundary = cache._prepare_batch(requests, [0] * len(tokens), positions)
        devices = [tokens.device.index] if tokens.is_cuda else []
        with torch.random.fork_rng(devices=devices):
            torch.default_generator.manual_seed(seed)
            if tokens.is_cuda:
                with torch.cuda.device(tokens.device):
                    torch.cuda.manual_seed(seed)
            hidden = model.prelude_prepared(tokens, boundary, cache)
        for loop in range(loops):
            core = cache._prepare_batch(requests, [loop] * len(tokens), positions)
            hidden, _ = model.recurrent_prepared(hidden, core, cache, compute_gate=False)
        logits = model.coda_prepared(hidden, boundary, cache)[:-1].float()
        loss = F.cross_entropy(logits, tokens[1:], reduction="sum")
        if not torch.isfinite(loss):
            raise ValueError("nonfinite native Huginn NLL")
        for position in positions:
            cache.finalize_token("quality", position, loops - 1)
        return float(loss), len(tokens) - 1
    finally:
        cache.free("quality")
