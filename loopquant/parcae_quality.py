"""Native Parcae likelihood with explicit seed and protected boundary KV."""

import torch
from torch.nn import functional as F

from vllm_rlt.core.kv_cache_manager import KVCacheManager
from vllm_rlt.models.parcae import ParcaeForCausalLM


@torch.inference_mode()
def native_parcae_window_nll(
    model: ParcaeForCausalLM, tokens: torch.Tensor, loops: int, cache: KVCacheManager, *, seed: int
) -> tuple[float, int]:
    if tokens.ndim != 1 or len(tokens) < 2 or loops != model.config.mean_recurrence:
        raise ValueError("expected a causal window and the registered fixed recurrence")
    if cache.recurrent_layers != model.recurrent_kv_layers:
        raise ValueError("Parcae requires its boundary and recurrent KV layer contract")
    if cache.num_used_blocks or not cache.allocate("quality", len(tokens)):
        raise ValueError("quality evaluation requires an empty cache with enough capacity")
    try:
        requests, positions = ["quality"] * len(tokens), list(range(len(tokens)))
        boundary = cache._prepare_batch(requests, [0] * len(tokens), positions)
        with torch.random.fork_rng(devices=[tokens.device.index] if tokens.is_cuda else []):
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
            raise ValueError("nonfinite native Parcae NLL")
        for position in positions:
            cache.finalize_token("quality", position, loops - 1)
        return float(loss), len(tokens) - 1
    finally:
        cache.free("quality")
