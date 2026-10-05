import pytest
import torch

from loopquant.adapters.nanbeige import NanbeigeAdapter
from loopquant.adapters.ouro import OuroAdapter
from loopquant.quality import native_window_nll, next_token_nll
from tests.helpers import tiny_nanbeige_config, tiny_ouro_config
from vllm_rlt.core.kv_cache_manager import KVCacheManager
from vllm_rlt.models.nanbeige import NanbeigeForCausalLM
from vllm_rlt.models.ouro import OuroForCausalLM


@pytest.mark.parametrize("kind", ["ouro", "nanbeige"])
def test_native_quality_matches_dense_reference_and_releases_reused_cache(kind):
    torch.manual_seed(19)
    if kind == "ouro":
        model = OuroForCausalLM(tiny_ouro_config())
        adapter = OuroAdapter(model)
    else:
        model = NanbeigeForCausalLM(tiny_nanbeige_config())
        adapter = NanbeigeAdapter(model)
    config = model.config
    cache = KVCacheManager(
        config.num_hidden_layers,
        config.num_key_value_heads,
        config.head_dim,
        config.total_ut_steps * 2,
        16,
        config.total_ut_steps,
        dtype=torch.float32,
    )
    for tokens in [torch.tensor([3, 8, 7, 4, 1]), torch.tensor([2, 11, 3])]:
        for loops in [1, config.total_ut_steps]:
            with torch.inference_mode():
                valid = torch.ones_like(tokens[None], dtype=torch.bool)
                reference = adapter(tokens[None], valid, loops)
                expected, count = next_token_nll(reference.logits, tokens[None], valid)
                actual, targets = native_window_nll(model, tokens, loops, cache)
            assert targets == count
            assert actual == pytest.approx(float(expected), abs=1e-5, rel=1e-6)
            assert cache.num_used_blocks == 0
