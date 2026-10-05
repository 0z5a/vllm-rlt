import pytest
import torch
from torch.nn import functional as F

from loopquant.quality import native_continuation_nll, native_window_nll
from tests.loopquant_int4_checks import int4_reference_model
from tests.reference.hrm_text import dense_hrm_reference
from tests.reference.loopformer import dense_loopformer_reference
from vllm_rlt.core.compact_kv_cache import CompactKVCacheManager
from vllm_rlt.core.kv_cache_manager import KVCacheManager


@pytest.mark.parametrize("family", ["hrm_text", "loopformer"])
@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.parametrize("prefix", [1, 5])
def test_native_continuation_matches_prefix_oracle_and_releases_cache(family, compact, prefix):
    torch.manual_seed(71)
    model = int4_reference_model(family).eval()
    config = model.config
    cache = (CompactKVCacheManager if compact else KVCacheManager)(
        config.num_hidden_layers,
        config.num_key_value_heads,
        config.head_dim,
        num_blocks=64,
        block_size=2,
        max_loops=config.total_ut_steps,
    )
    tokens = torch.tensor([3, 7, 4, 2, 6, 8, 9])
    if family == "hrm_text":
        _, logits, _ = dense_hrm_reference(model, tokens, prefix)
        with pytest.raises(ValueError, match="explicit prefix"):
            native_window_nll(model, tokens, config.total_ut_steps, cache)
    else:
        _, logits, _ = dense_loopformer_reference(model, tokens)
    expected = F.cross_entropy(logits[prefix - 1 : -1], tokens[prefix:], reduction="sum")
    actual, count = native_continuation_nll(model, tokens, prefix, config.total_ut_steps, cache)
    assert count == len(tokens) - prefix and cache.num_used_blocks == 0
    torch.testing.assert_close(torch.tensor(actual), expected, atol=3e-5, rtol=3e-5)
    again = native_continuation_nll(model, tokens, prefix, config.total_ut_steps, cache)
    assert again == (actual, count) and cache.num_used_blocks == 0


@pytest.mark.parametrize("family", ["ouro", "nanbeige", "loopformer"])
def test_causal_full_window_and_incremental_continuation_score_same_targets(family):
    torch.manual_seed(71)
    model = int4_reference_model(family).eval()
    config = model.config
    cache = KVCacheManager(
        config.num_hidden_layers,
        config.num_key_value_heads,
        config.head_dim,
        num_blocks=64,
        block_size=2,
        max_loops=config.total_ut_steps,
    )
    tokens = torch.tensor([3, 7, 4, 2, 6, 8, 9])
    full, full_count = native_window_nll(model, tokens, config.total_ut_steps, cache)
    actual, count = native_continuation_nll(model, tokens, 1, config.total_ut_steps, cache)
    assert count == full_count and cache.num_used_blocks == 0
    torch.testing.assert_close(torch.tensor(actual), torch.tensor(full), atol=3e-5, rtol=3e-5)
