"""A warm prefix must never survive a committed physical-policy change."""

import pytest
import torch

from tests.helpers import tiny_nanbeige_config, tiny_ouro_config
from vllm_rlt import LLM, CacheConfig, SamplingParams, SchedulerConfig
from vllm_rlt.models import NanbeigeForCausalLM, OuroForCausalLM


@pytest.mark.parametrize("family", ["ouro", "nanbeige"])
def test_prefix_is_invalidated_before_partial_weight_publication(family):
    torch.manual_seed(42)
    model = (
        OuroForCausalLM(tiny_ouro_config())
        if family == "ouro"
        else NanbeigeForCausalLM(tiny_nanbeige_config())
    )
    warm = LLM(model, cache_config=CacheConfig(128, 2, enable_prefix_caching=True))
    prompt = [2, 4, 6, 8, 10]
    params = SamplingParams(
        max_tokens=3,
        temperature=0.8,
        seed=31,
        ignore_eos=True,
        min_loops=model.config.total_ut_steps,
        logprobs=0,
        logprobs_mode="processed",
    )
    old = warm.generate([prompt], params)[0]
    cache = warm.engine.cache_manager
    assert len(cache.lookup_prefix(prompt)) == 2
    torch.manual_seed(99)
    replacement = type(model)(model.config)
    parameters = list(replacement.named_parameters())
    warm.start_weight_update(1)
    assert not cache.lookup_prefix(prompt) and not any(cache._refs)
    warm.update_weights(parameters[:1])
    with pytest.raises(ValueError, match="every physical parameter"):
        warm.finish_weight_update()
    with pytest.raises(RuntimeError, match="paused"):
        warm.generate([prompt], params)
    warm.update_weights(parameters[1:])
    warm.finish_weight_update()
    hits = cache.prefix_hits
    first = warm.generate([prompt], params)[0]
    assert cache.prefix_hits == hits
    second = warm.generate([prompt], params)[0]
    assert cache.prefix_hits == hits + 4
    cold = LLM(
        replacement,
        cache_config=CacheConfig(128, 2),
        scheduler_config=SchedulerConfig(prefill_chunk_size=2),
    )
    expected = cold.generate([prompt], params)[0]
    assert first.weight_version == second.weight_version == 1
    assert old.weight_version == expected.weight_version == 0
    assert first.token_ids == second.token_ids == expected.token_ids
    assert old.log_probs != first.log_probs
    for output in (first, second):
        torch.testing.assert_close(
            torch.tensor(output.log_probs), torch.tensor(expected.log_probs), atol=1e-5, rtol=3e-5
        )
    warm.engine.reset_prefix_cache()
    assert cache.num_used_blocks == 0 and not any(cache._refs)
    warm.close()
    cold.close()
