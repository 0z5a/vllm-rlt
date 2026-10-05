"""Boundary KV can finish after recurrent finalization without overwriting it."""

from itertools import product

import pytest
import torch

from tests.test_huginn import tiny_huginn_config
from vllm_rlt import CacheConfig, ExecutionConfig, SamplingParams, SchedulerConfig
from vllm_rlt.core.alias_kv_cache import AliasKVCacheManager
from vllm_rlt.core.compact_kv_cache import CompactKVCacheManager
from vllm_rlt.engine.llm_engine import LLMEngine
from vllm_rlt.models import HuginnForCausalLM


@pytest.mark.parametrize("cache_type", [AliasKVCacheManager, CompactKVCacheManager])
def test_coda_first_write_after_finalize_keeps_core_immutable(cache_type):
    cache = cache_type(4, 2, 8, 64, 2, max_loops=3, recurrent_layers=(1, 2))
    cache.allocate("a", 4)
    payload = torch.ones(1, 2, 8)
    for layer in (0, 1, 2):
        cache.write(layer, ["a"], [0], [0], payload * (layer + 1), -payload)
    cache.finalize_token("a", 0, 0)
    cache.write(3, ["a"], [0], [0], payload * 4, -payload)
    assert torch.equal(cache.read(3, "a", 0, 1)[0], payload * 4)
    for depth in range(3):
        assert torch.equal(cache.read(1, "a", depth, 1)[0], payload * 2)
    for layer in range(4):
        with pytest.raises(ValueError, match="finalized"):
            cache.write(layer, ["a"], [0], [0], payload * 100, payload)
    with pytest.raises(ValueError, match="boundary|finalized"):
        cache.write(3, ["a"], [1], [0], payload, payload)
    with pytest.raises(RuntimeError, match="uninitialized"):
        cache.read(3, "a", 1, 1)
    cache.free("a")
    assert cache.num_free_blocks == 64


@pytest.mark.parametrize("loops", [1, 3])
def test_huginn_full_engine_cache_modes_match_with_reuse(loops):
    torch.manual_seed(42)
    model = HuginnForCausalLM(tiny_huginn_config()).eval()
    results = []
    for mode, batched in product(("native", "alias", "compact", "credits"), (False, True)):
        engine = LLMEngine(
            model,
            cache_config=CacheConfig(
                128,
                2,
                alias_last_exited=mode == "alias",
                compact_last_exited=mode in ("compact", "credits"),
                reclaim_skipped_credits=mode == "credits",
            ),
            scheduler_config=SchedulerConfig(max_num_seqs=2, max_num_batched_tokens=8),
            execution_config=ExecutionConfig(prefill_batch_metadata=batched),
        )
        rounds = []
        for _ in range(2):
            torch.manual_seed(37)
            for rid, prompt in (("a", [2, 3, 4]), ("b", [5, 6])):
                engine.add_request(
                    rid,
                    prompt,
                    SamplingParams(
                        max_tokens=5,
                        min_loops=loops,
                        max_loops=loops,
                        ignore_eos=True,
                    ),
                )
            completed = {}
            while engine.has_unfinished_requests():
                for output in engine.step():
                    if output.finished:
                        completed[output.request_id] = (output.token_ids, output.exit_depths)
            assert len(completed) == 2 and engine.cache_manager.num_free_blocks == 128
            rounds.append(completed)
        assert rounds[0] == rounds[1]
        results.append(rounds[0])
        engine.close()
    assert all(result == results[0] for result in results)
