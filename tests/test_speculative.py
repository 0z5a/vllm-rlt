"""Fixed-depth greedy verification, including rejected KV tails."""

import pytest
import torch

from vllm_rlt import CacheConfig, ExitConfig, SamplingParams
from vllm_rlt.core.kv_cache_manager import KVCacheManager
from vllm_rlt.engine.llm_engine import LLMEngine
from vllm_rlt.models import OuroConfig, OuroForCausalLM
from vllm_rlt.speculative import FixedDepthGreedy, online_ngram


def test_ngram_uses_only_committed_history():
    assert online_ngram([1, 2, 3, 1, 2], 3) == [3, 1, 2]
    assert online_ngram([1, 2, 3], 4) == []


@pytest.mark.parametrize("prefix", [4, 6])
@pytest.mark.parametrize("incremental", [False, True])
def test_fork_commits_only_verified_kv(prefix, incremental):
    cache = KVCacheManager(1, 1, 2, 24, 4, 2)
    assert cache.allocate("base", 12, initial_tokens=prefix if incremental else None)
    for depth in range(2):
        for position in range(prefix):
            value = torch.full((1, 1, 2), depth * 100 + position, dtype=torch.float32)
            cache.write(0, ["base"], [depth], [position], value, value)
    assert cache.fork_prefix("base", "fork", prefix, prefix + 3)
    for depth in range(2):
        for position in range(prefix, prefix + 3):
            value = torch.full((1, 1, 2), depth * 100 + position, dtype=torch.float32)
            cache.write(0, ["fork"], [depth], [position], value, value)
        assert cache.read(0, "base", depth, prefix)[0].shape[0] == prefix
    cache.commit_fork("base", "fork", prefix + 1)
    for depth in range(2):
        assert torch.equal(
            cache.read(0, "base", depth, prefix + 1)[0][-1],
            torch.full((1, 2), depth * 100 + prefix, dtype=torch.float32),
        )
        with pytest.raises(RuntimeError, match="uninitialized"):
            cache.read(0, "base", depth, prefix + 2)
    cache.free("base")
    assert cache.num_used_blocks == 0


@pytest.mark.parametrize("reject_at", [0, 1, 2, 3])
def test_speculative_greedy_matches_native_after_rejection(reject_at):
    torch.manual_seed(7)
    config = OuroConfig.tiny()
    model = OuroForCausalLM(config)
    prompt = [2, 3, 4, 2, 3, 4]
    params = SamplingParams(
        max_tokens=8,
        min_loops=config.total_ut_steps,
        max_loops=config.total_ut_steps,
        ignore_eos=True,
    )
    native = LLMEngine(
        model,
        cache_config=CacheConfig(128, 4, "last_exited"),
        exit_config=ExitConfig("ouro"),
        attention_backend="torch",
    )
    native.add_request("reference", prompt, params)
    expected = None
    while native.has_unfinished_requests():
        for event in native.step():
            if event.finished:
                expected = event.token_ids
    cache = KVCacheManager(
        config.num_hidden_layers,
        config.num_key_value_heads,
        config.head_dim,
        128,
        4,
        config.total_ut_steps,
    )
    decoder = FixedDepthGreedy(model, cache, prefill_chunk_size=4)
    used = False

    def controlled(history, limit):
        nonlocal used
        if used:
            return []
        used = True
        draft = expected[len(history) - len(prompt) :][:limit]
        if reject_at < len(draft):
            draft[reject_at] = (draft[reject_at] + 1) % config.vocab_size
        return draft

    actual = decoder.generate(prompt, 8, gamma=3, proposer=controlled, ignore_eos=True)
    assert actual.token_ids == expected
    assert actual.accepted_lengths == [reject_at]
    assert cache.num_used_blocks == 0


def test_output_budget_one_never_drafts():
    torch.manual_seed(5)
    config = OuroConfig.tiny()
    model = OuroForCausalLM(config)
    cache = KVCacheManager(
        config.num_hidden_layers,
        config.num_key_value_heads,
        config.head_dim,
        32,
        4,
        config.total_ut_steps,
    )
    decoder = FixedDepthGreedy(model, cache)
    result = decoder.generate([2, 3], 1, gamma=4, ignore_eos=True)
    assert len(result.token_ids) == 1
    assert result.target_cycles == 0
    assert cache.num_used_blocks == 0


@pytest.mark.gpu
def test_triton_batched_verifier_matches_ordinary_decode():
    torch.manual_seed(19)
    config = OuroConfig.tiny()
    model = OuroForCausalLM(config).to("cuda:0", torch.bfloat16)
    cache = KVCacheManager(
        config.num_hidden_layers,
        config.num_key_value_heads,
        config.head_dim,
        96,
        4,
        config.total_ut_steps,
        device="cuda:0",
        dtype=torch.bfloat16,
        backend="triton",
    )
    decoder = FixedDepthGreedy(model, cache)
    prompt = [2, 3, 4, 2, 3, 4]
    baseline = decoder.generate(prompt, 7, ignore_eos=True)
    used = False

    def controlled(history, limit):
        nonlocal used
        if used:
            return []
        used = True
        return baseline.token_ids[len(history) - len(prompt) :][:limit]

    speculative = decoder.generate(prompt, 7, gamma=3, proposer=controlled, ignore_eos=True)
    assert speculative.token_ids == baseline.token_ids
    assert speculative.accepted_lengths == [3]
    assert cache.num_used_blocks == 0
