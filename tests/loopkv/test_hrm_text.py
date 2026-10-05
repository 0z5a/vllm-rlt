"""HRM prefix/decode masks, independent numerical oracle, loading and lifetimes."""

import json
from dataclasses import asdict

import pytest
import torch
from safetensors.torch import save_file

from experiments.loopkv.checkpoint import load_model
from tests.reference.hrm_text import dense_hrm_reference
from vllm_rlt import CacheConfig, ExecutionConfig, ExitConfig, SamplingParams, SchedulerConfig
from vllm_rlt.core.alias_kv_cache import AliasKVCacheManager
from vllm_rlt.core.compact_kv_cache import CompactKVCacheManager
from vllm_rlt.core.kv_cache_manager import KVCacheManager
from vllm_rlt.engine.llm_engine import LLMEngine
from vllm_rlt.models import AutoModelForCausalLM, HrmTextConfig, HrmTextForCausalLM


@pytest.fixture
def model():
    torch.manual_seed(71)
    return HrmTextForCausalLM(
        HrmTextConfig(
            vocab_size=32,
            n_embd=16,
            intermediate_size=24,
            module_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=8,
            H_cycles=2,
            L_cycles=2,
            max_position_embeddings=64,
            embedding_scale=4,
        )
    ).eval()


@pytest.mark.parametrize("kind", ["native", "alias", "compact", "credits"])
def test_prefix_and_incremental_logits_kv_match_dense(model, kind):
    cfg = model.config
    cache_cls = {
        "native": KVCacheManager,
        "alias": AliasKVCacheManager,
        "compact": CompactKVCacheManager,
        "credits": CompactKVCacheManager,
    }[kind]
    cache = cache_cls(
        num_layers=cfg.num_hidden_layers,
        num_kv_heads=2,
        head_dim=8,
        num_blocks=32,
        block_size=2,
        max_loops=cfg.H_cycles,
        **({"reclaim_skipped_credits": True} if kind == "credits" else {}),
    )
    tokens = torch.tensor([3, 7, 4, 2, 6, 8, 9])
    states, logits, kv = dense_hrm_reference(model, tokens, prompt_length=5)
    for _ in range(2):
        assert cache.allocate("r", 7)
        for positions in [list(range(5)), [5], [6]]:
            hidden = model.prelude(tokens[positions])
            for depth in range(cfg.H_cycles):
                batch = cache._prepare_batch(
                    ["r"] * len(positions),
                    [depth] * len(positions),
                    positions,
                    read_lengths=[5] * 5 if len(positions) == 5 else None,
                )
                hidden, _ = model.recurrent_prepared(hidden, batch, cache)
                torch.testing.assert_close(hidden, states[depth][positions], atol=3e-6, rtol=3e-5)
            torch.testing.assert_close(model.coda(hidden), logits[positions], atol=3e-6, rtol=3e-5)
            for position in positions:
                cache.finalize_token("r", position, cfg.H_cycles - 1)
        for (depth, layer), (keys, values) in kv.items():
            actual = cache.read(layer, "r", depth, len(tokens))
            for result, expected in zip(actual, (keys, values)):
                torch.testing.assert_close(result, expected, atol=3e-6, rtol=3e-5)
        cache.free("r")
        assert cache.num_used_blocks == 0


@pytest.mark.parametrize("storage", ["native", "alias", "compact", "credits"])
@pytest.mark.parametrize("asynchronous", [False, True])
def test_atomic_ragged_prefill_engine_and_id_reuse(model, storage, asynchronous):
    prompts = [[3, 2, 7, 6, 4], [8, 3], [7, 4, 2, 9]]
    expected = []
    for prompt in prompts:
        tokens = list(prompt)
        for _ in range(4):
            _, logits, _ = dense_hrm_reference(model, torch.tensor(tokens), len(prompt))
            tokens.append(int(logits[-1].argmax()))
        expected.append(tokens[len(prompt) :])
    engine = LLMEngine(
        model,
        cache_config=CacheConfig(
            128,
            2,
            alias_last_exited=storage == "alias",
            compact_last_exited=storage in ("compact", "credits"),
            reclaim_skipped_credits=storage == "credits",
        ),
        scheduler_config=SchedulerConfig(
            max_num_seqs=3, max_num_batched_tokens=6, prefill_chunk_size=1
        ),
        execution_config=ExecutionConfig(async_scheduling=asynchronous),
        exit_config=ExitConfig(mode="ouro_delayed" if asynchronous else "ouro"),
    )
    rounds = []
    with torch.inference_mode():
        for _ in range(2):
            for i, prompt in enumerate(prompts):
                engine.add_request(
                    str(i),
                    prompt,
                    SamplingParams(
                        max_tokens=4, min_loops=2, max_loops=2, exit_threshold=1, ignore_eos=True
                    ),
                )
            finished = {}
            while engine.has_unfinished_requests():
                for output in engine.step():
                    if output.finished:
                        finished[output.request_id] = asdict(output)
            for i, target in enumerate(expected):
                assert finished[str(i)]["token_ids"] == target
                assert finished[str(i)]["exit_depths"] == [2] * 4
            assert engine.cache_manager.num_used_blocks == 0
            rounds.append(finished)
    assert rounds[0] == rounds[1]


def test_checkpoint_dispatch_and_standalone_reader(tmp_path, model):
    (tmp_path / "config.json").write_text(json.dumps(model.config.to_dict()))
    save_file(model.state_dict(), tmp_path / "model.safetensors")
    loaded = AutoModelForCausalLM.from_pretrained(tmp_path, dtype=torch.float32)
    standalone = load_model(tmp_path, torch.device("cpu"), torch.float32)
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, loaded.state_dict()[name])
        assert torch.equal(tensor, standalone.state_dict()[name])
    # RoPE constants must stay FP32 when weights are cast to BF16.
    original = loaded.model.rotary_emb.inv_freq.clone()
    loaded.to(torch.bfloat16)
    assert torch.equal(original, loaded.model.rotary_emb.inv_freq)


def test_prefix_prompt_limit_and_full_depth_contract(model):
    engine = LLMEngine(
        model,
        cache_config=CacheConfig(32, 2),
        scheduler_config=SchedulerConfig(max_num_batched_tokens=4),
    )
    with pytest.raises(ValueError, match="atomic prompt"):
        engine.add_request("oversized", [1] * 5)
    with pytest.raises(ValueError, match="full H_cycles"):
        engine.add_request("shallow", [1], SamplingParams(min_loops=1, max_loops=1))
    assert not engine.has_unfinished_requests()


@pytest.mark.parametrize("cache_cls", [KVCacheManager, AliasKVCacheManager, CompactKVCacheManager])
def test_full_read_lengths_reject_unwritten_prefix_and_invalid_lengths(cache_cls):
    cache = cache_cls(
        num_layers=1, num_kv_heads=1, head_dim=4, num_blocks=8, block_size=2, max_loops=2
    )
    cache.allocate("r", 5)
    with pytest.raises(ValueError, match="read lengths"):
        cache._prepare_batch(["r"], [0], [0], read_lengths=[6])
    batch = cache._prepare_batch(["r"], [0], [0], read_lengths=[5])
    value = torch.ones(1, 1, 4)
    cache._write_prepared(0, batch, value, value)
    with pytest.raises(RuntimeError, match="uninitialized"):
        cache._attend_prepared(0, batch, value)
