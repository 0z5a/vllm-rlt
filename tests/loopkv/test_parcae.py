"""Parcae recurrence, token-value identity, checkpoint and request lifetimes."""

import json

import pytest
import torch

from experiments.loopkv.checkpoint import load_model
from tests.reference.parcae import dense_parcae_reference
from vllm_rlt import CacheConfig, ExecutionConfig, SamplingParams, SchedulerConfig
from vllm_rlt.core.alias_kv_cache import AliasKVCacheManager
from vllm_rlt.core.compact_kv_cache import CompactKVCacheManager
from vllm_rlt.core.kv_cache_manager import KVCacheManager
from vllm_rlt.engine.llm_engine import LLMEngine
from vllm_rlt.models import AutoModelForCausalLM, ParcaeConfig, ParcaeForCausalLM


def tiny_config():
    return ParcaeConfig(
        n_embd=32,
        intermediate_size=64,
        num_attention_heads=4,
        num_key_value_heads=4,
        n_layers_in_prelude=2,
        n_layers_in_recurrent_block=2,
        n_layers_in_coda=2,
        mean_recurrence=3,
        block_size=64,
        vocab_size=1024,
    )


@pytest.fixture
def model():
    torch.manual_seed(41)
    return ParcaeForCausalLM(tiny_config()).eval()


def make_cache(model, mode):
    config = model.config
    cls = {
        "native": KVCacheManager,
        "alias": AliasKVCacheManager,
        "compact": CompactKVCacheManager,
        "credits": CompactKVCacheManager,
    }[mode]
    options = (
        {"reclaim_skipped_credits": mode == "credits"} if mode in ("compact", "credits") else {}
    )
    return cls(
        config.num_hidden_layers,
        config.num_key_value_heads,
        config.head_dim,
        128,
        2,
        max_loops=config.mean_recurrence,
        dtype=next(model.parameters()).dtype,
        recurrent_layers=model.recurrent_kv_layers,
        **options,
    )


@pytest.mark.parametrize("mode", ["native", "alias", "compact", "credits"])
@pytest.mark.parametrize("chunks", [[[0, 1, 2, 3, 4]], [[0, 1], [2, 3], [4]]])
def test_states_logits_every_kv_match_dense_with_chunking(model, mode, chunks):
    tokens = torch.tensor([257, 511, 9, 768, 1023])
    state = torch.linspace(-0.1, 0.1, len(tokens) * model.config.n_embd).reshape(len(tokens), -1)
    expected_states, expected_logits, expected_kv = dense_parcae_reference(model, tokens, state)
    cache = make_cache(model, mode)
    cache.allocate("a", len(tokens))
    with torch.inference_mode():
        for positions in chunks:
            batch = cache._prepare_batch(["a"] * len(positions), [0] * len(positions), positions)
            hidden = model.prelude_prepared(tokens[positions], batch, cache)
            hidden = torch.cat((state[positions], hidden[:, model.config.n_embd :]), -1)
            for depth, expected in enumerate(expected_states):
                core = cache._prepare_batch(
                    ["a"] * len(positions), [depth] * len(positions), positions
                )
                hidden, _ = model.recurrent_prepared(hidden, core, cache)
                torch.testing.assert_close(
                    hidden[:, : model.config.n_embd], expected[positions], atol=3e-6, rtol=3e-5
                )
            for position in positions:
                cache.finalize_token("a", position, model.config.mean_recurrence - 1)
            logits = model.coda_prepared(hidden, batch, cache)
            torch.testing.assert_close(logits, expected_logits[positions], atol=3e-6, rtol=3e-5)
        for (layer, depth), expected in expected_kv.items():
            observed = cache.read(layer, "a", depth, len(tokens))
            for actual, value in zip(observed, expected):
                torch.testing.assert_close(actual, value, atol=3e-6, rtol=3e-5)
    cache.free("a")
    assert cache.num_free_blocks == 128


def test_bf16_token_bytes_and_native_random_initialization(model):
    freqs = model.freqs_cis.clone()
    model.to(torch.bfloat16)
    assert model.freqs_cis.dtype == torch.float32 and torch.equal(model.freqs_cis, freqs)
    cache = make_cache(model, "native")
    cache.allocate("a", 4)
    tokens = torch.tensor([255, 256, 511, 1023])
    batch = cache._prepare_batch(["a"] * 4, [0] * 4, list(range(4)))
    torch.manual_seed(109)
    expected = torch.randn(4, model.config.n_embd, dtype=torch.bfloat16)
    std = model.config.initializer_range
    torch.nn.init.trunc_normal_(expected, std=std, a=-3 * std, b=3 * std)
    torch.manual_seed(109)
    with torch.inference_mode():
        hidden = model.prelude_prepared(tokens, batch, cache)
    assert torch.equal(hidden[:, : model.config.n_embd], expected)
    assert torch.equal(hidden[:, -2].long() + 256 * hidden[:, -1].long(), tokens)
    assert torch.count_nonzero(expected) > 0


def test_engine_four_modes_reuse_ids_preserves_stochastic_state(model):
    all_modes = []
    for mode in ("native", "alias", "compact", "credits"):
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
        )
        rounds = []
        for _ in range(2):
            torch.manual_seed(83)
            for rid, prompt in [("a", [257, 5, 511]), ("b", [1023, 256])]:
                engine.add_request(
                    rid,
                    prompt,
                    SamplingParams(
                        max_tokens=5, min_loops=3, max_loops=3, exit_threshold=1, ignore_eos=True
                    ),
                )
            completed = {}
            while engine.has_unfinished_requests():
                for output in engine.step():
                    if output.finished:
                        completed[output.request_id] = (output.token_ids, output.exit_depths)
            assert len(completed) == 2 and engine.cache_manager.num_free_blocks == 128
            assert all(exits == [3] * 5 for _, exits in completed.values())
            rounds.append(completed)
        assert rounds[0] == rounds[1]
        all_modes.append(rounds[0])
        engine.close()
    assert all(result == all_modes[0] for result in all_modes)


def test_strict_torch_checkpoint_auto_and_standalone_load(tmp_path, model):
    config = model.config.to_dict()
    del config["model_type"]  # The published checkpoint identifies _class_name instead.
    (tmp_path / "config.json").write_text(json.dumps(config))
    torch.save(model.state_dict(), tmp_path / "pytorch_model.bin")
    for loaded in (
        AutoModelForCausalLM.from_pretrained(tmp_path, dtype=torch.float32),
        load_model(tmp_path, torch.device("cpu"), torch.float32),
    ):
        assert isinstance(loaded, ParcaeForCausalLM)
        assert loaded.lm_head.weight is loaded.transformer.wte.weight
        for name, expected in model.state_dict().items():
            torch.testing.assert_close(loaded.state_dict()[name], expected, rtol=0, atol=0)
    weights = dict(model.state_dict())
    del weights["transformer.adapter.B"]
    torch.save(weights, tmp_path / "pytorch_model.bin")
    with pytest.raises(RuntimeError, match="Missing key"):
        ParcaeForCausalLM.from_pretrained(tmp_path)


def test_reject_shallow_and_unqualified_async(model):
    engine = LLMEngine(model, cache_config=CacheConfig(128, 2))
    with pytest.raises(ValueError, match="full recurrence"):
        engine.add_request("a", [257], SamplingParams(max_tokens=2, max_loops=2, exit_threshold=1))
    with pytest.raises(ValueError, match="synchronous"):
        LLMEngine(model, execution_config=ExecutionConfig(async_scheduling=True))
