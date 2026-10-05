"""LoopFormer learned positions, time conditioning, causal caching and lifetimes."""

import json
from dataclasses import asdict

import pytest
import torch
from safetensors.torch import save_file

from experiments.loopkv.checkpoint import load_model
from tests.reference.loopformer import dense_loopformer_reference
from vllm_rlt import CacheConfig, ExecutionConfig, ExitConfig, SamplingParams, SchedulerConfig
from vllm_rlt.core.alias_kv_cache import AliasKVCacheManager
from vllm_rlt.core.compact_kv_cache import CompactKVCacheManager
from vllm_rlt.core.kv_cache_manager import KVCacheManager
from vllm_rlt.engine.llm_engine import LLMEngine
from vllm_rlt.models import AutoModelForCausalLM, LoopFormerConfig, LoopFormerForCausalLM


@pytest.fixture
def model():
    torch.manual_seed(92)
    return LoopFormerForCausalLM(
        LoopFormerConfig(
            vocab_size=32,
            n_embd=16,
            n_head=2,
            n_layer=2,
            intermediate_dim=24,
            block_size=64,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
        )
    ).eval()


@pytest.mark.parametrize("storage", ["native", "alias", "compact", "credits"])
def test_cached_states_logits_all_depth_kv(model, storage):
    cls = {
        "native": KVCacheManager,
        "alias": AliasKVCacheManager,
        "compact": CompactKVCacheManager,
        "credits": CompactKVCacheManager,
    }[storage]
    cache = cls(
        2, 2, 8, 64, 2, 8, **({"reclaim_skipped_credits": True} if storage == "credits" else {})
    )
    tokens = torch.tensor([3, 7, 2, 9, 5, 4, 8])
    states, logits, kv = dense_loopformer_reference(model, tokens)
    with torch.inference_mode():
        for _ in range(2):
            assert cache.allocate("r", len(tokens))
            for positions in [[0, 1, 2], [3, 4], [5], [6]]:
                hidden = model.prelude(tokens[positions])
                for depth in range(8):
                    hidden, _ = model.recurrent(
                        hidden, ["r"] * len(positions), [depth] * len(positions), positions, cache
                    )
                    torch.testing.assert_close(
                        hidden, states[depth][positions], atol=3e-6, rtol=3e-5
                    )
                torch.testing.assert_close(
                    model.coda(hidden), logits[positions], atol=3e-6, rtol=3e-5
                )
                for position in positions:
                    cache.finalize_token("r", position, 7)
            for (depth, layer), expected in kv.items():
                for actual, target in zip(cache.read(layer, "r", depth, len(tokens)), expected):
                    torch.testing.assert_close(actual, target, atol=3e-6, rtol=3e-5)
            cache.free("r")
            assert cache.num_used_blocks == 0


@pytest.mark.parametrize("storage", ["native", "alias", "compact", "credits"])
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("batched", [False, True])
def test_ragged_clock_routing_and_two_id_lifetimes(model, storage, asynchronous, batched):
    prompts = [[3, 7, 2], [9], [5, 4, 8, 2, 3]]
    expected = []
    for prompt in prompts:
        sequence = prompt.copy()
        for _ in range(4):
            _, logits, _ = dense_loopformer_reference(model, torch.tensor(sequence))
            sequence.append(int(logits[-1].argmax()))
        expected.append(sequence[len(prompt) :])
    engine = LLMEngine(
        model,
        cache_config=CacheConfig(
            192,
            2,
            alias_last_exited=storage == "alias",
            compact_last_exited=storage in ("compact", "credits"),
            reclaim_skipped_credits=storage == "credits",
        ),
        scheduler_config=SchedulerConfig(
            max_num_seqs=3, max_num_batched_tokens=6, prefill_chunk_size=2
        ),
        execution_config=ExecutionConfig(
            async_scheduling=asynchronous, prefill_batch_metadata=batched
        ),
        exit_config=ExitConfig("ouro_delayed" if asynchronous else "ouro"),
    )
    rounds = []
    with torch.inference_mode():
        for _ in range(2):
            for index, prompt in enumerate(prompts):
                engine.add_request(
                    str(index),
                    prompt,
                    SamplingParams(
                        max_tokens=4, min_loops=8, max_loops=8, exit_threshold=1, ignore_eos=True
                    ),
                )
            completed = {}
            while engine.has_unfinished_requests():
                for output in engine.step():
                    if output.finished:
                        completed[output.request_id] = asdict(output)
            for index, target in enumerate(expected):
                assert completed[str(index)]["token_ids"] == target
                assert completed[str(index)]["exit_depths"] == [8] * 4
            assert engine.cache_manager.num_used_blocks == 0
            rounds.append(completed)
    assert rounds[0] == rounds[1]


def test_strict_checkpoint_and_standalone_loader(tmp_path, model):
    (tmp_path / "config.json").write_text(json.dumps(model.config.to_dict()))
    save_file(model.state_dict(), tmp_path / "model.safetensors")
    for loaded in (
        AutoModelForCausalLM.from_pretrained(tmp_path, dtype=torch.float32),
        load_model(tmp_path, torch.device("cpu"), torch.float32),
    ):
        assert loaded.state_dict().keys() == model.state_dict().keys()
        for name, value in model.state_dict().items():
            assert torch.equal(value, loaded.state_dict()[name])


def test_fixed_steps_and_dtype_epsilon(model):
    engine = LLMEngine(model)
    with pytest.raises(ValueError, match="eight steps"):
        engine.add_request("shallow", [1, 3], SamplingParams(min_loops=1, max_loops=4))
    model.bfloat16()
    block = model.gpt.transformer.h.blocks[0]
    assert block.norm_1.eps is None and model.gpt.transformer.norm_f.eps is None
    hidden = torch.randn(3, 16).bfloat16()
    assert torch.equal(block.norm_1(hidden), torch.nn.functional.rms_norm(hidden, (16,)))
