"""A cached-prefix benchmark must generate the same outputs as actual prefill."""

import pytest
import torch

from benchmarks.context_sweep import build_prefix, measure, restore_prefix
from vllm_lt import CacheConfig, ExitConfig, SamplingParams, SchedulerConfig
from vllm_lt.engine.llm_engine import LLMEngine
from vllm_lt.models import OuroConfig, OuroForCausalLM


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
@pytest.mark.parametrize("buffer_mode", ["dynamic", "static", "static-pad"])
@pytest.mark.parametrize("policy", ["fixed4", "mixed234"])
def test_replicated_prefix_and_repeated_restores_match_normal_generation(
    policy, device, buffer_mode
):
    torch.manual_seed(123)
    model = OuroForCausalLM(OuroConfig.tiny()).to(device)
    options = dict(
        attention_backend="triton" if device == "cuda" else "torch",
        cache_config=CacheConfig(num_blocks=24),
        exit_config=ExitConfig("ouro_delayed"),
        scheduler_config=SchedulerConfig(
            max_num_seqs=3, max_num_batched_tokens=8, prefill_chunk_size=8
        ),
    )
    prompt = [2, 3, 4, 5] * 4
    reference = LLMEngine(model, **options)
    for i in range(3):
        depth = 4 if policy == "fixed4" else 2 + i % 3
        params = SamplingParams(
            max_tokens=4,
            ignore_eos=True,
            min_loops=depth - 1 if depth < 4 else 2,
            exit_threshold=0 if depth < 4 else 1,
        )
        reference.add_request(str(i), prompt, params)
    expected = {}
    while reference.has_unfinished_requests():
        for out in reference.step():
            if out.finished:
                expected[out.request_id] = dict(
                    token_ids=out.token_ids, exit_depths=out.exit_depths
                )
    engine = LLMEngine(model, **options)
    checkpoint = build_prefix(engine, prompt, 3, 4)
    for variant in ["S", "AS", "AM", "AS"]:
        restore_prefix(engine, checkpoint, 3, variant, policy, 8, buffer_mode=buffer_mode)
        warmup, _ = measure(engine, 3, 4, policy)
        runner = engine.model_runner
        restore_prefix(
            engine, checkpoint, 3, variant, policy, 8, reuse_runner=True, buffer_mode=buffer_mode
        )
        assert engine.model_runner is runner
        measured, _ = measure(engine, 3, 4, policy)
        assert warmup == measured == expected
        assert engine.cache_manager.num_used_blocks == 0
