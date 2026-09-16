"""Latency formulas and real-gate prefix benchmark equivalence."""

from dataclasses import replace

import pytest
import torch

from benchmarks.feature_stack import (
    build_independent_prefixes,
    configure,
    latency_metrics,
    params_for,
    restore_independent_prefixes,
    run_trial,
)
from vllm_lt import CacheConfig, ExitConfig, SchedulerConfig
from vllm_lt.engine.llm_engine import LLMEngine
from vllm_lt.kernels.flash_attention import FlashPagedAttention
from vllm_lt.kernels.paged_attention import torch_paged_attention, triton_paged_attention
from vllm_lt.models import OuroConfig, OuroForCausalLM


def test_latency_populations_and_completion_notification():
    rows = [
        dict(start_seconds=0, admitted_seconds=0.5, token_times_seconds=[1, 2, 5], end_seconds=6),
        dict(start_seconds=2, admitted_seconds=2, token_times_seconds=[4, 5, 6], end_seconds=6),
    ]
    result = latency_metrics(rows, "e2e")
    assert result["ttft_seconds"] == dict(p50=1, p95=2, p99=2)
    assert result["tpot_seconds"] == dict(p50=1, p95=2, p99=2)
    assert result["itl_seconds"] == dict(p50=1, p95=3, p99=3)
    assert result["e2e_seconds"] == dict(p50=4, p95=6, p99=6)
    assert result["itl_samples"] == 4
    decoded = latency_metrics(rows, "decode")
    assert decoded["ttft_seconds"] is None and decoded["e2e_seconds"] is None
    assert decoded["tpot_seconds"] == result["tpot_seconds"]


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_independent_prefix_real_gate_matches_e2e_all_stages(device):
    torch.manual_seed(73)
    model = OuroForCausalLM(replace(OuroConfig.tiny(), head_dim=64)).to(
        device=device, dtype=torch.bfloat16 if device == "cuda" else torch.float32
    )
    options = dict(
        cache_config=CacheConfig(num_blocks=32, block_size=16),
        scheduler_config=SchedulerConfig(
            max_num_seqs=3, max_num_batched_tokens=8, prefill_chunk_size=4
        ),
        exit_config=ExitConfig("ouro"),
        attention_backend="triton" if device == "cuda" else "torch",
    )
    engine = LLMEngine(model, **options)
    cached = LLMEngine(model, **options)
    prompts = [[2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14, 15, 16]]
    checkpoint = build_independent_prefixes(cached, prompts, 4)
    if device == "cuda":
        flash = FlashPagedAttention(torch.device(device), torch.bfloat16, 64, 16)
        choices = {
            "triton": (triton_paged_attention, {"backend": "triton"}),
            "flash_attn": (flash, flash.info),
        }
    else:
        choices = {
            b: (torch_paged_attention, {"backend": "torch-test"}) for b in ("triton", "flash_attn")
        }
    for stage in ["P0", "P1", "P2", "P3", "P4", "P5", "P6", "P7"]:
        params = params_for(stage, 4, 0.2)
        configure(engine, stage, 3, 8, choices)
        expected, _ = run_trial(engine, prompts + prompts, 3, params, "e2e", 6)
        configure(cached, stage, 3, 8, choices)
        restore_independent_prefixes(cached, checkpoint, 3, params)
        warm, _ = run_trial(cached, prompts, 3, params, "decode", 3)
        configure(cached, stage, 3, 8, choices, reuse=True)
        restore_independent_prefixes(cached, checkpoint, 3, params)
        actual, summary = run_trial(cached, prompts, 3, params, "decode", 3)

        def outputs(records, first=False):
            return {
                r["request_id"]: (r["token_ids"], r["exit_depths"])
                for r in records
                if not first or int(r["request_id"]) < 3
            }

        assert outputs(actual) == outputs(warm) == outputs(expected, True)
        assert summary["ttft_seconds"] is None
