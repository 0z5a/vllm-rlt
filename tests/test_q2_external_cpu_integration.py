"""One actual tiny native/official driver pair on CPU, without GPU qualification."""

import importlib.metadata
import importlib.util
import socket
import time

import pytest
import torch

from vllm_lt.benchmarks.q2_external_driver import execute_case, pool_descriptor, require_empty
from vllm_lt.config import CacheConfig, SchedulerConfig
from vllm_lt.engine.llm_engine import LLMEngine
from vllm_lt.models.config import OuroConfig
from vllm_lt.models.ouro import OuroForCausalLM
from vllm_lt.validation.official_cached import OfficialOuroCachedReference


def _compatible():
    try:
        return (
            importlib.metadata.version("transformers") == "4.55.0"
            and importlib.util.find_spec("kernels") is None
        )
    except importlib.metadata.PackageNotFoundError:
        return False


@pytest.mark.skipif(not _compatible(), reason="requires prepared Transformers 4.55.0 reference env")
def test_real_cpu_native_and_official_execute_case_share_weights_and_release_requests(monkeypatch):
    started = time.perf_counter_ns()
    deadline = started + 120 * 10**9

    def forbidden(*args, **kwargs):
        raise AssertionError("CPU integration must not discover/initialize CUDA or use network")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    for name in ("is_available", "device_count", "current_device", "init", "_lazy_init"):
        monkeypatch.setattr(torch.cuda, name, forbidden)
    # Only GPU timing/memory bookkeeping is replaced. Model, attention, cache,
    # scheduler, sampling, collector and adapter lifecycle remain real CPU code.
    for name in (
        "synchronize",
        "reset_peak_memory_stats",
        "memory_allocated",
        "memory_reserved",
        "max_memory_allocated",
        "max_memory_reserved",
    ):
        monkeypatch.setattr(torch.cuda, name, lambda *args, **kwargs: 0)

    torch.manual_seed(17)
    config = OuroConfig.tiny(
        vocab_size=8,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=24,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=256,
    )
    model = OuroForCausalLM(config).eval()
    model.requires_grad_(False)
    engine = LLMEngine(
        model,
        cache_config=CacheConfig(num_blocks=48, block_size=16),
        scheduler_config=SchedulerConfig(max_num_seqs=1, max_num_batched_tokens=128),
        attention_backend="torch",
    )
    weights = dict(model.named_parameters())
    official = OfficialOuroCachedReference(config.to_dict(), weights)
    assert all(
        value.data_ptr() == weights[name].data_ptr()
        for name, value in official.model.named_parameters()
    )
    assert all(value.device.type == "cpu" for value in model.parameters())
    pool = pool_descriptor(engine)
    prompt = [1 + index % 7 for index in range(128)]
    plan = {
        "plan_sha256": "cpu-integration-only-no-device-plan",
        "workload": {"requests": [{"prompt_token_ids": prompt, "max_output_tokens": 64}]},
        "contract": {
            "engine": {
                "sampling": {
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "top_k": -1,
                    "seed": 0,
                    "min_loops": 4,
                    "max_loops": 4,
                    "exit_threshold": 1.0,
                    "ignore_eos": True,
                }
            }
        },
    }
    results = []
    try:
        for implementation, bound in (("native", 380), ("official", 64)):
            result = execute_case(
                plan,
                {"implementation_id": implementation, "phase": "feasibility", "max_steps": bound},
                engine,
                official,
                started_ns=time.perf_counter_ns(),
                deadline_ns=deadline,
            )
            results.append(result)
            assert result["status"] == "complete" and not result["failures"]
            assert result["before_request"] == result["cleanup"] == require_empty(engine, official)
            assert result["cleanup"] == {
                "native_requests": 0,
                "native_used_blocks": 0,
                "official_cache_slots": 0,
            }
            assert pool_descriptor(engine) == pool
            assert official.status == "ready" and official.cache is None
            row = result["requests"][0]
            assert row["finished"] and row["finish_reason"] == "length"
            assert len(row["token_ids"]) == 64 and row["exit_depths"] == [4] * 64
            assert row["token_timestamps_ns"] == sorted(row["token_timestamps_ns"])
            assert result["arrival_ns"] <= row["token_timestamps_ns"][0]
            assert row["token_timestamps_ns"][-1] <= result["synchronized_ns"]
            assert result["metrics"]["generated_tokens"] == 64
            assert len([e for e in result["events"] if e["kind"] == "token_emitted"]) == 64

        native, external = results
        assert native["requests"][0]["token_ids"] == external["requests"][0]["token_ids"]
        assert native["counts"]["steps"] == 380
        assert native["counts"]["stage_counts"] == {
            "prefill": 1,
            "prelude": 63,
            "recurrent": 252,
            "coda": 64,
        }
        # The hook also observes four recurrent traversals inside prefill.
        assert native["feasibility"]["finite_checks"] == {"recurrent": 256, "coda": 64}
        assert native["official_cache"] is None
        assert external["counts"]["steps"] == external["counts"]["official_logits_checked"] == 64
        cached = external["official_cache"]
        snapshot, summary = cached["snapshot"], cached["final_summary"]
        assert snapshot["final_summary"] == summary
        assert snapshot["status"] == "awaiting_completion"
        assert snapshot["output_count"] == snapshot["forward_calls"] == 64
        assert snapshot["cache_slots"] == summary["slot_count"] == summary["max_cache_size"] == 96
        assert summary["lengths"] == [191] * 96
        assert summary["key_shapes"] == summary["value_shapes"] == [[1, 1, 191, 8]] * 96
        assert summary["distinct_storage"] is True and summary["all_finite"] is True
        assert summary["device"] == "cpu" and summary["dtype"] == "torch.float32"
        assert [r["position"] for r in snapshot["calls"]] == [0] + list(range(128, 191))
        assert [r["input_count"] for r in snapshot["calls"]] == [128] + [1] * 63
        assert [r["last_input_position"] for r in cached["calls"]] == list(range(127, 191))
        assert [r["token_id"] for r in cached["calls"]] == external["requests"][0]["token_ids"]
        assert time.perf_counter_ns() < deadline
    finally:
        official.close(completion_confirmed=True)
