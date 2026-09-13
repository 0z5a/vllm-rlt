import copy
import json
import math

import pytest
import torch

from benchmarks.observe import (
    RunCollector,
    instrument_engine,
    populated_page_stats,
    summarize_records,
)
from vllm_lt import CacheConfig, SamplingParams, SchedulerConfig
from vllm_lt.core.kv_cache_manager import KVCacheManager
from vllm_lt.engine.llm_engine import LLMEngine
from vllm_lt.models import OuroConfig, OuroForCausalLM
from vllm_lt.request import RequestOutput


def output(key, tokens, depths=None, *, finished=False, reason=None):
    return RequestOutput(key, [1], tokens, depths or [4] * len(tokens), finished, reason)


def test_independent_hand_calculated_timing_fixture():
    t0, ms = 1234, 1_000_000
    collector = RunCollector(["a", "b"], arrival_ns=t0)
    collector.observe_submission("a", started_ns=t0, ended_ns=t0 + 10)
    collector.observe_submission("b", started_ns=t0 + 10, ended_ns=t0 + 20)
    collector.observe_admission("a", admitted_ns=t0 + 100, reserved_pages=4)
    collector.observe_prefill("a", dispatched_ns=t0 + 120, step_id=0)
    collector.observe_outputs([output("a", [1])], step_id=0, returned_ns=t0 + 10 * ms)
    collector.observe_outputs(
        [output("a", [1, 2], [4, 2]), output("b", [3])],
        step_id=1,
        returned_ns=t0 + 20 * ms,
    )
    collector.observe_outputs(
        [
            output("a", [1, 2, 4], [4, 2, 3], finished=True, reason="length"),
            output("b", [3, 5], [4, 4], finished=True, reason="length"),
        ],
        step_id=2,
        returned_ns=t0 + 40 * ms,
    )
    result = collector.finish(synchronized_ns=t0 + 45 * ms)
    metrics = result["metrics"]
    assert metrics["generated_tokens"] == 5
    assert metrics["delivery_wall_ns"] == 40 * ms
    assert metrics["synchronized_wall_ns"] == 45 * ms
    assert metrics["generated_tokens_per_second"] == 125
    assert metrics["mean_decode_depth"] == 3
    assert metrics["token_weighted_tpot_ns"] == pytest.approx(50 * ms / 3)
    assert metrics["per_request"]["a"] == {
        "ttft_ns": 10 * ms,
        "completion_latency_ns": 40 * ms,
        "tpot_ns": 15 * ms,
        "token_gaps_ns": [10 * ms, 20 * ms],
        "admission_queue_ns": 100,
        "enqueue_ns": 10,
        "admitted_to_first_prefill_ns": 20,
    }
    assert metrics["per_request"]["b"]["ttft_ns"] == 20 * ms
    assert metrics["per_request"]["b"]["tpot_ns"] == 20 * ms
    emitted = [event for event in result["events"] if event["kind"] == "token_emitted"]
    assert [event["host_offset_ns"] for event in emitted] == [
        10 * ms,
        20 * ms,
        20 * ms,
        40 * ms,
        40 * ms,
    ]
    assert metrics == summarize_records(
        result["requests"], arrival_ns=t0, synchronized_ns=t0 + 45 * ms
    )


def test_cumulative_output_deduplication_and_deferred_prefix_validation():
    collector = RunCollector(["a"], arrival_ns=0)
    collector.observe_outputs([output("a", [1])], step_id=0, returned_ns=1)
    collector.observe_outputs([output("a", [1])], step_id=1, returned_ns=2)
    assert collector.finish(synchronized_ns=3)["metrics"]["generated_tokens"] == 1
    collector.observe_outputs([output("a", [9, 2])], step_id=2, returned_ns=4)
    with pytest.raises(ValueError, match="prefix changed"):
        collector.finish(synchronized_ns=5)


def test_start_moves_origin_before_observation_only():
    collector = RunCollector(["a"], arrival_ns=100)
    collector.start(1_000)
    collector.observe_submission("a", started_ns=1_010, ended_ns=1_020)
    collector.observe_outputs([output("a", [1])], step_id=0, returned_ns=1_050)
    result = collector.finish(synchronized_ns=1_070)
    assert result["events"][0]["host_offset_ns"] == 20
    assert result["metrics"]["per_request"]["a"]["ttft_ns"] == 50
    assert result["metrics"]["synchronized_wall_ns"] == 70
    with pytest.raises(ValueError, match="cannot restart"):
        collector.start(2_000)

    empty_step = RunCollector(["a"], arrival_ns=0)
    empty_step.observe_outputs([], step_id=0, returned_ns=1)
    with pytest.raises(ValueError, match="cannot restart"):
        empty_step.start(5)


def test_partial_snapshot_survives_bad_prefix_and_unavailable_sync():
    collector = RunCollector(["a"], arrival_ns=0)
    collector.observe_outputs([output("a", [1])], step_id=0, returned_ns=1)
    collector.observe_outputs([output("a", [9, 2])], step_id=1, returned_ns=2)
    with pytest.raises(ValueError, match="prefix changed"):
        collector.finish(synchronized_ns=3)
    raw = collector.snapshot(synchronized_ns=None)
    assert raw["requests"][0]["token_ids"] == [1, 2]
    assert len(raw["events"]) == 2
    assert raw["metrics"] is None
    assert "unavailable" in raw["metrics_error"]
    # A diagnostic snapshot deliberately does not claim to validate prefixes.
    assert collector.snapshot(synchronized_ns=3)["metrics"]["generated_tokens"] == 2
    raw["requests"][0]["token_ids"].append(99)
    raw["events"][0]["token_id"] = 99
    fresh = collector.snapshot(synchronized_ns=3)
    assert fresh["requests"][0]["token_ids"] == [1, 2]
    assert fresh["events"][0]["token_id"] == 1


def test_partial_snapshot_retains_raw_records_when_metrics_are_invalid():
    collector = RunCollector(["a"], arrival_ns=0)
    collector.observe_outputs([output("a", [-1])], step_id=0, returned_ns=10)
    raw = collector.snapshot(synchronized_ns=20)
    assert raw["requests"][0]["token_ids"] == [-1]
    assert raw["events"][0]["token_id"] == -1
    assert raw["metrics"] is None
    assert "token ID" in raw["metrics_error"]
    with pytest.raises(ValueError, match="token ID"):
        collector.finish(synchronized_ns=20)
    earlier = collector.snapshot(synchronized_ns=5)
    assert earlier["metrics"] is None
    assert "synchronized_ns" in earlier["metrics_error"]


def test_null_partial_and_single_output_metrics_have_reasons():
    collector = RunCollector(["a", "b"], arrival_ns=100)
    empty = collector.finish(synchronized_ns=100)["metrics"]
    assert empty["generated_tokens_per_second"] is None
    assert "generated_tokens_per_second" in empty["unavailable"]
    collector.observe_outputs([output("a", [1])], step_id=0, returned_ns=110)
    result = collector.finish(synchronized_ns=120)["metrics"]
    assert result["per_request"]["a"]["completion_latency_ns"] is None
    assert result["per_request"]["a"]["tpot_ns"] is None
    assert result["mean_decode_depth"] is None
    assert "mean_decode_depth" in result["unavailable"]
    assert result["per_request"]["b"]["ttft_ns"] is None


@pytest.mark.parametrize(
    "bad",
    [
        [output("a", [1, 2])],
        [output("a", [1], [4, 4])],
        [output("a", [1]), output("a", [1])],
        [output("unknown", [1])],
    ],
)
def test_collector_rejects_incompatible_output_accounting(bad):
    collector = RunCollector(["a"], arrival_ns=0)
    with pytest.raises(ValueError):
        collector.observe_outputs(bad, step_id=0, returned_ns=1)


def test_event_storage_bound_and_monotonic_clock():
    collector = RunCollector(["a"], arrival_ns=10, max_events=1)
    collector.observe_outputs([output("a", [1])], step_id=0, returned_ns=20)
    with pytest.raises(ValueError, match="storage limit"):
        collector.observe_outputs([output("a", [1, 2])], step_id=1, returned_ns=30)
    with pytest.raises(ValueError):
        collector.observe_outputs([], step_id=2, returned_ns=15)


@pytest.mark.parametrize(
    "field,value",
    [
        ("token_timestamps_ns", [float("nan")]),
        ("exit_depths", [True]),
        ("token_ids", [-1]),
        ("finished", 1),
        ("admitted_ns", 1000),
    ],
)
def test_pure_summary_rejects_invalid_raw_records(field, value):
    collector = RunCollector(["a"], arrival_ns=0)
    collector.observe_outputs([output("a", [1])], step_id=0, returned_ns=1)
    rows = copy.deepcopy(collector.finish(synchronized_ns=2)["requests"])
    rows[0][field] = value
    with pytest.raises(ValueError):
        summarize_records(rows, arrival_ns=0, synchronized_ns=2)


class Clock:
    def __init__(self):
        self.now = 0

    def __call__(self):
        self.now += 1
        return self.now


def tiny_engine():
    torch.manual_seed(9)
    return LLMEngine(
        OuroForCausalLM(OuroConfig.tiny()),
        cache_config=CacheConfig(8, 2),
        scheduler_config=SchedulerConfig(max_num_batched_tokens=2),
    )


@pytest.mark.parametrize("profile", [False, True])
def test_instance_instrumentation_counts_work_and_restores_methods(profile):
    engine = tiny_engine()
    clock = Clock()
    collector = RunCollector(["a", "b"], arrival_ns=0)
    original = engine.scheduler.schedule
    params = SamplingParams(max_tokens=3, ignore_eos=True, exit_threshold=0)
    with instrument_engine(engine, collector, clock=clock, profile=profile, max_steps=100) as obs:
        for key in ["a", "b"]:
            start = clock()
            engine.add_request(key, [2, 3], params)
            collector.observe_submission(key, started_ns=start, ended_ns=clock())
        while engine.has_unfinished_requests():
            outputs = engine.step()
            collector.observe_outputs(outputs, step_id=obs.step_id, returned_ns=clock())
        result = collector.finish(synchronized_ns=clock())
        assert obs.validate_gate_probabilities() == {"count": 8, "finite": True}
        summary = obs.summary()
    assert engine.scheduler.schedule == original
    assert "schedule" not in vars(engine.scheduler)
    assert "execute" not in vars(engine.model_runner)
    assert engine.cache_manager.num_used_blocks == 0
    admissions = [event for event in result["events"] if event["kind"] == "admitted"]
    assert [event["request_id"] for event in admissions] == ["a", "b"]
    # B waits for A's pages; unsuccessful allocate calls are never admissions.
    assert admissions[1]["host_offset_ns"] > result["requests"][0]["token_timestamps_ns"][1]
    assert summary["recurrent_depth_counts"] == {"1": 4, "2": 4}
    assert summary["request_work"]["a"] == {
        "prefill": 2,
        "prefill_token_traversals": 8,
        "coda": 3,
        "prelude": 2,
        "recurrent": 4,
    }
    assert summary["logical_copy_bytes"] == 8 * (engine.cache_manager.bytes_per_block // 2)
    if profile:
        snapshot = summary["snapshots"][0]
        assert snapshot["batch"]["rows"][0]["output_index"] == 0
        assert snapshot["batch"]["rows"][0]["input_position"] is None
        assert snapshot["schedule_start_ns"] <= snapshot["execute_start_ns"]
        assert snapshot["execute_end_ns"] <= snapshot["step_return_ns"]
        assert snapshot["kv_after_execute"] == {
            "populated_pages": 4,
            "all_layer_complete": True,
        }
        finalized = [row for row in summary["snapshots"] if "kv_after_finalize" in row]
        assert finalized
        assert all(row["kv_after_finalize"]["all_layer_complete"] for row in finalized)
    else:
        assert summary["snapshots"] == []


def test_observation_restores_preexisting_instance_override_after_failure():
    engine = tiny_engine()

    def fail(_):
        raise RuntimeError("injected failure")

    engine.model_runner.execute = fail
    collector = RunCollector(["a"], arrival_ns=0)
    with pytest.raises(RuntimeError, match="injected"):
        with instrument_engine(engine, collector, clock=Clock()):
            engine.add_request("a", [1], SamplingParams(max_tokens=1))
            engine.step()
    assert engine.model_runner.execute is fail
    assert engine.cache_manager.num_used_blocks == 0
    assert not engine.has_unfinished_requests()


def test_actual_gate_finiteness_is_checked_only_after_execution():
    engine = tiny_engine()
    collector = RunCollector(["a"], arrival_ns=0)
    with instrument_engine(engine, collector, clock=Clock()) as obs:
        obs.gate_probabilities.extend([0.5, math.nan])
        with pytest.raises(ValueError, match="nonfinite"):
            obs.validate_gate_probabilities()
        summary = obs.summary()
        assert summary["gate_probabilities"] == [0.5, None]
        assert summary["nonfinite_gate_probabilities"] == {"1": "nan"}
        json.dumps(summary, allow_nan=False)


def test_populated_pages_count_once_across_layers_and_track_sparse_slots():
    cache = KVCacheManager(2, 1, 2, 16, 2, 4)
    cache.allocate("a", 4)
    assert populated_page_stats(cache) == {"populated_pages": 0, "all_layer_complete": True}
    values = torch.ones(1, 1, 2)

    def write(layer, position):
        cache.write(layer, ["a"], [0], [position], values, values)

    write(0, 0)
    assert populated_page_stats(cache) == {"populated_pages": 1, "all_layer_complete": False}
    write(1, 0)
    assert populated_page_stats(cache) == {"populated_pages": 1, "all_layer_complete": True}
    write(0, 3)
    write(1, 3)
    assert populated_page_stats(cache) == {"populated_pages": 2, "all_layer_complete": True}
    write(0, 2)  # Same physical page, different initialized slots across layers.
    assert populated_page_stats(cache) == {"populated_pages": 2, "all_layer_complete": False}
    write(1, 2)
    cache.finalize_token("a", 0, 0)
    assert populated_page_stats(cache) == {"populated_pages": 5, "all_layer_complete": True}
    cache.free("a")
    cache.allocate("b", 4)
    assert populated_page_stats(cache) == {"populated_pages": 0, "all_layer_complete": True}


def test_timing_instrumentation_does_not_scan_cache_metadata(monkeypatch):
    def forbidden(_):
        pytest.fail("timing path scanned populated cache metadata")

    monkeypatch.setattr("benchmarks.observe.populated_page_stats", forbidden)
    engine = tiny_engine()
    collector = RunCollector(["a"], arrival_ns=0)
    with instrument_engine(engine, collector, clock=Clock(), profile=False):
        engine.add_request("a", [1], SamplingParams(max_tokens=1))
        while engine.has_unfinished_requests():
            engine.step()
