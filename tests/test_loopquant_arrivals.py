"""Arrival-boundary and mixed-work correctness for real tiny engine cohorts."""

from types import SimpleNamespace

from loopquant import bench
from tests.helpers import tiny_ouro_config
from vllm_rlt import LLM, CacheConfig, SchedulerConfig
from vllm_rlt.models.ouro import OuroForCausalLM


def make_engine():
    return LLM(
        OuroForCausalLM(tiny_ouro_config()),
        cache_config=CacheConfig(num_blocks=128, block_size=2),
        scheduler_config=SchedulerConfig(max_num_seqs=2, max_num_batched_tokens=8),
    ).engine


def test_mixed_trace_keeps_per_request_tokens_depths_and_actual_prefill_depth():
    engine = make_engine()
    requests = [
        bench.TraceRequest((5, 7), 2, 1),
        bench.TraceRequest((3, 9, 2), 4, 4),
        bench.TraceRequest((8,), 3, 2),
    ]
    records, queue = [], []
    result = bench.run_trace(engine, requests, concurrency=2, records=records, queue_samples=queue)
    assert result.completed_requests == 3 and result.output_tokens == 9
    assert result.loop_histogram == {1: 1, 4: 6, 2: 2}
    assert [row.exit_depths for row in records] == [[4, 1], [4, 4, 4, 4], [4, 2, 2]]
    assert result.prefill_loops == 4 and result.arrival_mode == "closed"
    assert result.kv_used_after_drain == result.failed_requests == 0
    assert result.schedule_shapes and result.itl_p95_ms is not None
    assert queue[-1].offered == queue[-1].completed == 3
    assert queue[-1].outstanding == 0
    assert records[2].entered_s >= min(row.completed_s for row in records[:2])


def test_open_trace_accounts_for_offered_arrivals_during_busy_engine_steps(monkeypatch):
    now = 0.0

    def clock():
        nonlocal now
        now += 0.01
        return now

    def sleep(seconds):
        nonlocal now
        now += seconds

    monkeypatch.setattr(bench, "time", SimpleNamespace(perf_counter=clock, sleep=sleep))
    records, queue = [], []
    arrivals = [0.0, 0.015, 0.016, 10.0]
    requests = [bench.TraceRequest((5, 7), 2, 4, at) for at in arrivals]
    result = bench.run_trace(
        make_engine(), requests, concurrency=None, records=records, queue_samples=queue
    )
    assert [row.entered_s for row in records] == arrivals
    assert records[1].admitted_s > records[1].entered_s
    assert queue[0].offered == 3 and queue[0].outstanding == 3
    assert result.elapsed_s > 10 and result.completed_requests == 4
    assert result.arrival_mode == "open" and queue[-1].outstanding == 0


def test_goodput_keeps_failures_in_denominator_and_counts_delivery_batches():
    records = [
        bench.RequestTiming(
            "good",
            0,
            first_token_s=1,
            completed_s=3,
            output_tokens=3,
            expected_tokens=3,
            finish_reason="length",
            delivered_token_times=[1, 3],
            delivered_token_counts=[1, 2],
            token_times_exact=False,
        ),
        bench.RequestTiming(
            "slow",
            0,
            first_token_s=4,
            completed_s=5,
            output_tokens=2,
            expected_tokens=2,
            finish_reason="length",
            delivered_token_times=[4, 5],
            delivered_token_counts=[1, 1],
        ),
        bench.RequestTiming("failed", 0, completed_s=10, finish_reason="abort", expected_tokens=3),
    ]
    result = bench.delivery_metrics(
        records, 10, ttft_budget_ms=2000, tpot_budget_ms=1100, steady_window=(2, 5)
    )
    assert result["request_goodput_s"] == 0.1
    assert result["slo_fraction"] == 1 / 3
    assert result["steady_output_tokens_s"] == 1
    assert result["steady_request_goodput_s"] == 1 / 3
