"""Closed-loop native-engine cohorts, including admission and final drain."""

import math
import statistics
import time
from bisect import bisect_right
from collections import Counter
from dataclasses import dataclass, field

from vllm_rlt.engine.llm_engine import LLMEngine
from vllm_rlt.request import Stage
from vllm_rlt.sampling_params import SamplingParams


@dataclass
class RequestTiming:
    request_id: str
    entered_s: float
    first_token_s: float | None = None
    completed_s: float | None = None
    output_tokens: int = 0
    delivered_token_times: list[float] = field(default_factory=list)
    delivered_token_counts: list[int] = field(default_factory=list)
    token_ids: list[int] = field(default_factory=list)
    exit_depths: list[int] = field(default_factory=list)
    finish_reason: str | None = None
    token_times_exact: bool = True
    admitted_s: float = 0.0
    expected_tokens: int = 0
    expected_loops: int = 0


@dataclass(frozen=True)
class TraceRequest:
    """loops applies to decode; the first output uses the model's prefill depth."""

    token_ids: tuple[int, ...]
    output_tokens: int
    loops: int
    arrival_s: float = 0.0


@dataclass(frozen=True)
class QueueSample:
    elapsed_s: float
    offered: int
    completed: int
    outstanding: int
    engine_waiting: int


@dataclass(frozen=True)
class CohortResult:
    elapsed_s: float
    output_tokens_s: float
    requests_s: float
    completed_requests: int
    failed_requests: int
    output_tokens: int
    mean_core_rows: float | None
    loop_histogram: dict[int, int]
    graph_captures: int
    graph_replays: int
    graph_fallbacks: int
    kv_used_after_drain: int
    ttft_p95_ms: float
    tpot_p95_ms: float | None
    ttft_p50_ms: float
    tpot_p50_ms: float | None
    itl_p50_ms: float | None
    itl_p95_ms: float | None
    schedule_shapes: dict[str, int]
    arrival_mode: str
    prefill_loops: int


def percentile(values: list[float], fraction: float) -> float:
    values = sorted(values)
    position = (len(values) - 1) * fraction
    low = int(position)
    high = min(low + 1, len(values) - 1)
    return values[low] + (values[high] - values[low]) * (position - low)


def run_cohort(
    engine: LLMEngine,
    prompts: list[list[int]],
    *,
    concurrency: int,
    output_tokens: int,
    loops: int,
    records: list[RequestTiming],
) -> CohortResult:
    """The caller owns records so a failed run still retains delivered requests.

    Warmup must run separately with the same shapes. A capture inside this cohort
    is reported and prevents the report layer from accepting a formal trial.
    """
    if engine.model_runner.model.config.total_ut_steps != loops:
        raise ValueError("fixed-depth cohorts require matching prefill and decode loop counts")
    return run_trace(
        engine,
        [TraceRequest(tuple(prompt), output_tokens, loops) for prompt in prompts],
        concurrency=concurrency,
        records=records,
        queue_samples=[],
    )


def run_trace(
    engine: LLMEngine,
    requests: list[TraceRequest],
    *,
    concurrency: int | None,
    records: list[RequestTiming],
    queue_samples: list[QueueSample],
) -> CohortResult:
    """None concurrency means open arrivals; otherwise refill a closed cohort.

    Open-arrival latency starts at the immutable offered timestamp, including
    submission delay while engine.step is busy. HTTP latency is a separate L2
    measurement. Mixed decode depths retain the model's configured prefill R.
    """
    if not requests or records or queue_samples or (concurrency is not None and concurrency < 1):
        raise ValueError("expected requests, empty output lists and positive concurrency")
    prefill_loops = engine.model_runner.model.config.total_ut_steps
    if any(
        not row.token_ids
        or row.output_tokens < 1
        or not 1 <= row.loops <= prefill_loops
        or not math.isfinite(row.arrival_s)
        or row.arrival_s < 0
        for row in requests
    ) or any(a.arrival_s > b.arrival_s for a, b in zip(requests, requests[1:])):
        raise ValueError("invalid prompt, token budget, loop depth or ordered arrival time")
    if engine.has_unfinished_requests() or engine.cache_manager.num_used_blocks:
        raise ValueError("engine must be fully drained with empty KV at cohort start")
    graph = engine.model_runner.graphs
    counters = (graph.captures, graph.replays, graph.fallbacks) if graph is not None else (0, 0, 0)
    active: dict[str, RequestTiming] = {}
    core_rows = []
    shapes: Counter[str] = Counter()
    submitted = 0
    next_sample = 0.0
    arrivals = [row.arrival_s for row in requests]
    start = time.perf_counter()
    while submitted < len(requests) or active:
        now = time.perf_counter() - start
        while (
            submitted < len(requests)
            and (concurrency is None or len(active) < concurrency)
            and requests[submitted].arrival_s <= now
        ):
            spec = requests[submitted]
            request_id = str(submitted)
            record = RequestTiming(
                request_id,
                spec.arrival_s if concurrency is None else now,
                admitted_s=now,
                expected_tokens=spec.output_tokens,
                expected_loops=spec.loops,
            )
            records.append(record)
            active[request_id] = record
            engine.add_request(
                request_id,
                list(spec.token_ids),
                SamplingParams(
                    max_tokens=spec.output_tokens,
                    temperature=0,
                    min_loops=spec.loops,
                    max_loops=spec.loops,
                    ignore_eos=True,
                ),
            )
            submitted += 1
        if not active:
            time.sleep(min(0.05, max(0.0, requests[submitted].arrival_s - now)))
            continue
        outputs = engine.step()
        now = time.perf_counter() - start
        schedule = engine.last_schedule
        if schedule is not None and schedule.stage == Stage.RECURRENT:
            core_rows.append(schedule.num_tokens)
        if schedule is not None:
            runner = engine.model_runner
            shapes[
                f"{schedule.stage.name}:{runner.last_effective_size}:{runner.last_submitted_size}"
            ] += 1
        for output in outputs:
            record = active[output.request_id]
            added = len(output.token_ids) - record.output_tokens
            if added:
                if record.first_token_s is None:
                    record.first_token_s = now
                record.token_times_exact &= added == 1
                record.delivered_token_times.append(now)
                record.delivered_token_counts.append(added)
            record.output_tokens = len(output.token_ids)
            record.token_ids = output.token_ids
            record.exit_depths = output.exit_depths
            if output.finished:
                record.completed_s = now
                record.finish_reason = output.finish_reason
                del active[output.request_id]
        if now >= next_sample or (submitted == len(requests) and not active):
            completed = submitted - len(active)
            offered = bisect_right(arrivals, now) if concurrency is None else submitted
            queue_samples.append(
                QueueSample(
                    now,
                    offered,
                    completed,
                    offered - completed,
                    len(engine.scheduler.queues[Stage.WAITING]),
                )
            )
            next_sample = now + 0.05
    engine.model_runner.synchronize()
    elapsed = time.perf_counter() - start
    successful = [
        record
        for record in records
        if record.output_tokens == record.expected_tokens and record.finish_reason == "length"
    ]
    failures = len(records) - len(successful)
    tokens = sum(record.output_tokens for record in successful)
    histogram = Counter(depth for record in records for depth in record.exit_depths)
    if any(
        len(row.exit_depths) != row.output_tokens
        or any(
            depth != (prefill_loops if index == 0 else row.expected_loops)
            for index, depth in enumerate(row.exit_depths)
        )
        for row in records
    ):
        raise RuntimeError("observed exit depths differ from the registered request trace")
    ttfts = [
        (record.first_token_s - record.entered_s) * 1000
        for record in records
        if record.first_token_s is not None
    ]
    tpots = [
        (record.delivered_token_times[-1] - record.first_token_s)
        * 1000
        / (record.output_tokens - 1)
        for record in records
        if record.first_token_s is not None and record.output_tokens > 1
    ]
    itls = [
        (right - left) * 1000
        for row in records
        if row.token_times_exact
        for left, right in zip(row.delivered_token_times, row.delivered_token_times[1:])
    ]
    after = (graph.captures, graph.replays, graph.fallbacks) if graph is not None else (0, 0, 0)
    return CohortResult(
        elapsed,
        tokens / elapsed,
        len(successful) / elapsed,
        len(successful),
        failures,
        tokens,
        statistics.mean(core_rows) if core_rows else None,
        dict(histogram),
        *(end - begin for begin, end in zip(counters, after)),
        engine.cache_manager.num_used_blocks,
        percentile(ttfts, 0.95) if ttfts else float("nan"),
        percentile(tpots, 0.95) if tpots else None,
        percentile(ttfts, 0.5) if ttfts else float("nan"),
        percentile(tpots, 0.5) if tpots else None,
        percentile(itls, 0.5) if itls else None,
        percentile(itls, 0.95) if itls else None,
        dict(shapes),
        "open" if concurrency is None else "closed",
        prefill_loops,
    )


def delivery_metrics(
    records: list[RequestTiming],
    elapsed_s: float,
    *,
    ttft_budget_ms: float,
    tpot_budget_ms: float,
    steady_window: tuple[float, float] | None = None,
) -> dict[str, float | int | None]:
    """Goodput includes every offered request and the complete cohort drain."""
    if not records or any(
        not math.isfinite(value) or value <= 0
        for value in (elapsed_s, ttft_budget_ms, tpot_budget_ms)
    ):
        raise ValueError("records, elapsed time and preregistered SLO budgets are required")
    good = []
    for row in records:
        if row.first_token_s is None or row.completed_s is None:
            continue
        ttft = (row.first_token_s - row.entered_s) * 1000
        tpot = (
            (row.delivered_token_times[-1] - row.first_token_s) * 1000 / (row.output_tokens - 1)
            if row.output_tokens > 1
            else 0.0
        )
        if (
            row.finish_reason == "length"
            and row.output_tokens == row.expected_tokens
            and ttft <= ttft_budget_ms
            and tpot <= tpot_budget_ms
        ):
            good.append(row)
    result = dict(
        offered_requests=len(records),
        slo_requests=len(good),
        request_goodput_s=len(good) / elapsed_s,
        slo_fraction=len(good) / len(records),
        steady_output_tokens_s=None,
        steady_request_goodput_s=None,
    )
    if steady_window is not None:
        begin, end = steady_window
        if not 0 <= begin < end <= elapsed_s:
            raise ValueError("steady window must be fixed in advance and fully observed")
        delivered = sum(
            count
            for row in records
            for at, count in zip(row.delivered_token_times, row.delivered_token_counts, strict=True)
            if begin <= at < end
        )
        result["steady_output_tokens_s"] = delivered / (end - begin)
        result["steady_request_goodput_s"] = sum(
            row.completed_s is not None and begin <= row.completed_s < end for row in good
        ) / (end - begin)
    return result
