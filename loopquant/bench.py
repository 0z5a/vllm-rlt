"""Closed-loop native-engine cohorts, including admission and final drain."""

import statistics
import time
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
    token_ids: list[int] = field(default_factory=list)
    exit_depths: list[int] = field(default_factory=list)
    finish_reason: str | None = None
    token_times_exact: bool = True


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
    if not prompts or concurrency < 1 or output_tokens < 1 or records:
        raise ValueError("expected prompts, positive workload sizes, and an empty records list")
    if engine.has_unfinished_requests() or engine.cache_manager.num_used_blocks:
        raise ValueError("engine must be fully drained with empty KV at cohort start")
    params = SamplingParams(
        max_tokens=output_tokens, temperature=0, min_loops=loops, max_loops=loops, ignore_eos=True
    )
    graph = engine.model_runner.graphs
    counters = (graph.captures, graph.replays, graph.fallbacks) if graph is not None else (0, 0, 0)
    active: dict[str, RequestTiming] = {}
    core_rows = []
    submitted = 0
    start = time.perf_counter()
    while submitted < len(prompts) or active:
        while submitted < len(prompts) and len(active) < concurrency:
            request_id = str(submitted)
            record = RequestTiming(request_id, time.perf_counter() - start)
            records.append(record)
            active[request_id] = record
            engine.add_request(request_id, prompts[submitted], params)
            submitted += 1
        outputs = engine.step()
        now = time.perf_counter() - start
        schedule = engine.last_schedule
        if schedule is not None and schedule.stage == Stage.RECURRENT:
            core_rows.append(schedule.num_tokens)
        for output in outputs:
            record = active[output.request_id]
            added = len(output.token_ids) - record.output_tokens
            if added:
                if record.first_token_s is None:
                    record.first_token_s = now
                record.token_times_exact &= added == 1
                record.delivered_token_times.append(now)
            record.output_tokens = len(output.token_ids)
            record.token_ids = output.token_ids
            record.exit_depths = output.exit_depths
            if output.finished:
                record.completed_s = now
                record.finish_reason = output.finish_reason
                del active[output.request_id]
    elapsed = time.perf_counter() - start
    successful = [
        record
        for record in records
        if record.output_tokens == output_tokens and record.finish_reason == "length"
    ]
    failures = len(records) - len(successful)
    tokens = sum(record.output_tokens for record in successful)
    histogram = Counter(depth for record in records for depth in record.exit_depths)
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
    )
