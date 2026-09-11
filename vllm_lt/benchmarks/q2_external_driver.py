"""One-request adapters and shared host timing for the finite Q2 comparison.

The native engine and official model share immutable weights. The native pool
stays resident for both implementations. Detailed feasibility checks and cache
reset are outside measured delivery intervals; no profiler is installed.
"""

import time
from collections import Counter
from contextlib import nullcontext

import torch

from vllm_lt.benchmarks.observe import RunCollector
from vllm_lt.benchmarks.profile import finite_checks
from vllm_lt.benchmarks.runner import memory
from vllm_lt.request import RequestOutput
from vllm_lt.sampling_params import SamplingParams


def check_deadline(deadline_ns):
    if time.perf_counter_ns() >= deadline_ns:
        raise TimeoutError("Q2 full-case or global deadline exceeded")


def request_state(engine, official):
    return {
        "native_requests": len(engine.scheduler.requests),
        "native_used_blocks": engine.cache_manager.num_used_blocks,
        "official_cache_slots": official.cache_slots,
    }


def require_empty(engine, official):
    state = request_state(engine, official)
    if any(state.values()) or engine.has_unfinished_requests():
        raise RuntimeError(f"Q2 request state was not released: {state}")
    if any(engine.scheduler.queues.values()):
        raise RuntimeError("Q2 native scheduler retains queued request IDs")
    return state


def shared_weight_proof(model, official):
    """Record storage identity without reading or copying device parameter values."""
    native = dict(model.named_parameters())
    external = dict(official.model.named_parameters())
    if native.keys() != external.keys():
        raise ValueError("native and official parameter names differ")
    records = []
    for name, left in native.items():
        right = external[name]
        same = (
            left.data_ptr() == right.data_ptr()
            and left.shape == right.shape
            and left.dtype == right.dtype == torch.float32
            and left.device == right.device
        )
        if not same:
            raise ValueError(f"official parameter storage is not shared: {name}")
        records.append(
            {
                "name": name,
                "shape": list(left.shape),
                "numel": left.numel(),
                "dtype": str(left.dtype),
                "device": str(left.device),
                "native_data_ptr": left.data_ptr(),
                "official_data_ptr": right.data_ptr(),
                "native_requires_grad": left.requires_grad,
                "official_requires_grad": right.requires_grad,
            }
        )
    return {
        "all_storage_equal": True,
        "parameter_count": sum(r["numel"] for r in records),
        "records": records,
    }


def pool_descriptor(engine):
    cache = engine.cache_manager
    return {
        "num_blocks": cache.num_blocks,
        "block_size": cache.block_size,
        "shape": list(cache.key_cache.shape),
        "bytes": cache.num_blocks * cache.bytes_per_block,
        "key_data_ptr": cache.key_cache.data_ptr(),
        "value_data_ptr": cache.value_cache.data_ptr(),
    }


def _native_outputs(engine, prompt, params, collector, deadline_ns, max_steps):
    engine.add_request("W1", prompt, params)
    counts = Counter()
    dispatches = []
    steps = 0
    while engine.has_unfinished_requests():
        check_deadline(deadline_ns)
        if steps >= max_steps:
            raise RuntimeError("Q2 native step bound exceeded")
        outputs = engine.step()
        returned = time.perf_counter_ns()
        collector.observe_outputs(outputs, step_id=steps, returned_ns=returned)
        batch = engine.last_schedule
        if batch is None:
            raise RuntimeError("Q2 active native request produced no dispatch")
        counts[batch.stage.value] += 1
        dispatches.append(
            {"step_id": steps, "stage": batch.stage.value, "num_tokens": batch.num_tokens}
        )
        steps += 1
    return {"steps": steps, "stage_counts": dict(counts), "dispatches": dispatches}


def _official_outputs(official, prompt, output_count, collector, deadline_ns, feasibility):
    ids = []
    calls = []
    checked = 0
    for index in range(output_count):
        check_deadline(deadline_ns)
        if index == 0:
            logits = official.start(prompt, max_outputs=output_count)
        else:
            logits = official.advance(ids[-1])
        if feasibility:
            if not bool(torch.isfinite(logits).all().item()):
                raise ValueError("nonfinite official logits during Q2 feasibility")
            checked += 1
        token = int(logits.argmax(dim=-1).item())
        ids.append(token)
        output = RequestOutput(
            "W1",
            prompt,
            list(ids),
            [4] * len(ids),
            len(ids) == output_count,
            "length" if len(ids) == output_count else None,
        )
        # Match the native API boundary: copied output history is ready before
        # the timestamp; collector accounting and diagnostics follow it.
        returned = time.perf_counter_ns()
        collector.observe_outputs([output], step_id=index, returned_ns=returned)
        calls.append(
            {
                "output_index": index,
                "input_count": len(prompt) if index == 0 else 1,
                "last_input_position": len(prompt) - 1 + index,
                "token_id": token,
            }
        )
        del logits
    return {
        "steps": output_count,
        "stage_counts": {"prefill": 1, "cached_decode": output_count - 1},
        "official_logits_checked": checked,
    }, calls


def execute_case(plan, run, engine, official, *, started_ns, deadline_ns):
    """Execute exactly one frozen row; outer owner persists its lifetime/ACK."""
    check_deadline(deadline_ns)
    before_request = require_empty(engine, official)
    setup_start = time.perf_counter_ns()
    official.reset()
    engine.last_schedule = None
    before_pool = pool_descriptor(engine)
    prompt = list(plan["workload"]["requests"][0]["prompt_token_ids"])
    output_count = plan["workload"]["requests"][0]["max_output_tokens"]
    params = SamplingParams(
        **{**plan["contract"]["engine"]["sampling"], "max_tokens": output_count}
    )
    setup_ns = time.perf_counter_ns() - setup_start
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    before = memory()
    arrival = time.perf_counter_ns()
    collector = RunCollector(["W1"], arrival_ns=arrival, max_events=128)
    feasibility = run["phase"] == "feasibility"
    official_cache = None
    finite = {}
    failures = []
    counts = {}
    synchronized = None
    result = None
    try:
        with (
            finite_checks(engine.model, feasibility)
            if run["implementation_id"] == "native"
            else nullcontext({}) as finite
        ):
            arrival = time.perf_counter_ns()
            collector.start(arrival)
            if run["implementation_id"] == "native":
                counts = _native_outputs(
                    engine, prompt, params, collector, deadline_ns, run["max_steps"]
                )
            else:
                counts, calls = _official_outputs(
                    official, prompt, output_count, collector, deadline_ns, feasibility
                )
                official_cache = {"calls": calls}
            torch.cuda.synchronize()
            synchronized = time.perf_counter_ns()
        collected = collector.finish(synchronized_ns=synchronized)
        rows = collected["requests"]
        if (
            len(rows) != 1
            or len(rows[0]["token_ids"]) != output_count
            or rows[0]["exit_depths"] != [4] * output_count
            or not rows[0]["finished"]
        ):
            raise ValueError("Q2 request did not complete the frozen four-loop history")
        if official_cache is not None:
            official_cache["snapshot"] = official.snapshot(
                inspect_cache=True, check_finite=feasibility
            )
            official_cache["final_summary"] = official_cache["snapshot"]["final_summary"]
            summary = official_cache["final_summary"]
            if (
                summary["slot_count"] != 96
                or summary["lengths"] != [191] * 96
                or not summary["distinct_storage"]
                or (feasibility and not summary["all_finite"])
            ):
                raise RuntimeError(
                    "official cache did not retain 96 distinct complete depth/layer histories"
                )
            official.complete(completion_confirmed=True)
        peaks = {
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        }
        official.reset()
        engine.last_schedule = None
        torch.cuda.synchronize()
        cleanup = require_empty(engine, official)
        if pool_descriptor(engine) != before_pool:
            raise RuntimeError("common native pool storage changed during Q2 execution")
        check_deadline(deadline_ns)
        result = {
            "schema_version": 1,
            "artifact_type": "q2_external_result",
            "run": run,
            "plan_sha256": plan["plan_sha256"],
            "status": "complete",
            "started_ns": started_ns,
            "deadline_ns": deadline_ns,
            "arrival_ns": arrival,
            "synchronized_ns": synchronized,
            "requests": rows,
            "events": collected["events"],
            "metrics": collected["metrics"],
            "setup_ns": setup_ns,
            "before_request": before_request,
            "cleanup": cleanup,
            "memory": {
                "before": before,
                **peaks,
                "after_requests": memory(),
                "pool_bytes": before_pool["bytes"],
            },
            "counts": counts,
            "official_cache": official_cache,
            "feasibility": {"finite_checks": dict(finite), "passed": True} if feasibility else None,
            "failures": [],
            "ended_ns": time.perf_counter_ns(),
        }
        return result
    except BaseException as error:
        failures.append({"type": type(error).__name__, "message": str(error)[:2000]})
        partial = collector.snapshot(synchronized_ns=synchronized)
        # Caller retains this evidence even when request cleanup itself fails.
        error.q2_partial_result = {
            "schema_version": 1,
            "artifact_type": "q2_external_result",
            "run": run,
            "plan_sha256": plan["plan_sha256"],
            "status": "failed",
            "started_ns": started_ns,
            "deadline_ns": deadline_ns,
            "arrival_ns": arrival,
            "synchronized_ns": synchronized,
            "requests": partial["requests"],
            "events": partial["events"],
            "metrics": partial["metrics"],
            "counts": counts,
            "official_cache": official_cache,
            "failures": failures,
            "ended_ns": time.perf_counter_ns(),
        }
        raise
