"""Audit every request in a native W5 job; whole-window proof stays separate."""

import argparse
import hashlib
import json
import math
import statistics
from collections.abc import Mapping
from pathlib import Path
from typing import TypedDict, cast

from run_w5_cost_arm import Arm, Case, PromptFile, Protocol, Scheduler

Distribution = Mapping[str, int | float]
Config = Mapping[str, bool | int | float | str | None]
Sampling = Mapping[str, bool | int | float | str | None | Config]


class Request(TypedDict):
    ordinal: int
    request_id: str
    prompt: list[int]
    token_ids: list[int]
    exit_depths: list[int]
    submitted_s: float
    admitted_s: float
    admission_boundary: str
    delivery_s: list[float]
    finished_s: float
    finish_reason: str


class Trial(TypedDict):
    index: int
    phase: str
    elapsed_s: float
    tps: float
    rps: float
    requests: list[Request]
    ttft_s: Distribution
    tpot_s: Distribution
    itl_s: Distribution
    latency_s: Distribution
    admission_wait_s: Distribution
    admission_timestamp_scope: str
    resident_observation_scope: str
    kv_peak_observation_scope: str
    resident_histogram: dict[str, int]
    resident_poststep_histogram: dict[str, int]
    outstanding_histogram: dict[str, int]
    decode_effective_rows_histogram: dict[str, int]
    neutral_core_head_work: dict[str, int]
    loopcd_work: dict[str, int]
    graph: dict[str, int]
    kv_peak_blocks: int
    kv_pool_bytes: int
    reference_peak_bytes: int
    peak_allocated_bytes: int
    peak_reserved_bytes: int
    drained: bool
    repeat_parity: bool


class Job(TypedDict):
    status: str
    job_id: str
    protocol_sha256: str
    case: Case
    arm: Arm
    kv_pool_bytes: int
    actual_storage_planes: int
    observer_sha256: str
    execution: Config
    sampling: Sampling
    cache_config: Config
    scheduler_config: Scheduler


def audit_configuration(
    execution: Config,
    sampling: Sampling,
    cache: Config,
    arm: Arm,
    pool_bytes: int,
) -> None:
    assert [execution[k] for k in ("loopcd", "prefill_depth", "multi_stream")] == [
        arm["guidance"],
        4,
        True,
    ]
    assert all(
        execution[k] is False
        for k in (
            "cuda_graphs",
            "async_scheduling",
            "static_buffers",
            "pad_to_power_of_two",
            "prefill_uva",
        )
    )
    assert [
        sampling[k]
        for k in (
            "max_tokens",
            "temperature",
            "min_loops",
            "max_loops",
            "ignore_eos",
            "top_p",
            "top_k",
            "seed",
            "exit_threshold",
            "priority",
        )
    ] == [128, 0.0, arm["depth"], arm["depth"], True, 1.0, -1, 0, 1.0, 0]
    assert sampling["loopcd"] == (
        {
            "mode": "logits",
            "reference_loop": 1,
            "strength": arm["strength"],
            "implementation": "two_head",
            "strength_mode": "fixed",
            "prefill_reference_loop": None,
        }
        if arm["guidance"]
        else None
    )
    assert cache == {
        "num_blocks": None,
        "block_size": 16,
        "layout": "last_exited",
        "gpu_memory_utilization": 0.9,
        "kv_cache_memory_bytes": pool_bytes,
        "memory_reserve_bytes": 256 * 1024**2,
        "enable_prefix_caching": False,
        "incremental_allocation": False,
        "watermark_ratio": 0.0,
    }


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_distribution(record: Distribution, values: list[float]) -> None:
    assert values and all(math.isfinite(v) and v >= 0 for v in values)
    ordered = sorted(values)
    expected = {
        "n": len(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "p95": ordered[math.ceil(0.95 * len(values)) - 1],
        "p99": ordered[math.ceil(0.99 * len(values)) - 1],
        "min": ordered[0],
        "max": ordered[-1],
    }
    assert record.keys() == expected.keys()
    for key, value in expected.items():
        assert math.isclose(record[key], value, rel_tol=1e-12, abs_tol=1e-12), key


def histogram_peak(hist: dict[str, int], maximum: int) -> int:
    assert hist and all(int(k) >= 0 and v > 0 for k, v in hist.items())
    peak = max(map(int, hist))
    assert peak <= maximum
    return peak


def audit_trial(
    trial: Trial, prompts: list[list[int]], *, B: int, C: int, depth: int, guided: bool
) -> str:
    rows = trial["requests"]
    assert len(rows) == len(prompts) and len(rows) >= C
    assert trial["drained"] and trial["repeat_parity"]
    assert trial["admission_timestamp_scope"] == "SCHEDULE_RETURN_BEFORE_MODEL_EXECUTE"
    assert trial["resident_observation_scope"] == "POST_SCHEDULE_PRE_EXECUTE"
    assert trial["kv_peak_observation_scope"] == "SCHEDULE_CORE_CODA_POSTSTEP_METADATA"
    previous_finish: dict[str, float] = {}
    for index, (row, prompt) in enumerate(zip(rows, prompts, strict=True)):
        assert row["ordinal"] == index and row["prompt"] == prompt
        assert 0 <= int(row["request_id"].removeprefix("slot-")) < C
        assert row["admission_boundary"] == "SCHEDULE_RETURN_BEFORE_MODEL_EXECUTE"
        assert len(row["token_ids"]) == len(row["delivery_s"]) == 128
        assert row["exit_depths"] == [4] + [depth] * 127
        assert row["finish_reason"] == "length"
        assert 0 <= row["submitted_s"] <= row["admitted_s"] <= row["delivery_s"][0]
        assert all(math.isfinite(t) for t in row["delivery_s"])
        assert row["delivery_s"] == sorted(row["delivery_s"])
        assert row["delivery_s"][-1] == row["finished_s"] <= trial["elapsed_s"]
        if row["request_id"] in previous_finish:
            assert row["submitted_s"] >= previous_finish[row["request_id"]]
        previous_finish[row["request_id"]] = row["finished_s"]
    events = sorted(
        [(r["submitted_s"], 1) for r in rows] + [(r["finished_s"], -1) for r in rows]
    )
    outstanding = 0
    for _, change in events:
        outstanding += change
        assert 0 <= outstanding <= C
    assert outstanding == 0
    check_distribution(
        trial["ttft_s"], [r["delivery_s"][0] - r["submitted_s"] for r in rows]
    )
    check_distribution(
        trial["tpot_s"],
        [(r["delivery_s"][-1] - r["delivery_s"][0]) / 127 for r in rows],
    )
    check_distribution(
        trial["latency_s"], [r["finished_s"] - r["submitted_s"] for r in rows]
    )
    check_distribution(
        trial["admission_wait_s"], [r["admitted_s"] - r["submitted_s"] for r in rows]
    )
    check_distribution(
        trial["itl_s"],
        [b - a for r in rows for a, b in zip(r["delivery_s"], r["delivery_s"][1:])],
    )
    assert math.isfinite(trial["elapsed_s"]) and trial["elapsed_s"] > 0
    assert math.isclose(
        trial["tps"] * trial["elapsed_s"], len(rows) * 128, rel_tol=1e-12, abs_tol=1e-6
    )
    assert math.isclose(
        trial["rps"] * trial["elapsed_s"], len(rows), rel_tol=1e-12, abs_tol=1e-6
    )
    neutral, work = trial["neutral_core_head_work"], trial["loopcd_work"]
    assert (
        neutral["prefill_core_rows"]
        == neutral["prefill_submitted_rows"]
        == 4 * sum(map(len, prompts))
    )
    assert (
        neutral["decode_core_rows"]
        == neutral["decode_submitted_rows"]
        == len(rows) * 127 * depth
    )
    assert (
        neutral["head_rows"] == len(rows) * 128 * (2 if guided else 1)
        and neutral["head_calls"] > 0
    )
    assert work["guided_rows"] == (len(rows) * 128 if guided else 0)
    for key in (
        "prefill_core_rows",
        "prefill_submitted_rows",
        "decode_core_rows",
        "decode_submitted_rows",
        "head_rows",
        "head_calls",
    ):
        assert work[key] == (neutral[key] if guided else 0)
    assert (trial["reference_peak_bytes"] > 0) is guided
    assert work["captures"] > 0 if guided else all(v == 0 for v in work.values())
    assert trial["graph"] == {"captures": 0, "replays": 0, "fallbacks": 0}
    assert histogram_peak(trial["outstanding_histogram"], C) == C
    histogram_peak(trial["resident_histogram"], B)
    histogram_peak(trial["resident_poststep_histogram"], B)
    histogram_peak(trial["decode_effective_rows_histogram"], B)
    assert (
        0 < trial["kv_peak_blocks"]
        and 0
        < trial["kv_pool_bytes"]
        <= trial["peak_allocated_bytes"]
        <= trial["peak_reserved_bytes"]
    )
    signature = [
        (
            r["ordinal"],
            r["request_id"],
            r["token_ids"],
            r["exit_depths"],
            r["finish_reason"],
        )
        for r in rows
    ]
    return hashlib.sha256(json.dumps(signature).encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("protocol", "prompts", "job", "trials", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    protocol = cast(Protocol, json.loads(args.protocol.read_text()))
    job = cast(Job, json.loads(args.job.read_text()))
    assert protocol["status"] == "CANONICAL_PROMPTS_FROZEN_NOT_GRANTED_NOT_ADMITTED"
    assert job["status"] == "COMPLETE" and job["protocol_sha256"] == digest(
        args.protocol
    )
    assert (
        job["observer_sha256"] == protocol["trial_observer_sha256"]
        and job["actual_storage_planes"] == 4
    )
    assert job["case"] in protocol["cases"] and job["arm"] in protocol["arms"]
    assert job["job_id"] in {
        f"{job['case']['case_id']}-fresh{fresh}-{job['arm']['arm_id']}"
        for fresh in (0, 1)
    }
    assert 0 < job["kv_pool_bytes"] <= protocol["common_pool_bytes"]
    assert job["scheduler_config"] == job["case"]["scheduler_config"]
    audit_configuration(
        job["execution"],
        job["sampling"],
        job["cache_config"],
        job["arm"],
        protocol["common_pool_bytes"],
    )
    assert digest(args.prompts) == protocol["resolved_prompt_manifest_sha256"]
    all_rows = cast(PromptFile, json.loads(args.prompts.read_text()))["records"]
    assert len(all_rows) == len({r["prompt_id"] for r in all_rows}) == 820
    rows = [r for r in all_rows if r["case_id"] == job["case"]["case_id"]]
    assert [r["prompt_id"] for r in rows] == job["case"]["prompt_ids"]
    assert len(rows) == 4 * job["case"]["outstanding_C"]
    paths = sorted(args.trials.glob("trial-*.json"))
    assert len(paths) == 10
    summary = cast(
        list[dict[str, object]], json.loads((args.trials / "summary.json").read_text())
    )
    assert len(summary) == 10
    signatures, measured = [], []
    for index, path in enumerate(paths):
        trial = cast(Trial, json.loads(path.read_text()))
        assert trial["index"] == index and trial["phase"] == (
            "warmup" if index < 5 else "measured"
        )
        assert trial["kv_pool_bytes"] == job["kv_pool_bytes"]
        assert summary[index] == {k: v for k, v in trial.items() if k != "requests"}
        signatures.append(
            audit_trial(
                trial,
                [r["prompt_token_ids"] for r in rows],
                B=job["case"]["batch_limit_B"],
                C=job["case"]["outstanding_C"],
                depth=job["arm"]["depth"],
                guided=job["arm"]["guidance"],
            )
        )
        if index >= 5:
            measured.append(trial)
    assert len(set(signatures)) == 1
    result = {
        "status": "PASS_COMPLETE_SINGLE_W5_JOB_RAW_ONLY",
        "job_id": job["job_id"],
        "protocol_sha256": digest(args.protocol),
        "job_sha256": digest(args.job),
        "auditor_sha256": digest(Path(__file__)),
        "raw_files": [{"path": str(p), "sha256": digest(p)} for p in paths],
        "raw_trials": 10,
        "requests_per_trial": len(rows),
        "same_policy_signature_sha256": signatures[0],
        "measured_elapsed_s": [t["elapsed_s"] for t in measured],
        "measured_tps": [t["tps"] for t in measured],
        "actual_A_peak": max(
            histogram_peak(t["resident_histogram"], job["case"]["batch_limit_B"])
            for t in measured
        ),
        "actual_S_peak": max(
            histogram_peak(
                t["decode_effective_rows_histogram"], job["case"]["batch_limit_B"]
            )
            for t in measured
        ),
        "target_A_reached": any(
            str(job["case"]["batch_limit_B"]) in t["resident_histogram"]
            for t in measured
        ),
        "whole_window_natural_exit_absence_and_offbox": "REQUIRES_SEPARATE_ACTUAL_CONTROLLER_PROOF",
        "two_fresh_3arm_and_full7matrix": "NOT_PROVEN_BY_SINGLE_JOB",
    }
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(
        json.dumps({"status": result["status"], "result_sha256": digest(args.output)})
    )


if __name__ == "__main__":
    main()
