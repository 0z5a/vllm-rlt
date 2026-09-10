"""Hand-recorded reports verify accounting independently of device execution."""

import hashlib
import json

import pytest

from vllm_lt.benchmarks.report import build_report


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


def bind_plan(folder, plan, *, nested=True):
    body = {key: value for key, value in plan.items() if key != "plan_sha256"}
    plan["plan_sha256"] = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    write_json(folder / ("inputs/plan.json" if nested else "plan.json"), plan)
    write_json(
        folder / "manifest.json",
        {
            "schema_version": 1,
            "artifact_type": "experiment_manifest",
            "plan_sha256": plan["plan_sha256"],
            "device_ids": [5],
            "status": "complete",
            "failures": [],
        },
    )
    for result_path in (folder / "runs").glob("*/result.json"):
        result = json.loads(result_path.read_text())
        result["plan_sha256"] = plan["plan_sha256"]
        write_json(result_path, result)


def example_experiment(folder, *, nested=True):
    workload = {
        "workload_id": "W4",
        "kind": "scheduler_replay",
        "requests": [
            {"request_id": "A", "max_output_tokens": 3, "prompt_token_ids": [1, 2]},
            {"request_id": "B", "max_output_tokens": 2, "prompt_token_ids": [3]},
        ],
        "replay": {
            "A": {"output_token_ids": [10, 11, 12], "exit_depths": [4, 2, 4]},
            "B": {"output_token_ids": [20, 21], "exit_depths": [4, 3]},
        },
    }
    order = []
    for mode, repetition in (("refill", 1), ("no_refill", 1), ("no_refill", 2), ("refill", 2)):
        order.append(
            {
                "run_id": f"W4-{mode}-{repetition}",
                "cell_id": f"W4-{mode}",
                "workload_id": "W4",
                "mode": mode,
                "phase": "measured",
                "repetition": repetition,
                "pair_id": f"W4-pair-{repetition}",
                "workload_sha256": "a" * 64,
                "controls_sha256": "b" * 64,
                "instrumentation": "timing",
            }
        )
    plan = {
        "schema_version": 1,
        "artifact_type": "execution_plan",
        "suite": {"workloads": [workload]},
        "contract": {},
        "model_path": "/deliberately/unavailable/checkpoint",
        "model_config": {},
        "source": {"commit": "frozen-source", "files": []},
        "model_files": [{"path": "/no/checkpoint/file", "sha256": "c" * 64}],
        "inputs": {},
        "workload_stats": {},
        "execution_order": order,
    }
    bind_plan(folder, plan, nested=nested)
    for index, entry in enumerate(order):
        write_run(folder, entry, index=index, scale=1 if entry["mode"] == "refill" else 2)
    return plan


def write_run(folder, entry, *, index=0, scale=1):
    # Arrival zero relative to each run. A: 10/20/40 ms, B: 20/40 ms.
    # Five tokens / 40 ms = 125 tokens/s; A TPOT 15 ms, B TPOT 20 ms.
    arrival = (index + 1) * 1_000_000_000
    ms = scale * 1_000_000
    requests = []
    for request_id, tokens, depths, times in (
        ("A", [10, 11, 12], [4, 2, 4], [10, 20, 40]),
        ("B", [20, 21], [4, 3], [20, 40]),
    ):
        requests.append(
            {
                "request_id": request_id,
                "token_ids": tokens,
                "exit_depths": depths,
                "token_timestamps_ns": [arrival + time * ms for time in times],
                "admitted_ns": arrival,
                "enqueue_start_ns": arrival,
                "enqueue_end_ns": arrival,
                "first_prefill_ns": arrival,
                "finished": True,
                "finish_reason": "length",
            }
        )
    metrics = {
        "delivery_wall_ns": 40 * ms,
        "synchronized_wall_ns": 45 * ms,
        "generated_tokens": 5,
        "generated_tokens_per_second": 125 / scale,
        "mean_decode_depth": 3.0,
        "token_weighted_tpot_ns": (50 / 3) * ms,
        "per_request": {
            "A": {
                "ttft_ns": 10 * ms,
                "completion_latency_ns": 40 * ms,
                "tpot_ns": float(15 * ms),
                "token_gaps_ns": [10 * ms, 20 * ms],
                "admission_queue_ns": 0,
                "enqueue_ns": 0,
                "admitted_to_first_prefill_ns": 0,
            },
            "B": {
                "ttft_ns": 20 * ms,
                "completion_latency_ns": 40 * ms,
                "tpot_ns": float(20 * ms),
                "token_gaps_ns": [20 * ms],
                "admission_queue_ns": 0,
                "enqueue_ns": 0,
                "admitted_to_first_prefill_ns": 0,
            },
        },
        "unavailable": {},
    }
    result = {
        **entry,
        "schema_version": 1,
        "artifact_type": "run_result",
        "status": "complete",
        "comparison_eligible": True,
        "arrival_ns": arrival,
        "synchronized_ns": arrival + 45 * ms,
        "plan_sha256": json.loads((folder / "manifest.json").read_text())["plan_sha256"],
        "requests": requests,
        "metrics": metrics,
        "cleanup": {"active_requests": 0, "used_kv_blocks": 0},
        "memory": {"peak_allocated_bytes": 4096, "pool_bytes": 2048},
        "counts": {
            "logical_copy_bytes": 100,
            "stage_tokens": {"prefill": 3, "prelude": 3, "recurrent": 9, "coda": 5},
            "request_work": {
                "A": {
                    "prefill": 2,
                    "prelude": 2,
                    "recurrent": 6,
                    "coda": 3,
                    "prefill_token_traversals": 8,
                },
                "B": {
                    "prefill": 1,
                    "prelude": 1,
                    "recurrent": 3,
                    "coda": 2,
                    "prefill_token_traversals": 4,
                },
            },
            "gate_probabilities": [0.5] * 9,
        },
        "failures": [],
    }
    run = folder / "runs" / entry["run_id"]
    write_json(
        run / "started.json",
        {
            "schema_version": 1,
            "artifact_type": "run_started",
            "run_id": entry["run_id"],
        },
    )
    write_json(run / "result.json", result)
    events = []
    for step, request_id, output_index, time in (
        (0, "A", 0, 10),
        (1, "A", 1, 20),
        (1, "B", 0, 20),
        (2, "A", 2, 40),
        (2, "B", 1, 40),
    ):
        request = requests[0 if request_id == "A" else 1]
        events.append(
            {
                "schema_version": 1,
                "artifact_type": "event",
                "event_seq": len(events),
                "run_id": entry["run_id"],
                "kind": "token_emitted",
                "step_id": step,
                "request_id": request_id,
                "output_index": output_index,
                "host_offset_ns": time * ms,
                "token_id": request["token_ids"][output_index],
                "exit_depth": request["exit_depths"][output_index],
            }
        )
    (run / "events.jsonl").write_text("".join(json.dumps(event) + "\n" for event in events))
    return result


def change_result(folder, run_id, change):
    path = folder / "runs" / run_id / "result.json"
    result = json.loads(path.read_text())
    change(result)
    write_json(path, result)


@pytest.mark.parametrize("nested", [True, False])
def test_hand_accounting_raw_pairs_ranges_and_offline_provenance(tmp_path, monkeypatch, nested):
    import torch

    def forbidden(*args, **kwargs):
        raise AssertionError("offline reporting attempted CUDA access")

    monkeypatch.setattr(torch.cuda, "is_available", forbidden)
    monkeypatch.setattr(torch.cuda, "synchronize", forbidden)
    monkeypatch.setattr(torch.cuda, "get_device_properties", forbidden)
    experiment = tmp_path / "experiment"
    example_experiment(experiment, nested=nested)
    report = build_report(experiment, tmp_path / "report")
    metrics = report["runs"][0]["recomputed_metrics"]
    assert metrics["generated_tokens"] == 5
    assert metrics["delivery_wall_ns"] == 40_000_000
    assert metrics["synchronized_wall_ns"] == 45_000_000
    assert metrics["generated_tokens_per_second"] == 125
    assert metrics["per_request"]["A"]["ttft_ns"] == 10_000_000
    assert metrics["per_request"]["B"]["ttft_ns"] == 20_000_000
    assert metrics["per_request"]["A"]["tpot_ns"] == 15_000_000
    assert metrics["per_request"]["B"]["tpot_ns"] == 20_000_000
    assert metrics["mean_decode_depth"] == 3
    paired = report["comparisons"][0]
    assert paired["complete_pairs"] == 2
    assert paired["mode_ranges"]["refill"]["delivery_wall_ns"] == {
        "values": [40_000_000, 40_000_000],
        "min": 40_000_000,
        "max": 40_000_000,
    }
    assert paired["pairs"][0]["refill_over_no_refill"]["delivery_wall_ns"] == 0.5
    assert report["runs"][0]["result"]["memory"]["peak_allocated_bytes"] == 4096
    assert report["source"]["commit"] == "frozen-source"
    assert (
        report["input_artifact_sha256"]["manifest.json"]
        == hashlib.sha256((experiment / "manifest.json").read_bytes()).hexdigest()
    )
    assert (tmp_path / "report/report.md").is_file()
    assert json.loads((tmp_path / "report/report.json").read_text()) == report


def test_missing_and_failed_runs_preserve_pairs_and_completed_evidence(tmp_path):
    experiment = tmp_path / "experiment"
    plan = example_experiment(experiment)
    first = plan["execution_order"][0]["run_id"]
    (experiment / "runs" / first / "result.json").unlink()
    second = plan["execution_order"][1]["run_id"]
    change_result(
        experiment,
        second,
        lambda row: row.update(
            status="failed", comparison_eligible=False, failures=["injected CUDA failure"]
        ),
    )
    report = build_report(experiment, tmp_path / "report")
    assert report["status"] == "partial"
    assert report["runs"][0]["status"] == "incomplete"
    assert report["runs"][0]["started"]
    assert report["runs"][1]["failures"] == ["injected CUDA failure"]
    assert report["comparisons"][0]["complete_pairs"] == 1
    assert report["comparisons"][0]["pairs"][0]["status"] == "excluded"
    assert report["comparisons"][0]["mode_ranges"]["refill"]["delivery_wall_ns"]["values"] == [
        40_000_000,
    ]


@pytest.mark.parametrize("field", ["controls_sha256", "workload_sha256", "instrumentation"])
def test_incompatible_controls_never_enter_pair_comparison(tmp_path, field):
    experiment = tmp_path / "experiment"
    plan = example_experiment(experiment)
    run_id = plan["execution_order"][1]["run_id"]
    change_result(experiment, run_id, lambda row: row.update({field: "changed"}))
    report = build_report(experiment, tmp_path / "report")
    assert report["comparisons"][0]["complete_pairs"] == 1
    assert any(field in error for error in report["runs"][1]["validation_errors"])
    assert report["cells"][1]["included_observations"] == 1


def test_frozen_pair_itself_with_different_controls_is_excluded(tmp_path):
    experiment = tmp_path / "experiment"
    plan = example_experiment(experiment)
    plan["execution_order"][1]["controls_sha256"] = "different-frozen-controls"
    bind_plan(experiment, plan)
    write_run(experiment, plan["execution_order"][1], index=1, scale=2)
    report = build_report(experiment, tmp_path / "report")
    assert report["comparisons"][0]["complete_pairs"] == 1
    assert "paired controls_sha256" in report["comparisons"][0]["pairs"][0]["reasons"][0]


def test_invalid_stored_metrics_cannot_hide_raw_accounting(tmp_path):
    experiment = tmp_path / "experiment"
    plan = example_experiment(experiment)
    run_id = plan["execution_order"][0]["run_id"]
    change_result(
        experiment, run_id, lambda row: row["metrics"].update(generated_tokens_per_second=999)
    )
    report = build_report(experiment, tmp_path / "report")
    assert report["runs"][0]["recomputed_metrics"]["generated_tokens_per_second"] == 125
    assert not report["runs"][0]["comparison_eligible"]
    assert report["comparisons"][0]["complete_pairs"] == 1


@pytest.mark.parametrize("mutation", ["work", "gate", "cleanup"])
def test_forced_output_agreement_cannot_hide_missing_execution(tmp_path, mutation):
    experiment = tmp_path / "experiment"
    plan = example_experiment(experiment)
    run_id = plan["execution_order"][0]["run_id"]

    def change(row):
        if mutation == "work":
            row["counts"]["request_work"]["A"]["recurrent"] = 5
        elif mutation == "gate":
            row["counts"]["gate_probabilities"].pop()
        else:
            row["cleanup"]["used_kv_blocks"] = 1

    change_result(experiment, run_id, change)
    report = build_report(experiment, tmp_path / "report")
    assert not report["runs"][0]["comparison_eligible"]
    assert report["comparisons"][0]["complete_pairs"] == 1


@pytest.mark.parametrize("mutation", ["duplicate", "timestamp", "shared_step_timestamp"])
def test_invalid_cumulative_output_events_fail_accounting(tmp_path, mutation):
    experiment = tmp_path / "experiment"
    plan = example_experiment(experiment)
    path = experiment / "runs" / plan["execution_order"][0]["run_id"] / "events.jsonl"
    events = [json.loads(line) for line in path.read_text().splitlines()]
    if mutation == "duplicate":
        events.append({**events[-1], "event_seq": 5})
    elif mutation == "timestamp":
        events[0]["host_offset_ns"] += 1
    else:
        events[2]["host_offset_ns"] += 1
    path.write_text("".join(json.dumps(event) + "\n" for event in events))
    report = build_report(experiment, tmp_path / "report")
    assert not report["runs"][0]["comparison_eligible"]
    assert report["comparisons"][0]["pairs"][0]["status"] == "excluded"


def test_measured_order_and_warmup_exclusion(tmp_path):
    experiment = tmp_path / "experiment"
    plan = example_experiment(experiment)
    warmup = {**plan["execution_order"][0], "run_id": "warmup", "phase": "warmup", "pair_id": None}
    plan["execution_order"].insert(0, warmup)
    bind_plan(experiment, plan)
    write_run(experiment, warmup, index=0, scale=100)
    change_result(
        experiment,
        plan["execution_order"][1]["run_id"],
        lambda row: row.update(arrival_ns=10_000_000_000, synchronized_ns=10_045_000_000),
    )
    report = build_report(experiment, tmp_path / "report")
    assert not report["runs"][0]["comparison_eligible"]
    assert report["comparisons"][0]["complete_pairs"] == 0


def test_profile_reports_actual_events_and_does_not_invent_gpu_cost(tmp_path):
    experiment = tmp_path / "experiment"
    example_experiment(experiment)
    profile = experiment / "profiles" / "capture"
    write_json(
        profile / "metadata.json",
        {
            "start_step": 2,
            "end_step": 10,
            "subsequent_outputs": 16,
            "gpu_events": {"kernel_count": 999},
        },
    )
    write_json(
        profile / "trace.json",
        {
            "traceEvents": [
                {"ph": "X", "cat": "user_annotation", "name": "m1.attend", "dur": 100},
                {"ph": "X", "cat": "user_annotation", "name": "m1.nested", "dur": 80},
            ]
        },
    )
    report = build_report(experiment, tmp_path / "report")
    capture = report["profiles"][0]
    assert not capture["gpu_trace_available"]
    assert capture["cpu_inclusive_scopes"][0]["total_us"] == 100
    assert capture["gpu_kernels"] == []
    assert "missing planned CPU/CUDA profiler" in report["next_action"]
    assert any("inclusive" in item for item in capture["limitations"])


def test_profile_ranks_observed_kernels_separately_from_scopes(tmp_path):
    experiment = tmp_path / "experiment"
    example_experiment(experiment)
    profile = experiment / "profiles" / "capture"
    write_json(profile / "metadata.json", {"start_step": 2, "end_step": 10})
    write_json(
        profile / "trace.json",
        {
            "traceEvents": [
                {"ph": "X", "cat": "kernel", "name": "observed_attention", "dur": 100},
                {"ph": "X", "cat": "kernel", "name": "observed_attention", "dur": 50},
                {"ph": "X", "cat": "kernel", "name": "observed_gemm", "dur": 200},
                {"ph": "X", "cat": "gpu_memcpy", "name": "DtoH", "dur": 20},
            ]
        },
    )
    report = build_report(experiment, tmp_path / "report")
    capture = report["profiles"][0]
    assert capture["gpu_trace_available"]
    assert [row["total_us"] for row in capture["gpu_kernels"]] == [200, 150]
    assert capture["gpu_memcpy"][0]["total_us"] == 20


def test_report_refuses_overwrite_and_modified_plan_without_reading_weights(tmp_path):
    experiment = tmp_path / "experiment"
    plan = example_experiment(experiment)
    output = tmp_path / "report"
    build_report(experiment, output)
    before = (output / "report.json").read_bytes()
    with pytest.raises(FileExistsError):
        build_report(experiment, output)
    assert (output / "report.json").read_bytes() == before
    plan["source"]["commit"] = "modified-after-freeze"
    write_json(experiment / "inputs/plan.json", plan)
    with pytest.raises(ValueError, match="hash mismatch"):
        build_report(experiment, tmp_path / "new-report")
    assert not (tmp_path / "new-report").exists()


def test_invalid_result_json_is_preserved_as_failed_accounting(tmp_path):
    experiment = tmp_path / "experiment"
    plan = example_experiment(experiment)
    result = experiment / "runs" / plan["execution_order"][0]["run_id"] / "result.json"
    result.write_text('{"schema_version": 1, "schema_version": 1}')
    report = build_report(experiment, tmp_path / "report")
    assert "duplicate JSON field" in report["runs"][0]["validation_errors"][0]
    assert report["comparisons"][0]["complete_pairs"] == 1
