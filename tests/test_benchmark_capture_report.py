"""Offline synthetic evidence tests; no reported number is a GPU measurement."""

from copy import deepcopy

import pytest
import test_benchmark_capture_schema as schema_fixtures
import torch

from vllm_lt.benchmarks import capture_report as report
from vllm_lt.benchmarks.schema import write_json

capture_plan = schema_fixtures.capture_plan


@pytest.fixture(autouse=True)
def no_cuda_or_model(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("offline capture report touched CUDA or a model loader")

    for name in (
        "is_available",
        "device_count",
        "current_device",
        "init",
        "_lazy_init",
        "synchronize",
    ):
        monkeypatch.setattr(torch.cuda, name, forbidden)
    monkeypatch.setattr(torch, "load", forbidden)


def records_for(plan):
    records = []
    for row in plan["execution_order"]:
        if row["kind"] != "benchmark":
            continue
        result = {
            **row,
            "setup_ns": 100_000_000 if row["implementation_id"] == "A" else 200_000_000,
            "requests": [
                {
                    "request_id": "q",
                    "token_ids": [1, 2],
                    "exit_depths": [4, 4],
                    "finished": True,
                    "finish_reason": "length",
                }
            ],
            "counts": {
                key: []
                for key in (
                    "gate_probabilities",
                    "logical_copy_bytes",
                    "nonfinite_gate_probabilities",
                    "recurrent_depth_counts",
                    "recurrent_occupancy",
                    "request_work",
                    "stage_counts",
                    "stage_tokens",
                )
            },
            "memory": {
                "pool_bytes": plan["workload_stats"][row["workload_id"]]["pool_bytes"],
                "peak_allocated_bytes": 100,
                "peak_reserved_bytes": 200,
                "after_engine_release": {"allocated_bytes": 0, "reserved_bytes": 0},
            },
        }
        records.append(
            {
                "planned": deepcopy(row),
                "run_id": row["run_id"],
                "result": result,
                "status": "complete",
                "validation_errors": [],
                "comparison_eligible": row["phase"] == "measured",
                "recomputed_metrics": {
                    "generated_tokens_per_second": 115 if row["use_graphs"] else 100,
                    "per_request": {"q": {"ttft_ns": 100}},
                },
            }
        )
    return records


def pick(records, side="B", repetition=1, cell="W1-refill"):
    return next(
        r
        for r in records
        if r["planned"]["phase"] == "measured"
        and r["planned"]["implementation_id"] == side
        and r["planned"]["repetition"] == repetition
        and r["planned"]["cell_id"] == cell
    )


def test_all14_pairs_use_exact_controls_and_separate_setup(capture_plan):
    records = records_for(capture_plan)
    for row in records:
        if row["planned"]["phase"] != "measured":
            row["recomputed_metrics"] = {"generated_tokens_per_second": float("nan")}
    cells = report.pair_results(capture_plan, records)
    assert len(cells) == 7 and all(c["status"] == "passed" for c in cells)
    assert sum(len(c["pairs"]) for c in cells) == 14
    pair = cells[0]["pairs"][0]
    assert pair["setup_increase_ns"] == 100_000_000
    assert pair["estimated_setup_break_even_tokens"] == 77
    assert pair["ttft"]["candidate_over_baseline"] == 1


@pytest.mark.parametrize("change", ["tps", "ttft", "control"])
def test_each_pair_must_pass_and_second_good_pair_cannot_hide_failure(capture_plan, change):
    records = records_for(capture_plan)
    row = pick(records, cell="W5-no_refill" if change == "control" else "W1-refill")
    if change == "ttft":
        row["recomputed_metrics"]["per_request"]["q"]["ttft_ns"] = 105.01
    else:
        row["recomputed_metrics"]["generated_tokens_per_second"] = (
            94.99 if change == "control" else 109.99
        )
    cells = report.pair_results(capture_plan, records)
    cell = cells[-1] if change == "control" else cells[0]
    assert cell["status"] == "failed"
    assert [p["status"] for p in cell["pairs"]] == ["failed", "passed"]


def test_passing_pairs_with_overlapping_ranges_are_only_inconclusive(capture_plan):
    records = records_for(capture_plan)
    pick(records, "A", 2)["recomputed_metrics"]["generated_tokens_per_second"] = 115
    pick(records, "B", 2)["recomputed_metrics"]["generated_tokens_per_second"] = 127
    cell = report.pair_results(capture_plan, records)[0]
    assert all(p["status"] == "passed" for p in cell["pairs"])
    assert cell["status"] == "inconclusive" and not cell["strict_range_separation"]
    pick(records)["recomputed_metrics"]["per_request"]["q"]["ttft_ns"] = 106
    assert report.pair_results(capture_plan, records)[0]["status"] == "failed"


@pytest.mark.parametrize(
    "change",
    ["missing", "duplicate", "tokens", "depths", "gates", "work", "worker", "pool", "eligibility"],
)
def test_invalid_pair_is_unqualified(capture_plan, change):
    records = records_for(capture_plan)
    row = pick(records)
    if change == "missing":
        records.remove(row)
    elif change == "duplicate":
        records.append(deepcopy(row))
    elif change in ("tokens", "depths"):
        row["result"]["requests"][0]["token_ids" if change == "tokens" else "exit_depths"][0] += 1
    elif change in ("gates", "work"):
        row["result"]["counts"][
            "gate_probabilities" if change == "gates" else "request_work"
        ].append(1)
    elif change == "worker":
        row["planned"]["worker_id"] = "B2"
    elif change == "pool":
        row["result"]["memory"]["pool_bytes"] = 0
    else:
        row["comparison_eligible"] = False
    cell = report.pair_results(capture_plan, records)[0]
    assert cell["pairs"][0]["status"] == "invalid" and cell["status"] == "inconclusive"


def profile_fixture(*, replay=True):
    def event(name, cat, ts, dur, args=None, tid=7):
        return {
            "name": name,
            "cat": cat,
            "ph": "X",
            "ts": ts,
            "dur": dur,
            "pid": 1,
            "tid": tid,
            "args": args or {},
        }

    kind = "replay" if replay else "eager"
    events, dispatches = [], []
    generations = {4: 1 if replay else 0, 8: 1 if replay else 0}
    initial = {
        "buckets": {
            str(k): {"generation": v, "graph_exec_id": k + 100 if replay else None}
            for k, v in generations.items()
        }
    }
    for i, bucket in enumerate((8, 4, 8), 1):
        generations[bucket] += 1
        start = i * 100
        dispatches.append(
            {
                "dispatch_id": i,
                "bucket_id": bucket,
                "kind": kind,
                "generation": generations[bucket],
                "graph_exec_id": bucket + 100 if replay else None,
            }
        )
        events.append(
            event(
                f"vllm_lt::graph_dispatch::{i}::bucket::{bucket}::{kind}",
                "user_annotation",
                start,
                30,
            )
        )
        if replay:
            events.append(
                event(
                    f"vllm_lt::graph_replay::{bucket}::exec::{bucket + 100}",
                    "user_annotation",
                    start + 1,
                    10,
                )
            )
            events.append(
                event("cudaGraphLaunch", "cuda_runtime", start + 2, 5, {"correlation": i})
            )
            events.append(
                event(
                    "actual_kernel",
                    "kernel",
                    start + 9,
                    10,
                    {"correlation": i, "graph id": bucket + 200},
                    tid=99,
                )
            )
    return {"traceEvents": events}, {
        "use_graphs": replay,
        "profile_dispatches": dispatches,
        "initial": initial,
        "final": {"buckets": {str(k): {"generation": v} for k, v in generations.items()}},
    }


def test_actual_graph_launches_correlate_to_buckets_and_do_not_supply_speedup():
    trace, capture = profile_fixture()
    evidence = report.audit_graph_profile(trace, capture)
    assert evidence["complete"] and evidence["graph_launches"] == 3
    assert evidence["graph_ids_by_bucket"] == {"4": 204, "8": 208}
    assert evidence["graph_kernel_count"] == 3
    assert not any("time" in key or "speed" in key for key in evidence)
    trace, capture = profile_fixture(replay=False)
    assert report.audit_graph_profile(trace, capture)["graph_launches"] == 0


@pytest.mark.parametrize(
    "change",
    [
        "launch",
        "correlation",
        "graph_id",
        "exec_id",
        "bucket",
        "duplicate",
        "foreign_launch",
        "scope",
        "shared_graph",
    ],
)
def test_missing_or_ambiguous_actual_trace_evidence_does_not_qualify(change):
    trace, capture = profile_fixture()
    events = trace["traceEvents"]
    if change == "launch":
        events.pop(2)
    elif change == "correlation":
        events[3]["args"]["correlation"] = 999
    elif change == "graph_id":
        events[3]["args"].pop("graph id")
    elif change == "exec_id":
        capture["profile_dispatches"][0]["graph_exec_id"] += 1
    elif change == "bucket":
        capture["profile_dispatches"][0]["bucket_id"] = 4
    elif change == "duplicate":
        events.append(deepcopy(events[0]))
    elif change == "foreign_launch":
        events.append({**events[2], "ts": 1000})
    elif change == "scope":
        events[2]["tid"] = 8
    else:
        for e in events:
            if e["cat"] == "kernel":
                e["args"]["graph id"] = 200
    with pytest.raises((ValueError, KeyError)):
        report.audit_graph_profile(trace, capture)


def test_missing_parent_artifacts_return_invalid_without_runtime_imports(tmp_path):
    value = report.build_report(tmp_path)
    assert value["evidence_status"] == "invalid" and value["decision"] == "inconclusive"
    assert value["errors"][0]["scope"] == "plan/manifest"


def test_rehashed_source_control_change_is_rejected_before_audits(capture_plan, tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    capture_plan["implementations"]["B"]["source"]["commit"] = "b" * 40
    schema_fixtures.rehash(capture_plan)
    write_json(root / "plan.json", capture_plan)
    value = report.build_report(root)
    assert value["evidence_status"] == "invalid" and value["decision"] == "inconclusive"
    assert value["errors"][0]["scope"] == "plan/manifest"


@pytest.mark.parametrize("metric", ["peak_allocated_bytes", "peak_reserved_bytes"])
def test_each_measured_memory_delta_is_a_required_gate(capture_plan, metric):
    records = records_for(capture_plan)
    row = pick(records)
    row["result"]["memory"][metric] += (
        capture_plan["contract"]["acceptance"]["peak_increase_bytes_max"] + 1
    )
    row["result"]["memory"]["peak_reserved_bytes"] = max(
        row["result"]["memory"]["peak_reserved_bytes"],
        row["result"]["memory"]["peak_allocated_bytes"],
    )
    pair = report.pair_results(capture_plan, records)[0]["pairs"][0]
    assert pair["status"] == "failed" and pair["gates"][metric] is False


@pytest.mark.parametrize(
    "invalid,complete,failure,variation,expected",
    [
        (False, False, True, True, ("incomplete", "failed")),
        (False, False, False, False, ("incomplete", "inconclusive")),
        (True, False, True, False, ("invalid", "inconclusive")),
        (False, True, True, True, ("complete", "failed")),
        (False, True, False, True, ("complete", "inconclusive")),
        (False, True, False, False, ("complete", "passed")),
    ],
)
def test_known_failure_precedes_missing_suffix_but_corruption_is_unqualified(
    invalid, complete, failure, variation, expected
):
    assert (
        report._decision(
            invalid=invalid, complete=complete, hard_failure=failure, variation=variation
        )
        == expected
    )


@pytest.fixture
def cpu_capture(monkeypatch):
    from test_recurrent_graph import fake_runtime, make_cache

    from vllm_lt.models import OuroConfig, OuroForCausalLM
    from vllm_lt.worker.model_runner import ModelRunner

    model = OuroForCausalLM(OuroConfig.tiny())
    cache = make_cache(model)
    fake_runtime(monkeypatch, cache)
    runner = ModelRunner(model, cache)
    runner._enable_recurrent_graph(use_graphs=True)
    initial = runner._graph_snapshot()
    cache.allocate("a", 2)
    runner._recurrent(torch.ones(1, model.config.hidden_size), ["a"], [0], [0])
    final = runner._graph_snapshot()
    cache.free("a")
    runner._close_recurrent_graph()
    capture = {
        "schema_version": 1,
        "implementation_id": "B",
        "use_graphs": True,
        "initial": initial,
        "final": final,
        "closed": runner._graph_snapshot(),
        "cleanup": {"closed": True},
        "profile_dispatches": [],
    }
    return capture, model.config.to_dict()


def audit_cpu_capture(value):
    capture, config = value
    return report.audit_capture_record(
        capture,
        {"implementation_id": "B", "phase": "measured"},
        capture["initial"]["limits"],
        model_config=config,
        block_size=16,
        expected_device="cpu",
    )


def test_actual_cpu_standin_setup_and_counters_audit_only_with_explicit_cpu_override(cpu_capture):
    assert audit_cpu_capture(cpu_capture)["counters"]["replays"] == 1
    capture, config = cpu_capture
    with pytest.raises(ValueError, match="device"):
        report.audit_capture_record(
            capture,
            {"implementation_id": "B", "phase": "measured"},
            capture["initial"]["limits"],
            model_config=config,
            block_size=16,
        )


@pytest.mark.parametrize(
    "change",
    [
        "generation",
        "commit",
        "captured_input",
        "scratch",
        "memory",
        "close",
        "steady_storage",
        "profile",
    ],
)
def test_counter_pointer_setup_or_cleanup_claim_changes_are_rejected(cpu_capture, change):
    capture, _ = cpu_capture
    if change == "generation":
        capture["final"]["buckets"]["4"]["generation"] += 1
    elif change == "commit":
        capture["final"]["counters"]["committed"] = 0
    elif change == "captured_input":
        capture["initial"]["buckets"]["4"]["captured_inputs"]["hidden"]["data_ptr"] += 64
    elif change == "scratch":
        capture["initial"]["setup"]["scratch"]["free_list_after"].reverse()
    elif change == "memory":
        capture["initial"]["setup"]["memory_deltas"]["retained_allocated_bytes"] = 999
    elif change == "close":
        capture["closed"]["buckets"] = capture["final"]["buckets"]
    elif change == "steady_storage":
        capture["final"]["buckets"]["4"]["tensors"]["hidden_in"]["data_ptr"] += 64
    else:
        capture["profile_dispatches"] = [{"unexpected": True}]
    with pytest.raises(ValueError):
        audit_cpu_capture(cpu_capture)


@pytest.fixture
def synthetic_run(capture_plan, tmp_path, monkeypatch):
    """Mock component auditors only; real parent/worker/row/source chronology remains checked."""
    from vllm_lt.benchmarks import capture

    root = tmp_path / "evidence"
    root.mkdir()
    plan = capture_plan
    write_json(root / "plan.json", plan)
    records = {r["run_id"]: r for r in records_for(plan)}
    audits = {
        "complete": True,
        "passed": True,
        "numerical": {
            "complete": True,
            "passed": True,
            "errors": [],
            "ledger": {"case_lifetimes": {}},
        },
        "lifecycle": {"complete": True, "passed": True, "errors": [], "evaluations": []},
        "kernels": {"complete": True, "passed": True, "errors": [], "evaluations": []},
    }
    now = 1_000_000_000
    deadline = now + 7200 * 10**9
    manifest = {
        "schema_version": 1,
        "artifact_type": "m3_capture_manifest",
        "plan_sha256": plan["plan_sha256"],
        "status": "complete",
        "failures": [],
        "started_ns": now,
        "deadline_ns": deadline,
        "completed_workers": [],
        "completed_executions": [],
        "workers": [],
    }
    for worker in plan["workers"]:
        now += 1000
        launch = {"worker_id": worker["worker_id"], "launched_ns": now, "exit_code": 0}
        worker_start = now + 10
        for row in [r for r in plan["execution_order"] if r["worker_id"] == worker["worker_id"]]:
            now += 1000
            start, end = now, now + 500
            lifetime = {
                "started_ns": start,
                "finished_ns": end,
                "deadline_ns": min(deadline, start + 600 * 10**9),
            }
            eid = row["execution_id"]
            if row["kind"] == "benchmark":
                record = records[eid]
                result = record["result"]
                result.update(
                    case_started_ns=start,
                    case_completed_ns=end,
                    arrival_ns=start + 100,
                    synchronized_ns=end - 10,
                    comparison_eligible=row["phase"] == "measured",
                )
                result["counts"]["stage_counts"] = {"recurrent": 1}
                result["capture"] = {
                    "setup_ns": result["setup_ns"],
                    "initial": {"setup": {"started_ns": start + 10, "finished_ns": start + 50}},
                    "final": {"counters": {"calls": 1}},
                }
                folder = root / "runs" / eid
                folder.mkdir(parents=True)
                write_json(folder / "graph-setup.json", result["capture"]["initial"])
                write_json(
                    folder / "started.json",
                    {
                        "schema_version": 1,
                        **row,
                        "started_ns": start,
                        "deadline_ns": lifetime["deadline_ns"],
                    },
                )
                write_json(folder / "result.json", result)
            else:
                if row["kind"] == "model":
                    audits["numerical"]["ledger"]["case_lifetimes"][eid] = lifetime
                    folder = root / "numerical" / "cases" / eid
                else:
                    key = "kernels" if row["kind"] == "kernel" else "lifecycle"
                    audits[key]["evaluations"].append({"evaluation_id": eid, **lifetime})
                    folder = root / key / "evaluations" / eid
                folder.mkdir(parents=True)
                write_json(folder / "started.json", lifetime)
        now += 1000
        controls = plan["contract"]["controls"]
        env = {
            "cuda_visible_devices": "0",
            "logical_device": "cuda:0",
            "cpu_affinity": controls["affinity"]["cpu_ids"],
            "numa_status": controls["affinity"]["numa_status"],
            "actual_torch_threads": {"intraop": 1, "interop": 1},
            "python": plan["dependencies"]["python"],
            "torch_cuda_version": plan["dependencies"]["torch_cuda_build"],
            "software": {},
            "arithmetic": {**plan["benchmark_contract"]["arithmetic"], "cudnn_allow_tf32": False},
            "scheduler": [{"type": "RUN", "gpu_id": 0, "user": "fixture"}],
            "account": "fixture",
            "host": "synthetic",
            "gpu_uuid": "synthetic",
            "gpu_name": "synthetic",
            "total_device_bytes": 0,
            "compute_capability": [0, 0],
            "reservation_environment": {"fixture": True},
        }
        child = {
            "schema_version": 1,
            "artifact_type": "m3_capture_worker_manifest",
            **worker,
            "plan_sha256": plan["plan_sha256"],
            "source": plan["implementations"][worker["implementation_id"]]["source"],
            "harness_sha256": plan["harness"]["sha256"],
            "affinity": controls["affinity"],
            "runtime_environment": plan["runtime_environment"],
            "environment": env,
            "status": "complete",
            "passed": True,
            "failures": [],
            "model_loads": 1,
            "completed_executions": worker["execution_ids"],
            "started_ns": worker_start,
            "ended_ns": now,
            "deadline_ns": deadline,
            "teardown_after_workspace_release": {"allocated_bytes": 0, "reserved_bytes": 0},
        }
        folder = root / "workers" / worker["worker_id"]
        folder.mkdir(parents=True)
        write_json(folder / "manifest.json", child)
        launch["returned_ns"] = now + 10
        manifest["workers"].append(launch)
        manifest["completed_workers"].append(worker["worker_id"])
        manifest["completed_executions"].extend(worker["execution_ids"])
        if worker["worker_id"] == "N-B":
            manifest["numerical_gate_ns"] = now + 20
    manifest["numerical_gate"] = deepcopy(audits)
    manifest["ended_ns"] = now + 1000
    write_json(root / "manifest.json", manifest)
    write_json(root / "numerical-gate.json", audits)
    monkeypatch.setattr(capture, "audit_correctness", lambda *args: deepcopy(audits))

    def record_from_disk(root, row, view, hashes):
        if not (root / "runs" / row["run_id"] / "result.json").exists():
            return {
                "planned": row,
                "run_id": row["run_id"],
                "result": None,
                "status": "incomplete",
                "started": False,
                "comparison_eligible": False,
                "validation_errors": ["planned run did not start"],
                "failures": [],
                "recomputed_metrics": None,
            }
        return deepcopy(records[row["run_id"]])

    monkeypatch.setattr(report, "_run_record", record_from_disk)
    monkeypatch.setattr(report, "audit_capture_record", lambda *args, **kwargs: {"synthetic": True})
    profiles = []
    for row in records.values():
        if row["planned"]["phase"] == "profile":
            folder = root / "profiles" / row["run_id"]
            folder.mkdir(parents=True)
            write_json(folder / "trace.json", {"synthetic": True})
            profiles.append(
                {
                    "capture_id": row["run_id"],
                    "validation_errors": [],
                    "gpu_trace_available": True,
                    "stage_dispatches": {"recurrent": 1},
                }
            )
    monkeypatch.setattr(
        report,
        "_profiles",
        lambda *args: deepcopy(
            [p for p in profiles if (root / "profiles" / p["capture_id"]).exists()]
        ),
    )
    monkeypatch.setattr(report, "audit_graph_profile", lambda *args: {"dispatches": [{}]})
    return root, plan, manifest, records


def test_combined101_row_report_checks_real_worker_order_controls_and_cleanup(synthetic_run):
    root, _, _, _ = synthetic_run
    value = report.build_report(root)
    assert value["errors"] == []
    assert value["evidence_status"] == "complete" and value["decision"] == "passed"
    assert value["counts"] == {
        "planned_executions": 101,
        "completed_executions": 101,
        "benchmark_runs": 78,
        "eligible_timing_runs": 28,
        "profiles": 4,
        "workers": 8,
    }
    assert value["milestone_status"] == "accepted_optional_path_pending_manual_evidence_review"


@pytest.mark.parametrize(
    "change", ["worker_overlap", "source", "memory", "case_deadline", "missing_scope"]
)
def test_combined_report_rejects_inconsistent_retained_evidence(synthetic_run, change, monkeypatch):
    root, plan, manifest, records = synthetic_run
    if change == "worker_overlap":
        manifest["workers"][1]["launched_ns"] = manifest["workers"][0]["returned_ns"] - 1
        write_json(root / "manifest.json", manifest)
    elif change in ("source", "memory"):
        from vllm_lt.benchmarks.schema import read_json

        path = root / "workers" / "N-A" / "manifest.json"
        child = read_json(path)
        if change == "source":
            child["source"]["commit"] = "0" * 40
        else:
            child["teardown_after_workspace_release"]["allocated_bytes"] = -1
        write_json(path, child)
    elif change == "case_deadline":
        next(iter(records.values()))["result"]["case_completed_ns"] += 700 * 10**9
    else:
        monkeypatch.setattr(report, "audit_graph_profile", lambda *args: {"dispatches": []})
    value = report.build_report(root)
    assert value["evidence_status"] == "invalid" and value["decision"] == "inconclusive"


@pytest.mark.parametrize("change", ["inventory", "generation_range", "generation_gap"])
def test_profile_scope_and_record_cannot_drift_together_from_owned_bucket(change):
    trace, capture = profile_fixture()
    if change == "inventory":
        capture["profile_dispatches"][0]["graph_exec_id"] += 99
        trace["traceEvents"][1]["name"] = "vllm_lt::graph_replay::8::exec::207"
    elif change == "generation_range":
        capture["profile_dispatches"][0]["generation"] = 99
    else:
        capture["final"]["buckets"]["8"]["generation"] = 9
        capture["profile_dispatches"][2]["generation"] = 5
    with pytest.raises(ValueError):
        report.audit_graph_profile(trace, capture)


@pytest.mark.parametrize("failed_pair", [False, True])
def test_combined_valid_stopped_prefix_keeps_observed_failure_and_missing_suffix(
    synthetic_run, failed_pair
):
    import shutil

    from vllm_lt.benchmarks.schema import read_json

    root, plan, manifest, records = synthetic_run
    retained = {"N-A", "N-B", "A1", "B1"}
    if failed_pair:
        pick(list(records.values()))["recomputed_metrics"]["generated_tokens_per_second"] = 109
    for worker in plan["workers"]:
        if worker["worker_id"] not in retained:
            shutil.rmtree(root / "workers" / worker["worker_id"])
            for row in plan["execution_order"]:
                if row["worker_id"] == worker["worker_id"] and row["kind"] == "benchmark":
                    shutil.rmtree(root / "runs" / row["run_id"])
    shutil.rmtree(root / "profiles")
    manifest.update(
        status="failed",
        failures=[{"type": "RuntimeError", "message": "retained prefix"}],
        workers=manifest["workers"][:4],
        completed_workers=manifest["completed_workers"][:3],
    )
    manifest["completed_executions"] = [
        r["execution_id"] for r in plan["execution_order"] if r["worker_id"] in retained
    ]
    manifest["ended_ns"] = manifest["workers"][-1]["returned_ns"] + 1000
    write_json(root / "manifest.json", manifest)
    path = root / "workers" / "B1" / "manifest.json"
    child = read_json(path)
    child.update(
        status="failed",
        passed=False,
        failures=[{"type": "RuntimeError", "message": "retained prefix"}],
    )
    write_json(path, child)
    value = report.build_report(root)
    assert value["errors"] == []
    assert value["evidence_status"] == "incomplete"
    assert value["decision"] == ("failed" if failed_pair else "inconclusive")
    assert value["missing"] and value["counts"]["completed_executions"] < 101


def test_observed_failed_worker_cleanup_is_a_failure_not_corrupt_control(synthetic_run):
    from vllm_lt.benchmarks.schema import read_json

    root, _, manifest, _ = synthetic_run
    path = root / "workers" / "P-B" / "manifest.json"
    child = read_json(path)
    child.update(
        status="failed",
        passed=False,
        failures=[{"type": "cleanup", "message": "retained CUDA storage"}],
        teardown_after_workspace_release={"allocated_bytes": 1, "reserved_bytes": 2},
    )
    write_json(path, child)
    manifest.update(
        status="failed",
        completed_workers=manifest["completed_workers"][:-1],
        failures=[{"type": "RuntimeError", "message": "worker failed"}],
    )
    write_json(root / "manifest.json", manifest)
    value = report.build_report(root)
    assert value["errors"] == []
    assert value["evidence_status"] == "incomplete" and value["decision"] == "failed"
    assert any(f["reason"] == "CUDA cleanup" for f in value["hard_failures"])


def test_failed_benchmark_preserves_missing_measurements_and_typed_failure(synthetic_run):
    from vllm_lt.benchmarks.schema import read_json

    root, plan, _, records = synthetic_run
    row = next(r for r in plan["execution_order"] if r["worker_id"] == "B1")
    child = read_json(root / "workers" / "B1" / "manifest.json")
    child["completed_executions"] = []
    value = {
        "schema_version": 1,
        "artifact_type": "run_result",
        **row,
        "plan_sha256": plan["plan_sha256"],
        "status": "failed",
        "comparison_eligible": False,
        "requests": [],
        "metrics": None,
        "cleanup": None,
        "failures": [{"type": "MemoryError", "message": "setup allocation failed"}],
    }
    path = root / "runs" / row["run_id"] / "result.json"
    write_json(path, value)
    output = {"plan_sha256": plan["plan_sha256"], "hard_failures": [], "hashes": {}}
    report._audit_failed_benchmark(root, row, value, output, {"B1": child})
    assert len(output["hard_failures"]) == 1
    assert value["metrics"] is value["cleanup"] is None
    assert str(path.relative_to(root)) in output["hashes"]
    child["completed_executions"] = [row["execution_id"]]
    with pytest.raises(ValueError, match="next"):
        report._audit_failed_benchmark(root, row, value, output, {"B1": child})


@pytest.mark.parametrize("change", ["reused_correlation", "unassigned_graph_kernel"])
def test_actual_graph_kernels_cannot_be_counted_twice_or_without_a_launch(change):
    trace, capture = profile_fixture()
    if change == "reused_correlation":
        trace["traceEvents"][6]["args"]["correlation"] = 1
    else:
        kernel = deepcopy(trace["traceEvents"][3])
        kernel["args"]["correlation"] = 999
        trace["traceEvents"].append(kernel)
    with pytest.raises(ValueError):
        report.audit_graph_profile(trace, capture)
