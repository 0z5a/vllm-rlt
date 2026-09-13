"""Synthetic CPU controller artifacts test the exact mixed-order and stop protocol."""

import signal
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import test_benchmark_ab_schema as fixtures

from benchmarks import ab, ab_schema
from benchmarks import m3_inactive_kernels as kernels
from benchmarks import m3_inactive_report as report
from benchmarks import m3_inactive_run as driver
from benchmarks import m3_inactive_schema as schema
from vllm_lt.benchmarks.schema import read_json, write_json

ab_plan = fixtures.ab_plan


@pytest.fixture
def m3_plan(ab_plan, monkeypatch, tmp_path):
    monkeypatch.setattr(schema, "probe_checkout", ab_schema.probe_checkout)
    monkeypatch.setattr(schema, "affinity_snapshot", lambda: deepcopy(fixtures.AFFINITY))
    path = tmp_path / "m3-contract.json"
    write_json(
        path, read_json(fixtures.ROOT / "benchmarks/fixtures/ouro-m3-inactive-contract.json")
    )
    return schema.make_plan(
        baseline_root=ab_plan["implementations"]["A"]["root"],
        candidate_root=ab_plan["implementations"]["B"]["root"],
        contract_path=path,
        model_path=ab_plan["model_path"],
        gpu_ids=[7],
        affinity=deepcopy(fixtures.AFFINITY),
    )


def rehash(plan):
    plan["plan_sha256"] = schema._digest({k: v for k, v in plan.items() if k != "plan_sha256"})


def test_cpu_plan_freezes_65_model_kernel_rows_without_profiler_or_checker(m3_plan):
    schema.verify_plan(m3_plan)
    rows = m3_plan["execution_order"]
    assert len(rows) == 65 and len({row["execution_id"] for row in rows}) == 65
    assert len(m3_plan["workers"]) == 2
    assert [len(row["execution_ids"]) for row in m3_plan["workers"]] == [35, 30]
    assert sum(row["kind"] == "model" for row in rows) == 13
    assert sum(row["kind"] == "kernel" for row in rows) == 52
    for worker in m3_plan["workers"]:
        selected = [row for row in rows if row["worker_id"] == worker["worker_id"]]
        assert selected[0]["phase"] == "feasibility"
        assert all(row["kind"] == "kernel" for row in selected[1:27])
        assert all(row["kind"] == "model" for row in selected[27:])
    assert m3_plan["contract"]["limits"]["profile_total_bytes_max"] == 0
    assert m3_plan["kernels"]["checker"]["execution_count"] == 0
    assert schema.loading_view(m3_plan)["workload_stats"]["model"]["pool_bytes"] == 1006632960


@pytest.mark.parametrize(
    "change",
    [
        "null_gpu",
        "bool_gpu",
        "extra_case",
        "checker",
        "kernel_layout",
        "precision",
        "affinity",
        "source",
        "unknown",
    ],
)
def test_invalid_rehashed_plan_is_rejected(m3_plan, change):
    plan = deepcopy(m3_plan)
    if change == "null_gpu":
        plan["contract"]["controls"]["gpu_ids"] = None
    elif change == "bool_gpu":
        plan["contract"]["controls"]["gpu_ids"] = [True]
    elif change == "extra_case":
        plan["execution_order"].append(deepcopy(plan["execution_order"][-1]))
    elif change == "checker":
        plan["kernels"]["checker"]["execution_count"] = 13
    elif change == "kernel_layout":
        plan["kernels"]["layouts"][-1]["table_width"] = 64
    elif change == "precision":
        plan["contract"]["controls"]["dtype"] = "bfloat16"
    elif change == "affinity":
        plan["contract"]["controls"]["affinity"]["numactl_show"]["policy"] = "default"
    elif change == "source":
        plan["implementations"]["B"]["source"]["status"] = " M extra.py"
    else:
        plan["unrecognized"] = True
    rehash(plan)
    with pytest.raises((ValueError, TypeError)):
        schema.validate_plan(plan)


def test_embedded_input_drift_rejected_against_original_file(m3_plan):
    plan = deepcopy(m3_plan)
    # Descriptive provenance may change without altering frozen fixture tensor IDs,
    # but the executed embedded contract must still match its source file bytes.
    plan["inputs"]["suite"]["contents"]["provenance"]["prompt_construction"] += " changed"
    plan["numerical"] = schema.build_model_plan(
        plan["inputs"]["suite"]["contents"],
        plan["inputs"]["contract"]["contents"],
        plan["model_config"],
    )
    rehash(plan)
    schema.validate_plan(plan)
    with pytest.raises(ValueError, match="embedded input"):
        schema.verify_plan(plan)


def test_checkpoint_byte_drift_rejected(m3_plan):
    (Path(m3_plan["model_path"]) / "model.safetensors").write_bytes(b"changed")
    with pytest.raises(ValueError, match="checkpoint file"):
        schema.verify_plan(m3_plan)


@pytest.fixture
def controller(m3_plan, monkeypatch, tmp_path):
    clock = [10**12]

    def now():
        clock[0] += 1000
        return clock[0]

    monkeypatch.setattr(driver.time, "perf_counter_ns", now)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7")
    monkeypatch.setattr(driver, "verify_plan", lambda *a, **kw: None)
    monkeypatch.setattr(ab, "audit_worker_controls", lambda *a, **kw: None)
    model_times, kernel_times = {}, {}
    state = {"fail": False, "launches": []}

    def launch(plan, worker, output, deadline, *, module, active_deadline):
        assert module == "benchmarks.m3_inactive_run"
        assert active_deadline is driver.active_deadline
        state["launches"].append(worker["worker_id"])
        started = now()
        completed = []
        for row in plan["execution_order"]:
            if row["worker_id"] != worker["worker_id"]:
                continue
            first, last = now(), now()
            timing = {
                "started_ns": first,
                "finished_ns": last,
                "deadline_ns": min(deadline, first + 600 * 10**9),
            }
            if row["kind"] == "model":
                path = output / "numerical/cases" / row["execution_id"]
                path.mkdir(parents=True)
                write_json(path / "started.json", {"case_id": row["execution_id"]})
                if state["fail"]:
                    break
                model_times[row["execution_id"]] = timing
            else:
                path = output / "kernels/evaluations" / row["execution_id"]
                path.mkdir(parents=True)
                marker = {
                    "evaluation": {"evaluation_id": row["execution_id"]},
                    **timing,
                    "kernel_plan_sha256": plan["kernels"]["kernel_plan_sha256"],
                }
                write_json(path / "started.json", marker)
                write_json(path / "result.json", {**marker, "status": "complete", "passed": True})
                kernel_times[row["execution_id"]] = {"evaluation_id": row["execution_id"], **timing}
            completed.append(row["execution_id"])
        failed = state["fail"]
        value = {
            "schema_version": 1,
            "artifact_type": "m3_inactive_worker_manifest",
            **worker,
            "plan_sha256": plan["plan_sha256"],
            "source": plan["implementations"][worker["implementation_id"]]["source"],
            "harness_sha256": plan["harness"]["sha256"],
            "affinity": plan["contract"]["controls"]["affinity"],
            "runtime_environment": plan["runtime_environment"],
            "model_loads": 1,
            "started_ns": started,
            "ended_ns": now(),
            "deadline_ns": deadline,
            "status": "failed" if failed else "complete",
            "passed": not failed,
            "failures": [{"message": "injected feasibility failure"}] if failed else [],
            "completed_executions": completed,
        }
        folder = output / "workers" / worker["worker_id"]
        folder.mkdir()
        write_json(folder / "manifest.json", value)
        return int(failed)

    monkeypatch.setattr(ab, "_launch_worker", launch)

    def numerical(*args):
        complete = len(model_times) == 13
        return {
            "complete": complete,
            "passed": complete,
            "completed_cases": list(model_times),
            "comparisons": [],
            "errors": [] if complete else [{"message": "incomplete prefix"}],
            "ledger": {"case_lifetimes": deepcopy(model_times)},
            "counts": {
                "planned_cases": 13,
                "verified_cases": 13 if complete else 0,
                "planned_comparisons": 28,
                "verified_comparisons": 28 if complete else 0,
            },
        }

    def kernel(*args):
        complete = len(kernel_times) == 52
        return {
            "complete": complete,
            "passed": complete,
            "completed_evaluations": list(kernel_times),
            "evaluations": deepcopy(list(kernel_times.values())),
            "errors": [],
        }

    monkeypatch.setattr(report, "audit_model_rows", numerical)
    monkeypatch.setattr(kernels, "audit_kernel_outputs", kernel)
    return SimpleNamespace(
        plan=m3_plan,
        output=tmp_path / "run",
        state=state,
        model_times=model_times,
        kernel_times=kernel_times,
    )


def test_shared_launcher_seam_and_mixed65_row_audit(controller):
    value = driver.run(controller.plan, output_dir=controller.output)
    assert value["status"] == "complete", value["failures"]
    assert controller.state["launches"] == ["N-A", "N-B"]
    assert len(value["completed_executions"]) == 65
    checked = report.build_report(controller.output)
    assert checked["evidence_status"] == "complete", checked["errors"]
    assert checked["decision"] == "passed"


def test_normal_failed_feasibility_is_incomplete_failed_not_invalid(controller):
    controller.state["fail"] = True
    value = driver.run(controller.plan, output_dir=controller.output)
    assert value["status"] == "failed" and controller.state["launches"] == ["N-A"]
    checked = report.build_report(controller.output)
    assert checked["evidence_status"] == "incomplete", checked["errors"]
    assert checked["decision"] == "failed" and not checked["errors"]
    assert "stopped" in checked["failure_basis"]


@pytest.mark.parametrize(
    "change",
    [
        "kernel_before_feasibility",
        "model_outside_worker",
        "forged_deadline",
        "worker_overlap",
        "unplanned",
    ],
)
def test_offline_chronology_rejects_control_drift(controller, change):
    driver.run(controller.plan, output_dir=controller.output)
    output = controller.output
    if change == "kernel_before_feasibility":
        controller.kernel_times["K-A-torch-L00"]["started_ns"] -= 100000
    elif change == "model_outside_worker":
        first = controller.plan["numerical"]["execution_order"][0]["case_id"]
        controller.model_times[first]["started_ns"] = 1
    elif change == "forged_deadline":
        controller.kernel_times["K-A-torch-L00"]["deadline_ns"] += 1
    elif change == "unplanned":
        (output / "numerical/cases/extra-unbudgeted-case").mkdir()
    else:
        path = output / "manifest.json"
        manifest = read_json(path)
        manifest["workers"][1]["launched_ns"] = manifest["workers"][0]["returned_ns"] - 1
        write_json(path, manifest)
    checked = report.build_report(output)
    assert checked["evidence_status"] == "invalid" and checked["decision"] != "passed"


def test_shared_launcher_keeps_polling_caps_and_uses_m3_watchdog(m3_plan, tmp_path, monkeypatch):
    class Child:
        pid = 424242
        attempts = 0

        def wait(self, timeout=None):
            self.attempts += 1
            if self.attempts == 1:
                raise ab.subprocess.TimeoutExpired("synthetic child", timeout)
            return 0

        def poll(self):
            return 0

    child = Child()
    calls = []

    def start(command, **kwargs):
        assert command[2] == "benchmarks.m3_inactive_run"
        assert kwargs["start_new_session"] is True
        return child

    monkeypatch.setattr(ab.subprocess, "Popen", start)
    (tmp_path / "workers").mkdir()
    value = ab._launch_worker(
        m3_plan,
        m3_plan["workers"][0],
        tmp_path,
        10**30,
        module="benchmarks.m3_inactive_run",
        active_deadline=lambda *a: calls.append("watchdog"),
    )
    assert value == 0 and calls == ["watchdog", "watchdog"]


def test_watchdog_rejects_unplanned_marker_and_invalid_interval(tmp_path):
    path = tmp_path / "numerical"
    path.mkdir()
    worker = {"execution_ids": ["known"], "worker_id": "N-A"}
    write_json(
        path / "ledger.json",
        {"active_case": {"case_id": "other", "started_ns": 1, "deadline_ns": 2}},
    )
    with pytest.raises(ValueError, match="unplanned"):
        driver.active_deadline(tmp_path, worker)
    write_json(
        path / "ledger.json",
        {"active_case": {"case_id": "known", "started_ns": 2, "deadline_ns": 1}},
    )
    with pytest.raises(ValueError, match="interval"):
        driver.active_deadline(tmp_path, worker)


def test_controller_sigterm_is_recorded_and_handler_restored(controller, monkeypatch):
    original = signal.getsignal(signal.SIGTERM)

    def interrupted(*args, **kwargs):
        signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)

    monkeypatch.setattr(ab, "_launch_worker", interrupted)
    value = driver.run(controller.plan, output_dir=controller.output)
    assert value["status"] == "incomplete" and value["completed_executions"] == []
    assert len(value["workers"]) == 1 and value["workers"][0]["returned_ns"] is not None
    assert signal.getsignal(signal.SIGTERM) == original
