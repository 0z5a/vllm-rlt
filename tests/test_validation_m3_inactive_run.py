"""Synthetic CPU controller artifacts test the exact mixed-order and stop protocol."""

import signal
from copy import deepcopy
from types import SimpleNamespace

import pytest
import test_validation_m3_inactive_schema as fixtures

from vllm_lt.benchmarks import ab
from vllm_lt.benchmarks.schema import read_json, write_json
from vllm_lt.validation import m3_inactive_kernels as kernels
from vllm_lt.validation import m3_inactive_report as report
from vllm_lt.validation import m3_inactive_run as driver

ab_plan = fixtures.ab_plan
m3_plan = fixtures.m3_plan


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
        assert module == "vllm_lt.validation.m3_inactive_run"
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
        assert command[2] == "vllm_lt.validation.m3_inactive_run"
        assert kwargs["start_new_session"] is True
        return child

    monkeypatch.setattr(ab.subprocess, "Popen", start)
    (tmp_path / "workers").mkdir()
    value = ab._launch_worker(
        m3_plan,
        m3_plan["workers"][0],
        tmp_path,
        10**30,
        module="vllm_lt.validation.m3_inactive_run",
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
