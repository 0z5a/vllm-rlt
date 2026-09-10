"""CPU controller failures retain the prefix and stop task-owned worker groups."""

import signal
import subprocess
import time
from types import SimpleNamespace

import pytest
import test_benchmark_capture_schema as fixtures

from vllm_lt.benchmarks import ab, capture, runner
from vllm_lt.benchmarks.schema import write_json
from vllm_lt.validation import m3_capture

capture_plan = fixtures.capture_plan
no_cuda_or_weights = fixtures.no_cuda_or_weights


@pytest.fixture
def worker_host(capture_plan, monkeypatch, tmp_path):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(capture, "verify_plan", lambda *a, **kw: None)
    monkeypatch.setattr(capture, "affinity_snapshot", lambda: fixtures.AFFINITY)
    monkeypatch.setattr(runner, "configure_process", lambda *a: None)
    monkeypatch.setattr(runner, "environment", lambda: {})
    monkeypatch.setattr(runner, "load_model", lambda *a: object())
    monkeypatch.setattr(runner.torch.cuda, "is_initialized", lambda: False)
    released = []

    def release(manifest):
        released.append(manifest["worker_id"])
        manifest["teardown_after_workspace_release"] = {"allocated_bytes": 0, "reserved_bytes": 0}

    monkeypatch.setattr(runner, "release_device", release)
    monkeypatch.setattr(
        runner, "run_loaded_rows", lambda *a, **kw: pytest.fail("timing after failed prerequisite")
    )
    return capture_plan, tmp_path / "output", released


def test_worker_keeps_fully_completed_failed_numerical_case_and_skips_performance(
    worker_host, monkeypatch
):
    plan, output, released = worker_host
    cases = [c for c in plan["numerical"]["execution_order"] if c["implementation_id"] == "A"]

    def held(plan, side, output, deadline):
        for key in ("kernels", "lifecycle"):
            for row in plan[key]["execution_order"]:
                if row["implementation_id"] == side:
                    yield row["evaluation_id"], {"status": "complete", "passed": True}

    monkeypatch.setattr(capture, "run_held_checks", held)

    def numerical(model, parent, side, output, deadline, *, after_case):
        for case in cases[:-1]:
            after_case(case, {"status": "complete"})
        # The real validator deliberately does not call its success callback on
        # the last fully recorded case when a required numerical comparison fails.
        return {
            "complete": True,
            "passed": False,
            "completed_cases": [c["case_id"] for c in cases],
            "errors": [{"case_id": cases[-1]["case_id"], "message": "required gate failed"}],
        }

    monkeypatch.setattr(m3_capture, "run_model_rows", numerical)
    result = capture.run_worker(
        plan, worker_id="N-A", output_dir=output, deadline_ns=time.perf_counter_ns() + 100 * 10**9
    )
    assert result["status"] == "failed" and not result["passed"]
    assert result["completed_executions"] == plan["workers"][0]["execution_ids"][:-7]
    assert result["completed_executions"][-1] == cases[-1]["case_id"]
    assert released == ["N-A"]
    assert result["teardown_after_workspace_release"] == {"allocated_bytes": 0, "reserved_bytes": 0}


def test_partial_environment_initialization_still_releases_owned_device(worker_host, monkeypatch):
    plan, output, released = worker_host

    def partial_environment():
        monkeypatch.setattr(runner.torch.cuda, "is_initialized", lambda: True)
        raise RuntimeError("metadata failed after initialization")

    monkeypatch.setattr(runner, "environment", partial_environment)
    monkeypatch.setattr(
        runner, "load_model", lambda *a: pytest.fail("loaded after environment failure")
    )
    result = capture.run_worker(
        plan, worker_id="N-A", output_dir=output, deadline_ns=time.perf_counter_ns() + 100 * 10**9
    )
    assert result["status"] == "failed" and released == ["N-A"]
    assert result["model_loads"] == 0
    assert result["failures"][0]["message"] == "metadata failed after initialization"


def test_controller_preserves_stopped_prefix_and_never_launches_B(
    capture_plan, monkeypatch, tmp_path
):
    monkeypatch.setattr(capture, "verify_plan", lambda *a: None)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    launches = []
    original_handler = signal.getsignal(signal.SIGTERM)

    def launch(plan, worker, output, deadline, **kwargs):
        launches.append(worker["worker_id"])
        folder = output / "workers" / worker["worker_id"]
        folder.mkdir()
        write_json(
            folder / "manifest.json",
            {
                "completed_executions": worker["execution_ids"][:1],
                "status": "failed",
                "passed": False,
            },
        )
        return 1

    monkeypatch.setattr(ab, "_launch_worker", launch)
    result = capture.run_ab(capture_plan, output_dir=tmp_path / "controller")
    assert launches == ["N-A"]
    assert result["completed_workers"] == [] and result["status"] == "failed"
    assert result["completed_executions"] == capture_plan["workers"][0]["execution_ids"][:1]
    assert signal.getsignal(signal.SIGTERM) is original_handler


def test_lifecycle_watchdog_stays_active_until_terminal_result(capture_plan, tmp_path):
    worker = capture_plan["workers"][0]
    row = capture_plan["lifecycle"]["execution_order"][0]
    folder = tmp_path / "lifecycle/evaluations" / row["evaluation_id"]
    folder.mkdir(parents=True)
    write_json(folder / "started.json", {"evaluation": row, "started_ns": 100, "deadline_ns": 200})
    write_json(folder / "result.pending.json", {"status": "complete"})
    assert capture.active_case_deadline(tmp_path, worker) == 200
    write_json(folder / "result.json", {"status": "failed"})
    assert capture.active_case_deadline(tmp_path, worker) is None


@pytest.mark.parametrize("reason", ["case_deadline", "sigterm", "artifact_cap"])
def test_actual_shared_launcher_cleans_only_its_child_group(
    capture_plan, monkeypatch, tmp_path, reason
):
    (tmp_path / "workers").mkdir()
    worker = capture_plan["workers"][0]
    killed, starts = [], []
    process = SimpleNamespace(pid=123456, done=False)

    def wait(timeout=None):
        if reason == "sigterm" and not killed:
            signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
        if not killed:
            raise subprocess.TimeoutExpired("owned worker", timeout)
        process.done = True
        return -15

    process.wait = wait
    process.poll = lambda: -15 if process.done else None

    def launch(*args, **kwargs):
        starts.append(kwargs)
        return process

    monkeypatch.setattr(ab.subprocess, "Popen", launch)
    monkeypatch.setattr(ab.os, "killpg", lambda pid, sig: killed.append((pid, sig)))
    if reason == "artifact_cap":

        def over_cap(*args):
            raise ValueError("artifact cap exceeded")

        monkeypatch.setattr(ab, "artifact_usage", over_cap)
    previous = signal.getsignal(signal.SIGTERM)

    def stopped(signum, frame):
        raise KeyboardInterrupt("controller received SIGTERM")

    signal.signal(signal.SIGTERM, stopped)
    try:
        with pytest.raises((TimeoutError, KeyboardInterrupt, ValueError)):
            ab._launch_worker(
                capture_plan,
                worker,
                tmp_path,
                time.perf_counter_ns() + 100 * 10**9,
                module="vllm_lt.benchmarks.capture",
                active_deadline=lambda *a: 1 if reason == "case_deadline" else None,
            )
    finally:
        signal.signal(signal.SIGTERM, previous)
    assert killed == [(123456, signal.SIGTERM)] and process.done
    assert starts[0]["start_new_session"] is True
    assert starts[0]["cwd"] == capture_plan["implementations"]["A"]["root"]


def test_controller_io_failure_keeps_bounded_context_without_retry(
    capture_plan, monkeypatch, tmp_path
):
    from vllm_lt.benchmarks.schema import read_json

    monkeypatch.setattr(capture, "verify_plan", lambda *args: None)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    launches = []
    original_handler = signal.getsignal(signal.SIGTERM)
    filename = "/owned/run/" + "x" * 2000

    def launch(plan, worker, output, deadline, **kwargs):
        launches.append(worker["worker_id"])
        raise OSError(5, "Input/output error " + "y" * 10000, filename)

    monkeypatch.setattr(ab, "_launch_worker", launch)
    folder = tmp_path / "controller-io"
    result = capture.run_ab(capture_plan, output_dir=folder)
    assert launches == ["N-A"]
    assert result["status"] == "failed" and result["completed_workers"] == []
    assert result["workers"][0]["returned_ns"] is not None
    failure = result["failures"][0]
    assert failure["type"] == "OSError" and failure["errno"] == 5
    assert failure["filename"] == filename[:1024] and failure["filename2"] is None
    assert len(failure["message"]) <= 2048 and len(failure["traceback"]) <= 8192
    assert "OSError" in failure["traceback"] and "Input/output error" in failure["traceback"]
    assert read_json(folder / "manifest.json")["failures"] == result["failures"]
    assert signal.getsignal(signal.SIGTERM) is original_handler
