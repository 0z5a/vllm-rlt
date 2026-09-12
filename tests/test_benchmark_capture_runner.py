"""CPU controller failures retain the prefix and stop task-owned worker groups."""

import signal
import time

import pytest
import test_benchmark_capture_schema as fixtures

from benchmarks.capture import runner as capture
from benchmarks.capture import validation as m3_capture
from vllm_lt.benchmarks import ab, runner
from vllm_lt.benchmarks.schema import write_json

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


@pytest.mark.parametrize("failure", ["worker", "io"])
def test_controller_preserves_failure_and_stops_before_next_worker(
    capture_plan, monkeypatch, tmp_path, failure
):
    monkeypatch.setattr(capture, "verify_plan", lambda *args: None)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    launches = []
    original_handler = signal.getsignal(signal.SIGTERM)

    def launch(plan, worker, output, deadline, **kwargs):
        assert kwargs == {
            "module": "benchmarks.capture",
            "active_deadline": capture.active_case_deadline,
        }
        launches.append(worker["worker_id"])
        if failure == "io":
            raise OSError(5, "Input/output error " + "y" * 10000, "/owned/" + "x" * 2000)
        path = output / "workers" / worker["worker_id"] / "manifest.json"
        path.parent.mkdir()
        write_json(
            path,
            {
                "completed_executions": worker["execution_ids"][:1],
                "status": "failed",
                "passed": False,
            },
        )
        return 1

    monkeypatch.setattr(ab, "_launch_worker", launch)
    result = capture.run_ab(capture_plan, output_dir=tmp_path / "controller")
    assert (
        launches == ["N-A"] and result["completed_workers"] == [] and result["status"] == "failed"
    )
    assert signal.getsignal(signal.SIGTERM) is original_handler
    if failure == "worker":
        assert result["completed_executions"] == capture_plan["workers"][0]["execution_ids"][:1]
    else:
        error = result["failures"][0]
        assert error["type"] == "OSError" and error["errno"] == 5
        assert len(error["filename"]) == 1024 and len(error["message"]) <= 2048
        assert "Input/output error" in error["traceback"] and len(error["traceback"]) <= 8192
