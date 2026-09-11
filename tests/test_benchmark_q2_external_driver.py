import os
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from vllm_lt.benchmarks import q2_external as controller
from vllm_lt.benchmarks import q2_external_driver as driver
from vllm_lt.benchmarks.observe import RunCollector
from vllm_lt.request import RequestOutput


class Official:
    def __init__(self):
        self.consumed = []
        self.started = 0

    def _logits(self):
        result = torch.full((16,), -1.0)
        result[3 + len(self.consumed)] = 2.0
        return result

    def start(self, prompt, max_outputs):
        self.started += 1
        assert prompt == [1, 2]
        assert max_outputs == 4
        return self._logits()

    def advance(self, token):
        assert token == 3 + len(self.consumed)
        self.consumed.append(token)
        return self._logits()


def test_official_driver_consumes_actual_argmax_without_extra_forward():
    official = Official()
    collector = RunCollector(["W1"], arrival_ns=0, max_events=128)
    counts, calls = driver._official_outputs(official, [1, 2], 4, collector, 2**63, True)
    result = collector.finish(synchronized_ns=2**63 - 1)
    assert result["requests"][0]["token_ids"] == [3, 4, 5, 6]
    assert official.started == 1
    assert official.consumed == [3, 4, 5]
    assert counts == {
        "steps": 4,
        "stage_counts": {"prefill": 1, "cached_decode": 3},
        "official_logits_checked": 4,
    }
    assert [r["input_count"] for r in calls] == [2, 1, 1, 1]
    assert [r["last_input_position"] for r in calls] == [1, 2, 3, 4]
    assert [r["token_id"] for r in calls] == [3, 4, 5, 6]
    assert result["requests"][0]["enqueue_start_ns"] is None


def test_official_nonfinite_stops_before_sampling_or_next_call():
    official = Official()
    official._logits = lambda: torch.tensor([float("nan")])
    collector = RunCollector(["W1"], arrival_ns=0)
    with pytest.raises(ValueError, match="nonfinite"):
        driver._official_outputs(official, [1, 2], 4, collector, 2**63, True)
    assert official.started == 1
    assert official.consumed == []
    assert collector.snapshot(synchronized_ns=None)["requests"][0]["token_ids"] == []


class Native:
    def __init__(self):
        self.steps = 0
        self.last_schedule = None

    def add_request(self, request_id, prompt, params):
        assert request_id == "W1"

    def has_unfinished_requests(self):
        return self.steps < 4

    def step(self):
        self.steps += 1
        self.last_schedule = SimpleNamespace(stage=SimpleNamespace(value="coda"), num_tokens=1)
        if self.steps == 1:
            return []
        size = self.steps - 1
        return [
            RequestOutput(
                "W1", [1], [3, 4, 5][:size], [4] * size, size == 3, "length" if size == 3 else None
            )
        ]


def test_native_driver_keeps_cumulative_deltas_and_empty_step_accounting():
    collector = RunCollector(["W1"], arrival_ns=0)
    native = Native()
    counts = driver._native_outputs(native, [1], None, collector, 2**63, 4)
    result = collector.finish(synchronized_ns=2**63 - 1)
    assert counts["steps"] == 4
    assert counts["stage_counts"] == {"coda": 4}
    assert result["requests"][0]["token_ids"] == [3, 4, 5]
    assert len([e for e in result["events"] if e["kind"] == "token_emitted"]) == 3


def test_native_step_budget_stops_without_an_extra_dispatch():
    native = Native()
    collector = RunCollector(["W1"], arrival_ns=0)
    with pytest.raises(RuntimeError, match="step bound"):
        driver._native_outputs(native, [1], None, collector, 2**63, 2)
    assert native.steps == 2


@pytest.mark.parametrize("prefix", ["", "GPU-"])
def test_actual_gpu_uuid_and_affinity_are_checked(prefix):
    controls = {
        "gpu_ids": [0],
        "gpu_uuid": "GPU-cbf66259-f4ab-0ede-1811-82037dde5924",
        "cpu_threads": 1,
        "interop_threads": 1,
        "affinity": {"cpu_ids": [56, 57]},
    }
    observed = {
        "cuda_visible_devices": "0",
        "gpu_uuid": prefix + "cbf66259-f4ab-0ede-1811-82037dde5924",
        "actual_torch_threads": {"intraop": 1, "interop": 1},
        "cpu_affinity": [56, 57],
    }
    controller.check_environment({"contract": {"controls": controls}}, observed)
    with pytest.raises(ValueError, match="UUID"):
        controller.check_environment(
            {"contract": {"controls": controls}}, {**observed, "gpu_uuid": "GPU-other"}
        )
    with pytest.raises(ValueError, match="UUID"):
        controller.check_environment(
            {"contract": {"controls": controls}},
            {**observed, "gpu_uuid": "cbf66259-f4ab-0ede-1811-82037dde5925"},
        )
    with pytest.raises(ValueError, match="affinity"):
        controller.check_environment(
            {"contract": {"controls": controls}}, {**observed, "cpu_affinity": [56]}
        )


def test_source_failure_before_device_work_preserves_failure_without_cuda_cleanup(
    tmp_path, monkeypatch
):
    def reject(plan):
        raise ValueError("source changed")

    monkeypatch.setattr(controller, "verify_plan", reject)
    monkeypatch.setattr(
        controller, "release_device", lambda manifest: pytest.fail("unexpected device cleanup")
    )
    result = controller.run_worker({"plan_sha256": "a" * 64}, tmp_path, 2**63)
    assert result["status"] == "failed"
    assert result["completed_runs"] == []
    assert result["device_initialization"] == "not_started"
    assert result["failures"] == [{"type": "ValueError", "message": "source changed"}]


def test_per_case_budget_includes_partial_files(tmp_path, monkeypatch):
    case = tmp_path / "runs" / "N-feas"
    case.mkdir(parents=True)
    (case / "partial.json").write_bytes(b"12345")
    monkeypatch.setitem(controller.LIMITS, "case_bytes_max", 4)
    with pytest.raises(RuntimeError, match="byte budget"):
        controller.artifact_usage(tmp_path)


def test_artifact_links_rejected(tmp_path):
    (tmp_path / "link").symlink_to(tmp_path / "missing")
    with pytest.raises(ValueError, match="ordinary"):
        controller.artifact_usage(tmp_path)


@pytest.fixture
def worker_stubs(monkeypatch):
    from vllm_lt.benchmarks.q2_external_schema import ARITHMETIC, ENGINE, execution_order
    from vllm_lt.engine import llm_engine
    from vllm_lt.validation import official_cached

    plan = {
        "plan_sha256": "a" * 64,
        "contract": {"engine": ENGINE, "arithmetic": ARITHMETIC},
        "model_config": {},
        "resource_estimates": {"native_pool_bytes": 6442450944},
        "execution_order": execution_order(),
    }
    engine = SimpleNamespace(scheduler=SimpleNamespace(requests={}), last_schedule=None)
    official = SimpleNamespace(close=lambda **kwargs: None)
    monkeypatch.setattr(controller, "verify_plan", lambda plan: None)
    monkeypatch.setattr(controller, "source_probe", lambda: {})
    monkeypatch.setattr(controller, "configure_process", lambda contract: None)
    monkeypatch.setattr(controller, "affinity_snapshot", lambda: {})
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_bf16_reduced_precision_reduction", False)
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_fp16_reduced_precision_reduction", False)
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)
    monkeypatch.setattr(controller, "environment", lambda: {})
    monkeypatch.setattr(controller, "check_environment", lambda *args: None)
    monkeypatch.setattr(
        controller, "load_model", lambda *args: SimpleNamespace(state_dict=lambda: {})
    )
    monkeypatch.setattr(llm_engine, "LLMEngine", lambda *args, **kwargs: engine)
    monkeypatch.setattr(official_cached, "OfficialOuroCachedReference", lambda *args: official)
    monkeypatch.setattr(controller, "shared_weight_proof", lambda *args: {})
    monkeypatch.setattr(controller, "pool_descriptor", lambda *args: {})
    monkeypatch.setattr(controller, "require_empty", lambda *args: {})
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(
        controller,
        "release_device",
        lambda worker: worker.update(
            teardown_after_workspace_release={"allocated_bytes": 0, "reserved_bytes": 0}
        ),
    )
    return plan


def test_failed_feasibility_blocks_warmups_and_measured_rows(tmp_path, monkeypatch, worker_stubs):
    calls = []

    def execute(plan, row, *args, **kwargs):
        calls.append(row["run_id"])
        token = 1 if row["implementation_id"] == "native" else 2
        return {"requests": [{"token_ids": [token] * 64}], "status": "complete", "failures": []}

    monkeypatch.setattr(controller, "execute_case", execute)
    worker = controller.run_worker(worker_stubs, tmp_path, 2**63)
    assert calls == [r["run_id"] for r in worker_stubs["execution_order"][:2]]
    assert worker["completed_runs"] == calls[:1]
    assert worker["status"] == "failed"
    assert "timing blocked" in worker["failures"][0]["message"]
    assert worker["teardown_after_workspace_release"] == {"allocated_bytes": 0, "reserved_bytes": 0}
    failed = controller.read_json(tmp_path / "runs" / calls[1] / "result.json")
    assert failed["status"] == "failed" and failed["requests"][0]["token_ids"] == [2] * 64


def test_slow_final_ack_is_inside_case_deadline(tmp_path, monkeypatch, worker_stubs):
    expired = False
    calls = []
    original_write = controller.write_json

    def write(path, value):
        nonlocal expired
        original_write(path, value)
        if path.name == "acknowledged.json":
            expired = True

    def check(deadline):
        if expired:
            raise TimeoutError("case deadline after ACK write")

    def execute(plan, row, *args, **kwargs):
        calls.append(row["run_id"])
        return {"requests": [{"token_ids": [1] * 64}], "status": "complete", "failures": []}

    monkeypatch.setattr(controller, "write_json", write)
    monkeypatch.setattr(controller, "check_deadline", check)
    monkeypatch.setattr(controller, "execute_case", execute)
    worker = controller.run_worker(worker_stubs, tmp_path, 2**63)
    assert len(calls) == 1
    assert worker["status"] == "failed"
    assert worker["failures"][0]["message"] == "case deadline after ACK write"
    case = tmp_path / "runs" / calls[0]
    assert controller.read_json(case / "result.json")["status"] == "complete"
    assert controller.read_json(case / "completed.json")["result"] == controller.file_record(
        case / "result.json"
    )
    assert controller.read_json(case / "failure.json")["failure"]["type"] == "TimeoutError"


def test_parent_watchdog_stops_its_real_cpu_child_on_case_deadline(tmp_path, monkeypatch):
    """Exercise OS process cleanup with a task-owned CPU child, never CUDA."""
    root = tmp_path / "run"
    child_source = """
import json,pathlib,sys,time
root=pathlib.Path(sys.argv[1])
now=time.perf_counter_ns()
(root/'active-case.json').write_text(json.dumps({'deadline_ns':now+20_000_000}))
while True:
    time.sleep(1)
"""
    real_popen = subprocess.Popen
    children = []

    def launch(command, **kwargs):
        child = real_popen([sys.executable, "-c", child_source, str(root)], **kwargs)
        children.append(child)
        return child

    monkeypatch.setattr(controller, "validate_plan", lambda plan: None)
    monkeypatch.setattr(controller, "verify_plan", lambda plan: None)
    monkeypatch.setattr(controller.subprocess, "Popen", launch)
    plan = {"plan_sha256": "a" * 64, "interpreter": sys.executable, "execution_order": []}
    result = controller.run_plan(plan, root)
    assert result["status"] == "failed"
    assert len(children) == 1 and children[0].poll() is not None
    assert result["launch"]["returncode"] < 0
    assert result["failures"][0]["type"] == "TimeoutError"
    with pytest.raises(ProcessLookupError):
        os.kill(children[0].pid, 0)
