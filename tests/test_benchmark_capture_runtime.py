"""CPU tests of excluded observation and failure ownership around the M1 loop."""

from types import SimpleNamespace

import pytest
import torch

from benchmarks.capture import runtime
from vllm_lt.benchmarks import runner
from vllm_lt.benchmarks.schema import read_json, write_json


def test_feasibility_checks_replay_return_without_module_hooks():
    class Model:
        def recurrent(self, value):
            return value, value

        def coda(self, value):
            return value

    class Runner:
        def _recurrent(self, value):
            return value, value

    engine = SimpleNamespace(model=Model(), model_runner=Runner())
    with runtime.finite_checks(engine, True) as counts:
        engine.model.recurrent(torch.ones(2))
        engine.model_runner._recurrent(torch.ones(2))
        engine.model.coda(torch.ones(2))
        with pytest.raises(ValueError, match="nonfinite recurrent"):
            engine.model_runner._recurrent(torch.tensor([float("nan")]))
    assert counts == {"recurrent": 2, "coda": 1}
    assert vars(engine.model) == vars(engine.model_runner) == {}
    with runtime.finite_checks(engine, False) as counts:
        assert vars(engine.model) == vars(engine.model_runner) == {}
        engine.model_runner._recurrent(torch.tensor([float("nan")]))
    assert counts == {"recurrent": 0, "coda": 0}


@pytest.mark.parametrize("export_failure", [False, True])
def test_adapter_closes_after_setup_failure_and_retains_primary(
    tmp_path, monkeypatch, export_failure
):
    events = []
    if export_failure:

        def fail_export(*args):
            raise OSError("disk full")

        monkeypatch.setattr(runtime, "write_json", fail_export)

    class Executor:
        status = "failed"

        def snapshot(self):
            return {"status": self.status}

        def close(self):
            events.append("close")
            self.status = "closed"

    class Runner:
        _decode_executor = Executor()

        def _graph_snapshot(self):
            return self._decode_executor.snapshot()

        def _enable_recurrent_graph(self, **kwargs):
            raise RuntimeError("capture failed")

    engine = SimpleNamespace(model_runner=Runner(), scheduler=SimpleNamespace(requests={}))
    adapter = runtime.ExecutionAdapter(implementation_id="B", graph_limits={})
    with pytest.raises(RuntimeError, match="capture failed") as caught:
        adapter.prepare(engine, {"implementation_id": "B", "use_graphs": True}, tmp_path)
    adapter.abort(caught.value, tmp_path)
    assert events == ["close", "close"]
    assert adapter.engine is None
    if export_failure:
        assert len(caught.value.__notes__) == 2
        assert all("disk full" in note for note in caught.value.__notes__)
    else:
        evidence = read_json(tmp_path / "graph-setup-failure.json")
        assert evidence["failed_setup"]["status"] == "failed"
        assert evidence["after_close"]["status"] == "closed"


def test_adapter_retains_uncertain_resources_and_original_error(tmp_path):
    events = []

    class Executor:
        status = "in_flight"

        def settle_failure(self, error):
            events.append("settle")
            self.status = "failed"

        def snapshot(self):
            return {"status": self.status}

        def close(self):
            events.append("close")
            raise RuntimeError("unconfirmed completion")

    engine = SimpleNamespace(
        model_runner=SimpleNamespace(_decode_executor=Executor()),
        scheduler=SimpleNamespace(requests={}),
    )
    adapter = runtime.ExecutionAdapter(implementation_id="B", graph_limits={})
    adapter.engine = engine
    primary = KeyboardInterrupt("stop")
    adapter.abort(primary, tmp_path)
    assert events == ["settle", "close"]
    assert adapter.engine is engine
    assert "unconfirmed completion" in primary.__notes__[0]
    assert (
        read_json(tmp_path / "graph-failure-cleanup.json")["primary"]["type"] == "KeyboardInterrupt"
    )


def test_experiment_rejects_budget_decline_before_any_inference(tmp_path):
    class Runner:
        _decode_executor = None

        def _enable_recurrent_graph(self, **kwargs):
            pass

        def _graph_snapshot(self):
            return {"enabled": False, "budget_decline": {"reason": "capture_budget"}}

        def _close_recurrent_graph(self):
            pass

    engine = SimpleNamespace(model_runner=Runner(), scheduler=SimpleNamespace(requests={}))
    adapter = runtime.ExecutionAdapter(implementation_id="B", graph_limits={})
    with pytest.raises(ValueError, match="declined its budget") as primary:
        adapter.prepare(engine, {"implementation_id": "B", "use_graphs": True}, tmp_path)
    adapter.abort(primary.value, tmp_path)
    evidence = read_json(tmp_path / "graph-setup-failure.json")
    assert evidence["runner_setup"]["budget_decline"]["reason"] == "capture_budget"
    assert adapter.engine is None


def test_shared_loop_calls_adapter_abort_for_setup_systemexit(tmp_path, monkeypatch):
    events = []

    def execute(*args, **kwargs):
        raise SystemExit("setup interrupted")

    monkeypatch.setattr(runner, "_execute", execute)
    adapter = SimpleNamespace(abort=lambda error, folder: events.append((error, folder)))
    row = {"run_id": "test", "workload_id": "W1", "phase": "feasibility"}
    plan = {"suite": {"workloads": [{"workload_id": "W1"}]}, "plan_sha256": "test"}
    with pytest.raises(SystemExit, match="setup interrupted"):
        runner.run_loaded_rows(
            None,
            plan,
            [row],
            output_dir=tmp_path,
            deadline=10**30,
            record_completed=lambda *args: pytest.fail("failed case marked complete"),
            execution_adapter=adapter,
        )
    assert isinstance(events[0][0], SystemExit)
    assert read_json(tmp_path / "runs/test/result.json")["status"] == "failed"


def test_graph_final_export_deadline_prevents_completion(tmp_path, monkeypatch):
    clock = [10]
    run_dir = tmp_path / "runs/test"
    run_dir.mkdir(parents=True)
    write_json(run_dir / "started.json", {"started_ns": 1, "deadline_ns": 100})
    result = {"status": "complete", "memory": {}, "failures": [], "comparison_eligible": True}
    monkeypatch.setattr(runner, "_execute", lambda *args, **kwargs: result)
    monkeypatch.setattr(runner.gc, "collect", lambda: None)
    monkeypatch.setattr(runner.torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(runner, "memory", lambda: {"allocated_bytes": 0, "reserved_bytes": 0})
    monkeypatch.setattr(runner.time, "perf_counter_ns", lambda: clock[0])
    exports = []

    def export(path, value):
        write_json(path, value)
        exports.append(path.name)
        if exports.count("result.pending.json") == 2:
            clock[0] = 101

    monkeypatch.setattr(runner, "write_json", export)
    with pytest.raises(TimeoutError, match="final result export"):
        runner.run_loaded_rows(
            None,
            {"suite": {"workloads": [{"workload_id": "W1"}]}},
            [{"run_id": "test", "workload_id": "W1"}],
            output_dir=tmp_path,
            deadline=1000,
            record_completed=lambda *args: pytest.fail("late export marked complete"),
            execution_adapter=object(),
        )
    assert read_json(run_dir / "result.json")["status"] == "incomplete"
