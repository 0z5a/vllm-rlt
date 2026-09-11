"""Tiny CPU executions use the same case driver, streams, and original policy as Q1."""

import copy
import importlib.metadata
import importlib.util
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from vllm_lt.models import OuroConfig, OuroForCausalLM
from vllm_lt.validation import runner
from vllm_lt.validation.diagnostics import DiagnosticDump, SpoolBudget, TensorSpool
from vllm_lt.validation.schema import _comparisons, _resolve_cases


@pytest.fixture(autouse=True)
def forbid_cuda_discovery(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("CPU execute_case must not discover or initialize CUDA")

    for name in ("is_available", "device_count", "current_device", "_lazy_init"):
        monkeypatch.setattr(torch.cuda, name, forbidden)


@pytest.fixture
def tiny_plan():
    root = Path(__file__).resolve().parents[1]
    suite = json.loads((root / "benchmarks/fixtures/ouro-q1.json").read_text())
    contract = json.loads((root / "benchmarks/fixtures/ouro-q1-contract.json").read_text())
    config = OuroConfig.tiny()
    # Resolve the real 168-case structure with CPU-sized histories and no device work.
    for row in suite["fixtures"] + suite["feasibility_fixtures"]:
        length = (
            3 + int(row["fixture_id"].split("-L")[-1].split("-")[0]) // 64
            if "-L" in row["fixture_id"]
            else 4
        )
        row["prompt_token_ids"] = [
            token % config.vocab_size for token in row["prompt_token_ids"][:length]
        ]
        row["continuation_input_ids"] = [
            token % config.vocab_size for token in row["continuation_input_ids"]
        ]
    suite["original_reproduction"]["prompt_token_ids"] = [
        [token % config.vocab_size for token in prompt]
        for prompt in suite["original_reproduction"]["prompt_token_ids"]
    ]
    contract["engine"]["cache"]["block_size"] = 2
    contract["engine"]["scheduler"]["max_num_batched_tokens"] = 3
    order, fixtures = _resolve_cases(suite, contract, config)
    comparisons = _comparisons(order, fixtures, contract, config)
    return {
        "plan_sha256": "0" * 64,
        "suite": suite,
        "contract": contract,
        "execution_order": order,
        "comparison_order": comparisons,
    }


@pytest.fixture
def tiny_model():
    torch.manual_seed(41)
    model = OuroForCausalLM(OuroConfig.tiny())
    with torch.no_grad():
        model.model.early_exit_gate.weight.zero_()
        model.model.early_exit_gate.bias.fill_(math.log(0.4 / 0.6))
    return model


def execute(model, plan, cases, output):
    budget = SpoolBudget()
    dumps = DiagnosticDump(output / "dumps", ["unselected-a", "unselected-b"], budget=budget)
    results = []
    try:
        for case in cases:
            results.append(
                runner.execute_case(model, plan, case, output, budget, dumps, time.monotonic() + 60)
            )
    finally:
        dumps.close()
    return results


def case_json(output, case):
    return json.loads((output / "cases" / case["case_id"] / "result.json").read_text())


def native_subset(plan, *, family="main", schedule="serial"):
    plan = copy.deepcopy(plan)
    native = next(
        row
        for row in plan["execution_order"]
        if row["family"] == family
        and row["implementation"] == "native"
        and row["dtype"] == "float32"
        and row["schedule"] == schedule
        and row["backend"] == ("torch" if schedule == "serial" else "triton")
        and (family == "live_gate" or row["group_id"] == "Q1-G2")
    )
    native["backend"] = "torch"  # Execute the scheduled case on CPU.
    native["fixture_ids"] = native["fixture_ids"][:2]
    refs = [
        next(
            row
            for row in plan["execution_order"]
            if row["implementation"] == "oracle"
            and row["family"] == family
            and row["dtype"] == "float32"
            and row["fixture_ids"] == [fixture_id]
        )
        for fixture_id in native["fixture_ids"]
    ]
    plan["comparison_order"] = [
        row
        for row in plan["comparison_order"]
        if row["candidate_case_id"] == native["case_id"]
        and row["fixture_id"] in native["fixture_ids"]
    ]
    return plan, refs, native


def read_stream(output, comparison):
    path = output / "comparisons" / (comparison["comparison_id"] + ".jsonl")
    return [json.loads(line) for line in path.read_text().splitlines()], json.loads(
        path.with_suffix(".summary.json").read_text()
    )


@pytest.mark.parametrize("family", ["main", "live_gate"])
@pytest.mark.parametrize("schedule", ["serial", "refill", "no_refill"])
def test_case_driver_writes_complete_native_comparisons_and_nine_actual_predictions(
    tiny_plan, tiny_model, tmp_path, family, schedule
):
    plan, refs, native = native_subset(tiny_plan, family=family, schedule=schedule)
    results = execute(tiny_model, plan, [*refs, native], tmp_path)
    assert all(result["status"] == "complete" for result in results)
    for result in results:
        assert result["cleanup"] == {"requests_remaining": 0, "used_blocks": 0}
        assert result == case_json(tmp_path, result["case"])
        for fixture_id, traces in result["traces"].items():
            fixture = runner.fixtures_for(plan)[fixture_id]
            assert [trace["output_index"] for trace in traces] == list(range(9))
            assert [trace["position"] for trace in traces] == list(
                range(len(fixture["prompt_token_ids"]) - 1, len(fixture["prompt_token_ids"]) + 8)
            )
            assert all("top_two_margin" in trace for trace in traces)
            if family == "main":
                assert [trace["emitted_token_id"] for trace in traces[:8]] == fixture[
                    "continuation_input_ids"
                ]
                assert [trace["exit_depth"] for trace in traces] == fixture["forced_exit_depths"]
            else:
                assert [trace["exit_depth"] for trace in traces] == [4] + [3] * 8
                assert all(
                    trace["actual_token_id"] == trace["emitted_token_id"] for trace in traces
                )
            assert traces[-1]["emitted_token_id"] == traces[-1]["actual_token_id"]
    assert results[-1]["steps"] == len(results[-1]["schedule"])
    assert results[-1]["comparisons"] == [row["comparison_id"] for row in plan["comparison_order"]]
    for comparison in plan["comparison_order"]:
        records, summary = read_stream(tmp_path, comparison)
        assert summary["complete"] and summary["required_failures"] == 0
        assert (
            len(records)
            == summary["count"]
            == results[-1]["boundary_counts"][comparison["fixture_id"]]
        )
        assert len({row["key"] for row in records}) == len(records)
        assert [row["status"] for row in summary["behavior"]] == ["matched"] * 9
        logits = [row for row in records if row["metadata"]["operation"] == "logits"]
        assert len(logits) == 9
        actual_traces = results[-1]["traces"][comparison["fixture_id"]]
        assert [row["stats"]["actual_top1_ids"][0] for row in logits] == [
            row["actual_token_id"] for row in actual_traces
        ]
        kv = [row for row in records if row["metadata"]["operation"] == "populated_kv"]
        assert {row["metadata"]["component"] for row in kv} == {"keys", "values"}
        assert all(row["metadata"]["output_index"] == 8 for row in kv)
        assert all(row["metadata"]["depth_range"] == [1, 4] for row in kv)
        assert all(row["metadata"]["layer_range"] == [0, 1] for row in kv)


def test_legacy_dense_and_packed_streams_keep_four_depths_and_all_prompt_positions(
    tiny_plan, tiny_model, tmp_path
):
    cases = [
        row
        for row in tiny_plan["execution_order"]
        if row["family"] == "original" and row["dtype"] == "float32" and row["backend"] != "triton"
    ]
    ids = {row["case_id"] for row in cases}
    plan = {
        **tiny_plan,
        "comparison_order": [
            row for row in tiny_plan["comparison_order"] if row["candidate_case_id"] in ids
        ],
    }
    results = execute(tiny_model, plan, cases, tmp_path)
    assert len(results) == 4
    for result in results:
        assert result["status"] == "complete" and result["traces"] == {}
        assert result["steps"] == 4
        assert set(result["boundary_counts"].values()) == {4}
        assert result["cleanup"] == {"requests_remaining": 0, "used_blocks": 0}
    for comparison in plan["comparison_order"]:
        records, summary = read_stream(tmp_path, comparison)
        assert summary["complete"] and summary["required_failures"] == 0
        assert [row["metadata"]["depth"] for row in records] == [1, 2, 3, 4]
        assert all(row["metadata"]["positions"] == list(range(5)) for row in records)
        assert all(len(row["stats"]["actual_top1_ids"]) == 5 for row in records)


def official_available():
    try:
        version = importlib.metadata.version("transformers")
    except importlib.metadata.PackageNotFoundError:
        return False
    return version == "4.55.0" and importlib.util.find_spec("kernels") is None


@pytest.mark.skipif(
    not official_available(),
    reason="requires prepared official reference environment",
)
def test_official_case_compares_nine_final_selected_logits(tiny_plan, tiny_model, tmp_path):
    official = next(
        row
        for row in tiny_plan["execution_order"]
        if row["implementation"] == "official" and row["dtype"] == "float32"
    )
    reference = next(
        row
        for row in tiny_plan["execution_order"]
        if row["implementation"] == "oracle"
        and row["family"] == "main"
        and row["dtype"] == "float32"
        and row["fixture_ids"] == official["fixture_ids"]
    )
    plan = {
        **tiny_plan,
        "comparison_order": [
            row
            for row in tiny_plan["comparison_order"]
            if row["candidate_case_id"] == official["case_id"]
        ],
    }
    results = execute(tiny_model, plan, [reference, official], tmp_path)
    assert results[-1]["status"] == "complete"
    assert results[-1]["boundary_counts"] == {official["fixture_ids"][0]: 9}
    records, summary = read_stream(tmp_path, plan["comparison_order"][0])
    assert summary["complete"] and summary["required_failures"] == 0
    assert len(records) == 9 and all(row["metadata"]["depth"] == 4 for row in records)
    assert [row["metadata"]["output_index"] for row in records] == list(range(9))


@pytest.mark.parametrize(
    "implementation,schedule", [("native", "serial"), ("native", "refill"), ("oracle", "serial")]
)
def test_failure_keeps_partial_prediction_trace_counts_and_observed_cleanup(
    tiny_plan, tiny_model, tmp_path, monkeypatch, implementation, schedule
):
    plan, refs, native = native_subset(tiny_plan, schedule=schedule)
    case = native if implementation == "native" else refs[0]
    if implementation == "native":
        execute(tiny_model, plan, refs, tmp_path)
    observed = []
    if implementation == "native":
        base = runner.ValidationEngine

        class BrokenEngine(base):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                observed.append(self)
                execute_step = self.model_runner.execute

                def fail_after_prediction(batch):
                    if len(self.traces[case["fixture_ids"][0]]) >= 2:
                        raise RuntimeError("injected after two actual predictions")
                    return execute_step(batch)

                self.model_runner.execute = fail_after_prediction

        monkeypatch.setattr(runner, "ValidationEngine", BrokenEngine)
    else:
        base = runner.SerialOuroOracle

        class BrokenOracle(base):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                observed.append(self)

            def advance(self, *args, **kwargs):
                if self.output_index >= 1:
                    raise RuntimeError("injected after two actual predictions")
                return super().advance(*args, **kwargs)

        monkeypatch.setattr(runner, "SerialOuroOracle", BrokenOracle)
    with pytest.raises(RuntimeError, match="after two actual"):
        execute(tiny_model, plan, [case], tmp_path)
    result = case_json(tmp_path, case)
    assert result["status"] == "failed" and result["failures"]
    assert len(result["traces"][case["fixture_ids"][0]]) == 2
    assert result["steps"] > 0 and result["boundary_counts"][case["fixture_ids"][0]] > 0
    assert result["cleanup"] == {"requests_remaining": 0, "used_blocks": 0}
    if implementation == "native":
        engine = observed[0]
        assert not engine.scheduler.requests and engine.cache_manager.num_used_blocks == 0
        assert all(not module._forward_hooks for module in tiny_model.modules())
        _, summary = read_stream(tmp_path, plan["comparison_order"][0])
        assert not summary["complete"] and summary["count"] > 0
    else:
        assert observed[0]._closed and observed[0].key_cache is None
        spool = TensorSpool.open(tmp_path / "spools", case["case_id"], case["fixture_ids"][0])
        assert len(spool.index) == result["boundary_counts"][case["fixture_ids"][0]]
        assert (
            len(
                [
                    entry
                    for entry in spool.index.values()
                    if entry["metadata"]["operation"] == "logits"
                ]
            )
            == 2
        )
        spool.close()


def test_expired_case_records_cleanup_and_no_fabricated_completed_predictions(
    tiny_plan, tiny_model, tmp_path
):
    plan, refs, _ = native_subset(tiny_plan)
    budget = SpoolBudget()
    dumps = DiagnosticDump(tmp_path / "dumps", ["unselected-a", "unselected-b"], budget=budget)
    try:
        with pytest.raises(TimeoutError, match="time budget"):
            runner.execute_case(
                tiny_model, plan, refs[0], tmp_path, budget, dumps, time.monotonic() - 1
            )
    finally:
        dumps.close()
    result = case_json(tmp_path, refs[0])
    assert result["status"] == "failed"
    assert result["traces"] == {refs[0]["fixture_ids"][0]: []}
    assert result["steps"] == 0 and set(result["boundary_counts"].values()) == {0}
    assert result["cleanup"] == {"requests_remaining": 0, "used_blocks": 0}


def test_failed_finalization_still_closes_remaining_spools_and_writes_result(
    tiny_plan, tiny_model, tmp_path, monkeypatch
):
    cases = [
        row
        for row in tiny_plan["execution_order"]
        if row["family"] == "original" and row["dtype"] == "float32" and row["backend"] != "triton"
    ]
    candidate = cases[-1]
    plan = {
        **tiny_plan,
        "comparison_order": [
            row
            for row in tiny_plan["comparison_order"]
            if row["candidate_case_id"] == candidate["case_id"]
        ],
    }
    execute(tiny_model, plan, cases[:-1], tmp_path)
    original_close, closed = TensorSpool.close, []

    def close_then_fail_once(self):
        original_close(self)
        if candidate["case_id"] in str(self.data_path):
            closed.append(self.data_path)
            if len(closed) == 1:
                raise OSError("injected first spool finalization failure")

    monkeypatch.setattr(TensorSpool, "close", close_then_fail_once)
    with pytest.raises(OSError, match="first spool finalization"):
        execute(tiny_model, plan, [candidate], tmp_path)
    result = case_json(tmp_path, candidate)
    assert len(closed) == 3
    assert result["status"] == "failed" and result["cleanup"] == {
        "requests_remaining": 0,
        "used_blocks": 0,
    }
    assert result["failures"][-1]["phase"] == "spool_finalization"
    for fixture_id in candidate["fixture_ids"]:
        index = json.loads(
            (tmp_path / "spools" / candidate["case_id"] / f"{fixture_id}.index.json").read_text()
        )
        assert len(index["records"]) == 4


def test_cleanup_failure_records_observed_remaining_requests(
    tiny_plan, tiny_model, tmp_path, monkeypatch
):
    plan, refs, native = native_subset(tiny_plan)
    execute(tiny_model, plan, refs, tmp_path)
    native["max_steps"] = 0  # Stop with the native request still waiting for admission.
    base, observed = runner.ValidationEngine, []

    class BrokenCleanupEngine(base):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            observed.append(self)

        def abort_request(self, request_id):
            return None

    monkeypatch.setattr(runner, "ValidationEngine", BrokenCleanupEngine)
    try:
        with pytest.raises(RuntimeError, match="cleanup leaked"):
            execute(tiny_model, plan, [native], tmp_path)
        result = case_json(tmp_path, native)
        assert result["status"] == "failed"
        assert result["cleanup"] == {"requests_remaining": 1, "used_blocks": 0}
    finally:
        for engine in observed:
            for request_id in list(engine.scheduler.requests):
                base.abort_request(engine, request_id)


@pytest.mark.parametrize("mode", ["normal", "noop", "raises"])
def test_oracle_close_records_before_and_after_even_on_failure(tiny_model, mode, monkeypatch):
    oracle = runner.SerialOuroOracle(
        tiny_model.config, dict(tiny_model.named_parameters()), capacity=4
    )
    close = oracle.close
    progress = {}
    if mode == "noop":
        monkeypatch.setattr(oracle, "close", lambda: None)
    elif mode == "raises":

        def fail():
            raise RuntimeError("injected oracle close failure")

        monkeypatch.setattr(oracle, "close", fail)
    try:
        if mode == "normal":
            runner._close_oracle(oracle, progress)
        else:
            with pytest.raises(RuntimeError, match="oracle"):
                runner._close_oracle(oracle, progress)
        lifecycle = progress["oracle_lifecycle"]
        assert not lifecycle["before_close"]["closed"]
        assert lifecycle["before_close"]["weight_references"] > 0
        assert "key_cache" in lifecycle["before_close"]["retained_fields"]
        assert lifecycle["after_close"]["closed"] is (mode == "normal")
        if mode != "normal":
            assert progress["cleanup"] == {"requests_remaining": 1, "used_blocks": None}
            assert lifecycle["after_close"] == lifecycle["before_close"]
    finally:
        close()


@pytest.fixture
def mocked_outer(tiny_plan, tmp_path, monkeypatch):
    """Every driver/device operation below is a CPU-side mock, including discovery."""
    plan = copy.deepcopy(tiny_plan)
    plan["contract"]["controls"]["gpu_ids"] = [2]
    plan["model_path"] = "/prepared/model"
    plan["execution_order"] = [
        {"case_id": "first", "dtype": "float32"},
        {"case_id": "second", "dtype": "float32"},
        {"case_id": "third", "dtype": "bfloat16"},
    ]
    plan["resource_estimates"] = {
        "parameter_bytes": {"float32": 10},
        "native_pool_bytes": {"float32": 20},
    }
    state = SimpleNamespace(
        calls=[],
        initialized=False,
        tearing_down=False,
        verify_error=None,
        dependencies={"transformers": "4.55.0"},
        kernels=False,
        environment_failure=None,
        fail_case=None,
        execution_exception=RuntimeError,
        fail_cleanup=set(),
        fail_dump=False,
        live_bytes=0,
        manifest_at_environment=None,
        manifests_at_load=[],
        manifests_at_execute=[],
        model_loads=[],
        executed=[],
        numeric_failures={"second"},
        output=tmp_path / "outer",
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")

    def verify(value):
        state.calls.append("verify")
        assert value is plan
        if state.verify_error:
            raise ValueError(state.verify_error)

    def provenance():
        state.calls.append("provenance")
        return {"dependencies": state.dependencies, "optional_kernels_present": state.kernels}

    def environment():
        state.calls.append("environment")
        state.manifest_at_environment = json.loads((state.output / "manifest.json").read_text())
        if state.environment_failure == "before_init":
            raise ValueError("injected reservation/environment rejection")
        state.initialized = True
        if state.environment_failure == "after_init":
            raise ValueError("injected environment metadata failure after initialization")
        return {"mock": True}

    def device_call(name, value=None):
        def call(*args, **kwargs):
            state.calls.append("device:" + name)
            if state.tearing_down and name in state.fail_cleanup:
                raise RuntimeError("injected teardown " + name)
            return value

        return call

    class FakeModel:
        def requires_grad_(self, enabled):
            assert enabled is False

        def eval(self):
            return self

    def load_model(path, *, device, dtype):
        assert path == "/prepared/model" and device == "cuda"
        state.calls.append("load_model")
        state.model_loads.append(str(dtype))
        state.manifests_at_load.append(json.loads((state.output / "manifest.json").read_text()))
        return FakeModel()

    def execute_model(model, actual_plan, case, output, budget, dumps, deadline):
        assert isinstance(model, FakeModel) and actual_plan is plan
        state.calls.append("execute:" + case["case_id"])
        state.executed.append(case["case_id"])
        state.manifests_at_execute.append(json.loads((state.output / "manifest.json").read_text()))
        if case["case_id"] == state.fail_case:
            raise state.execution_exception("injected execution failure")
        folder = output / "cases" / case["case_id"]
        folder.mkdir(parents=True)
        return {
            "status": "complete",
            "elapsed_s": 0.1,
            "numeric_required_failures": int(case["case_id"] in state.numeric_failures),
        }

    class FakeDumps:
        def __init__(self, directory, selected, **kwargs):
            self.selected_fixture_ids = selected
            self.written_bytes = 0
            self.fixture_written_bytes = {}

        def close(self):
            state.calls.append("dump_close")
            state.tearing_down = True
            if state.fail_dump:
                raise OSError("injected dump close failure")

    def memory():
        state.calls.append("device:memory")
        return {"allocated_bytes": state.live_bytes, "reserved_bytes": 0}

    monkeypatch.setattr(runner, "verify_plan", verify)
    monkeypatch.setattr(runner, "official_provenance", provenance)
    monkeypatch.setattr(runner, "environment", environment)
    monkeypatch.setattr(runner, "execute_case", execute_model)
    monkeypatch.setattr(runner, "OuroForCausalLM", SimpleNamespace(from_pretrained=load_model))
    monkeypatch.setattr(runner, "DiagnosticDump", FakeDumps)
    monkeypatch.setattr(runner, "memory", memory)
    for name in ("set_num_threads", "set_num_interop_threads", "manual_seed"):
        monkeypatch.setattr(torch, name, lambda *args: None)
    monkeypatch.setattr(torch.backends, "cuda", SimpleNamespace(matmul=SimpleNamespace()))
    monkeypatch.setattr(torch.backends, "cudnn", SimpleNamespace())
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: state.initialized)
    for name, value in (
        ("mem_get_info", (1000, 2000)),
        ("synchronize", None),
        ("reset_peak_memory_stats", None),
        ("max_memory_allocated", 0),
        ("max_memory_reserved", 0),
        ("empty_cache", None),
    ):
        monkeypatch.setattr(torch.cuda, name, device_call(name, value))
    monkeypatch.setattr(
        torch._C,
        "_cuda_clearCublasWorkspaces",
        device_call("clear_cublas_workspaces"),
        raising=False,
    )
    return plan, state


@pytest.mark.parametrize("rejection", ["plan", "dependencies", "kernels", "visibility"])
def test_outer_rejects_unfrozen_inputs_and_visibility_before_device_or_output(
    mocked_outer, monkeypatch, rejection
):
    plan, state = mocked_outer
    if rejection == "plan":
        state.verify_error = "injected frozen plan mismatch"
    elif rejection == "dependencies":
        state.dependencies["transformers"] = "5.14.1"
    elif rejection == "kernels":
        state.kernels = True
    else:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
    with pytest.raises(ValueError):
        runner.run(plan, state.output)
    assert state.calls == (["verify"] if rejection == "plan" else ["verify", "provenance"])
    assert not state.output.exists()


def test_outer_executes_each_planned_case_once_and_continues_after_numeric_failure(mocked_outer):
    plan, state = mocked_outer
    manifest = runner.run(plan, state.output)
    assert state.executed == ["first", "second", "third"]
    assert state.model_loads == ["torch.float32", "torch.bfloat16"]
    assert manifest["status"] == "complete"
    assert manifest["completed_cases"] == state.executed
    assert manifest["active_case_id"] is None
    assert manifest["failures"] == []
    second = json.loads((state.output / "cases/second/result.json").read_text())
    assert second["numeric_required_failures"] == 1
    assert state.manifest_at_environment["phase"] == "environment"
    assert state.manifest_at_environment["completed_cases"] == []
    assert all(
        row["environment"] is not None and row["phase"] == "model_load"
        for row in state.manifests_at_load
    )
    assert [row["active_case_id"] for row in state.manifests_at_execute] == state.executed
    assert all(row["phase"] == "executing_case" for row in state.manifests_at_execute)
    assert manifest == json.loads((state.output / "manifest.json").read_text())
    assert state.calls[-5:] == [
        "device:synchronize",
        "device:memory",
        "device:clear_cublas_workspaces",
        "device:empty_cache",
        "device:memory",
    ]


@pytest.mark.parametrize(
    "exception,status",
    [(RuntimeError, "failed"), (TimeoutError, "incomplete"), (KeyboardInterrupt, "incomplete")],
)
def test_outer_stops_on_execution_failure_and_preserves_active_case(
    mocked_outer, exception, status
):
    plan, state = mocked_outer
    state.fail_case = "second"
    state.execution_exception = exception
    manifest = runner.run(plan, state.output)
    assert state.executed == ["first", "second"]
    assert manifest["completed_cases"] == ["first"]
    assert manifest["active_case_id"] == "second"
    assert manifest["status"] == status
    assert manifest["failures"][0]["message"] == "injected execution failure"
    assert not (state.output / "cases/second/result.json").exists()
    assert "device:empty_cache" in state.calls and "dump_close" in state.calls


@pytest.mark.parametrize("when", ["before_init", "after_init"])
def test_outer_environment_failure_cleans_only_newly_initialized_device(mocked_outer, when):
    plan, state = mocked_outer
    state.environment_failure = when
    manifest = runner.run(plan, state.output)
    assert manifest["status"] == "failed" and manifest["environment"] is None
    assert state.executed == []
    assert ("device:empty_cache" in state.calls) is (when == "after_init")
    if when == "before_init":
        assert not any(call.startswith("device:") for call in state.calls)
    assert state.calls.count("dump_close") == 1
    assert manifest == json.loads((state.output / "manifest.json").read_text())


@pytest.mark.parametrize(
    "failure",
    ["dumps", "synchronize", "clear_cublas_workspaces", "empty_cache", "live_tensors", "combined"],
)
def test_outer_teardown_failures_remain_visible_without_skipping_later_cleanup(
    mocked_outer, failure
):
    plan, state = mocked_outer
    if failure in ("dumps", "combined"):
        state.fail_dump = True
    if failure in ("synchronize", "clear_cublas_workspaces", "empty_cache"):
        state.fail_cleanup.add(failure)
    if failure == "live_tensors":
        state.live_bytes = 4
    if failure == "combined":
        state.fail_cleanup.update(("synchronize", "clear_cublas_workspaces"))
    manifest = runner.run(plan, state.output)
    assert manifest["status"] == "failed"
    assert manifest["completed_cases"] == ["first", "second", "third"]
    phases = {row["phase"] for row in manifest["failures"] if row["type"] == "cleanup"}
    expected = {"diagnostic_dumps" if failure == "dumps" else failure}
    if failure == "combined":
        expected = {"diagnostic_dumps", "synchronize", "clear_cublas_workspaces"}
    assert phases == expected
    assert "device:empty_cache" in state.calls
    assert manifest["teardown_after_workspace_release"]["allocated_bytes"] == state.live_bytes
    assert manifest == json.loads((state.output / "manifest.json").read_text())
