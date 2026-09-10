"""Real tiny CPU executions cover logical/physical observation and M3 evidence reuse."""

import math
import time
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from vllm_lt.models import OuroConfig, OuroForCausalLM
from vllm_lt.validation import m3_inactive as m3
from vllm_lt.validation.schema import read_json, write_json

ROOT = Path(__file__).resolve().parents[1]


def inputs():
    return (
        read_json(ROOT / "benchmarks/fixtures/ouro-q1.json"),
        read_json(ROOT / "benchmarks/fixtures/ouro-q1-contract.json"),
    )


@pytest.fixture(autouse=True)
def no_cuda(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("M3 CPU tests must not discover or initialize CUDA")

    for name in ("is_available", "device_count", "current_device", "_lazy_init"):
        monkeypatch.setattr(torch.cuda, name, forbidden)


def test_exact_model_subset_and_original_gates_have_no_bitwise_promise():
    suite, contract = inputs()
    plan = m3.build_model_plan(suite, contract, OuroConfig().to_dict())
    m3.validate_model_plan(plan)
    assert plan["resource_estimates"] == {
        "cases": 13,
        "comparisons": 28,
        "validation_cases": 11,
        "feasibility_cases": 2,
        "validation_comparisons": 27,
        "retained_tensor_bytes_upper_bound": 3067709328,
        "retained_index_records_upper_bound": 71583,
        "comparison_records_upper_bound": 133776,
        "max_retained_case_bytes": 939262464,
    }
    assert plan["contract"]["comparison_policy"] == contract["comparison_policy"]
    assert all(row["require_exact"] is False for row in plan["comparison_order"])
    assert [row["implementation_id"] for row in plan["execution_order"]] == ["A"] * 9 + ["B"] * 4
    assert (
        sum(row["comparison_kind"] == "implementation_fidelity" for row in plan["comparison_order"])
        == 10
    )
    assert all(
        row["padding"] == m3.PADDING
        for row in plan["execution_order"]
        if row["implementation_id"] == "B"
    )


@pytest.mark.parametrize("change", ["require_exact", "row_count", "logit_tolerance", "extra_case"])
def test_rehashed_relaxed_or_expanded_plan_rejected(change):
    plan = m3.build_model_plan(*inputs(), OuroConfig().to_dict())
    if change == "require_exact":
        plan["comparison_order"][0]["require_exact"] = True
    elif change == "row_count":
        plan["execution_order"][-1]["padding"]["row_count"] = 4
    elif change == "logit_tolerance":
        plan["contract"]["comparison_policy"]["logits"]["float32"]["atol"] = 1.0
    else:
        plan["execution_order"].append(deepcopy(plan["execution_order"][-1]))
    plan["numerical_plan_sha256"] = m3._digest(
        {k: v for k, v in plan.items() if k != "numerical_plan_sha256"}
    )
    with pytest.raises(ValueError):
        m3.validate_model_plan(plan)


@pytest.fixture(scope="module")
def completed_model_run(tmp_path_factory):
    from vllm_lt.core.kv_cache_manager import KVCacheManager

    with pytest.MonkeyPatch.context() as patch:
        initialize = KVCacheManager.__init__

        def cpu(self, *args, **kwargs):
            kwargs["backend"] = "torch"
            initialize(self, *args, **kwargs)

        patch.setattr(KVCacheManager, "__init__", cpu)

        def forbidden(*args, **kwargs):
            pytest.fail("tiny model integration attempted CUDA")

        for name in ("is_available", "device_count", "current_device", "_lazy_init"):
            patch.setattr(torch.cuda, name, forbidden)
        config = OuroConfig.tiny(vocab_size=49152, max_position_embeddings=512)
        torch.manual_seed(41)
        model = OuroForCausalLM(config)
        with torch.no_grad():
            model.lm_head.weight.zero_()
            model.model.early_exit_gate.weight.zero_()
            model.model.early_exit_gate.bias.fill_(math.log(0.4 / 0.6))
        parent = {
            "plan_sha256": "a" * 64,
            "numerical": m3.build_model_plan(*inputs(), config.to_dict()),
        }
        output = tmp_path_factory.mktemp("m3-model")
        a = m3.run_model_rows(model, parent, "A", output, time.perf_counter_ns() + 180 * 10**9)
        assert a["passed"], a["errors"]
        b = m3.run_model_rows(model, parent, "B", output, time.perf_counter_ns() + 180 * 10**9)
        assert b["passed"], b["errors"]
        yield output, parent, a, b


def test_real_engine_padding_and_offline_coverage(completed_model_run):
    output, parent, a, b = completed_model_run
    assert len(a["completed_cases"]) == 9 and len(b["completed_cases"]) == 4
    report = m3.audit_model_rows(output, parent)
    assert report["complete"] and report["passed"], report["errors"]
    assert report["counts"]["verified_qualification_cases"] == 11
    assert report["counts"]["verified_excluded_feasibility_cases"] == 2
    assert report["counts"]["verified_qualification_comparisons"] == 27
    assert report["counts"]["verified_excluded_feasibility_comparisons"] == 1
    paths = [
        output / "numerical/cases" / case["case_id"] / "result.json"
        for case in parent["numerical"]["execution_order"]
        if case["implementation_id"] == "B"
    ]
    observations = [row for path in paths for row in read_json(path)["padding_observations"]]
    assert observations and all(row["row_count"] == 8 for row in observations)
    assert any(len(row["live_rows"]) == 4 for row in observations)
    assert all(row["inactive_hidden_zero"] and row["inactive_gate_zero"] for row in observations)


@pytest.mark.parametrize("corrupt", ["missing", "mapping", "schedule"])
def test_padding_evidence_cannot_be_removed_or_relabelled(completed_model_run, corrupt):
    output, parent, _, _ = completed_model_run
    path = (
        output
        / "numerical/cases"
        / parent["numerical"]["execution_order"][-1]["case_id"]
        / "result.json"
    )
    original = path.read_bytes()
    try:
        result = read_json(path)
        if corrupt == "missing":
            result["padding_observations"] = []
        elif corrupt == "mapping":
            result["padding_observations"][0]["live_rows"] = [0]
        else:
            result["padding_observations"][0]["positions"][0] += 1
        write_json(path, result)
        audited = m3.audit_model_rows(output, parent)
        assert not audited["complete"] and not audited["passed"]
        assert (
            "padding" in audited["errors"][-1]["message"]
            or "padded" in audited["errors"][-1]["message"]
        )
    finally:
        path.write_bytes(original)


def test_layer_cannot_change_physical_mapping_after_core_context(monkeypatch):
    from vllm_lt.config import CacheConfig, SchedulerConfig
    from vllm_lt.sampling_params import SamplingParams
    from vllm_lt.validation.native import ValidationEngine, observe_native

    fixture = {
        "fixture_id": "r",
        "prompt_token_ids": [1, 2],
        "continuation_input_ids": list(range(3, 11)),
        "history_policy": "fixed",
        "forced_exit_depths": [4] * 9,
    }
    model = OuroForCausalLM(OuroConfig.tiny())
    engine = ValidationEngine(
        model,
        fixtures=[fixture],
        history_mode="teacher_forced",
        cache_config=CacheConfig(num_blocks=64, block_size=2),
        scheduler_config=SchedulerConfig(max_num_batched_tokens=64),
        attention_backend="torch",
    )
    original = model.model.layers[0].forward

    def corrupt(hidden, positions, batch, cache):
        return original(
            hidden, positions, replace(batch, live_rows=tuple(reversed(batch.live_rows))), cache
        )

    monkeypatch.setattr(model.model.layers[0], "forward", corrupt)
    engine.add_request(
        "r", [1, 2], SamplingParams(max_tokens=9, ignore_eos=True, exit_threshold=1.0)
    )
    with observe_native(engine, lambda *args: None):
        with pytest.raises(RuntimeError, match="layer prepared physical rows"):
            engine.step()
    if "r" in engine.scheduler.requests:
        engine.abort_request("r")
    assert engine.cache_manager.num_used_blocks == 0


def test_prepared_core_cannot_invent_scheduler_identity(monkeypatch):
    from vllm_lt.config import CacheConfig, SchedulerConfig
    from vllm_lt.sampling_params import SamplingParams
    from vllm_lt.validation.native import ValidationEngine, observe_native

    fixture = {
        "fixture_id": "r",
        "prompt_token_ids": [1, 2],
        "continuation_input_ids": list(range(3, 11)),
        "history_policy": "fixed",
        "forced_exit_depths": [4] * 9,
    }
    engine = ValidationEngine(
        OuroForCausalLM(OuroConfig.tiny()),
        fixtures=[fixture],
        history_mode="teacher_forced",
        attention_backend="torch",
        cache_config=CacheConfig(num_blocks=64, block_size=2),
        scheduler_config=SchedulerConfig(max_num_batched_tokens=64),
    )
    prepare = engine.cache_manager._prepare_batch

    def reordered(*args, **kwargs):
        batch = prepare(*args, **kwargs)
        # Valid owned allocations and dimensions, but reversed logical positions.
        return replace(batch, rows=tuple(reversed(batch.rows)))

    monkeypatch.setattr(engine.cache_manager, "_prepare_batch", reordered)
    engine.add_request(
        "r", [1, 2], SamplingParams(max_tokens=9, ignore_eos=True, exit_threshold=1.0)
    )
    with observe_native(engine, lambda *args: None):
        with pytest.raises(RuntimeError, match="prepared core logical rows"):
            engine.step()
    assert not engine.scheduler.requests and engine.cache_manager.num_used_blocks == 0


def test_padding_runner_override_does_not_retain_engine_cache(monkeypatch):
    import gc
    import weakref

    from vllm_lt.validation import runner

    references = []

    class Runner:
        def _recurrent_padded(self, *args, **kwargs):
            pass

    class Engine:
        def __init__(self):
            self.model_runner = Runner()
            references.append(weakref.ref(self.model_runner))

    monkeypatch.setattr(runner, "ValidationEngine", Engine)
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        with m3._padding_execution({"padding": m3.PADDING}, []):
            engine = runner.ValidationEngine()
            del engine
        assert references[0]() is None
    finally:
        if was_enabled:
            gc.enable()
