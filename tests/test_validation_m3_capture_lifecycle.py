"""CPU stream/graph substitutes qualify harness behavior, not CUDA capture."""

import time
from copy import deepcopy

import pytest
import torch

from benchmarks.capture import lifecycle as capture
from vllm_lt.core.kv_cache_manager import KVCacheManager
from vllm_lt.validation.schema import read_json

pytestmark = pytest.mark.usefixtures("forbid_cuda")


def install_cpu(monkeypatch):
    from types import SimpleNamespace

    from test_recurrent_graph import fake_runtime

    def cache(device):
        assert device.type == "cpu"
        value = KVCacheManager(2, 16, 128, 160, 16, 4)
        # Do not let monkeypatch's undo stack retain the actual cache past the
        # lifecycle's weak-reference cleanup check.
        fake_runtime(monkeypatch, SimpleNamespace(backend="torch"))
        value.backend = "triton"
        return value

    monkeypatch.setattr(capture, "_create_cache", cache)


@pytest.fixture
def completed(tmp_path, monkeypatch):
    install_cpu(monkeypatch)
    plan = capture.build_lifecycle_plan()
    for row in plan["execution_order"]:
        result = capture.run_lifecycle_evaluation(
            plan,
            row["evaluation_id"],
            tmp_path,
            device="cpu",
            deadline_ns=time.perf_counter_ns() + 600 * 10**9,
        )
        assert result["passed"], result
    return tmp_path, plan


def test_plan_has_two_buckets_long_four_row_replay_and_exact_finite_budget():
    plan = capture.build_lifecycle_plan()
    assert len(plan["execution_order"]) == 2
    assert [s["row_count"] for s in plan["steps"] if s["kind"] == "supported"] == [
        8,
        8,
        4,
        4,
        4,
        8,
        4,
    ]
    step = plan["steps"][2]
    assert step["request_ids"] == ["long"] and step["positions"] == [511]
    assert plan["steps"][4]["positions"] == [2, 2]
    assert sum(len(capture._tensor_keys(s)) for s in plan["steps"]) == 173
    assert (
        len(plan["initial_guard_chunks"]) + sum(len(s["guard_chunks"]) for s in plan["steps"])
        == 480
    )
    assert plan["expected_executor"]["bucket_visits"] == {"4": 4, "8": 3}
    assert plan["setup_budget"]["B"] == {"warmups": 6, "captures": 2, "verification_replays": 2}
    bad = deepcopy(plan)
    bad["steps"][2]["positions"] = [510]
    bad["lifecycle_plan_sha256"] = capture.base._digest(
        {k: v for k, v in bad.items() if k != "lifecycle_plan_sha256"}
    )
    with pytest.raises(ValueError, match="matrix"):
        capture.validate_lifecycle_plan(bad)


def test_cpu_pair_proves_bookkeeping_raw_replay_evidence_and_guard_audit(completed):
    root, plan = completed
    report = capture.audit_lifecycle_outputs(root, plan, expected_device="cpu")
    assert report["passed"], report
    assert len(report["evaluations"]) == 2
    assert all(
        r["tensor_records"] == 173 and r["guard_chunks"] == 480 for r in report["evaluations"]
    )


@pytest.mark.parametrize(
    "change", ["prefix", "bucket", "scratch", "held", "guard", "stale", "pointer", "count"]
)
def test_offline_audit_rejects_lifecycle_evidence_gaps(completed, change):
    root, plan = completed
    path = root / "evaluations" / "LIFE-B-triton" / "result.json"
    result = read_json(path)
    if change == "prefix":
        result["steps"][2]["prefixes_at_publication"]["long"][0][1] = 0
    elif change == "bucket":
        result["steps"][2]["after"]["last_dispatch"]["bucket_id"] = 8
    elif change == "scratch":
        result["graph_initial"]["setup"]["scratch"]["restored"] = False
    elif change == "held":
        result["held_checks"].pop()
    elif change == "guard":
        result["steps"][-1]["guard_chunks"].pop()
    elif change == "stale":
        result["steps"][5]["stale_allocation_rejected"] = False
    elif change == "pointer":
        result["steps"][0]["actual_inputs"]["hidden"]["data_ptr"] += 4
    else:
        result["graph_final"]["counters"]["replays"] -= 1
    capture.base._write(path, result)
    report = capture.audit_lifecycle_outputs(root, plan, expected_device="cpu")
    assert not report["passed"] and report["errors"]


def test_partial_lifecycle_retains_failure_and_cleanup_without_false_pass(tmp_path, monkeypatch):
    install_cpu(monkeypatch)
    original = capture._probe_model

    def model(device):
        value = original(device)

        def fail(*args, **kwargs):
            raise KeyboardInterrupt("injected held body failure")

        value._recurrent_tensor = fail
        return value

    monkeypatch.setattr(capture, "_probe_model", model)
    plan = capture.build_lifecycle_plan()
    result = capture.run_lifecycle_evaluation(plan, "LIFE-A-triton", tmp_path, device="cpu")
    assert not result["passed"] and "injected" in result["errors"][0]["message"]
    assert result["cleanup"]["completion_confirmed"]
    assert (tmp_path / "evaluations" / "LIFE-A-triton" / "result.json").is_file()
    assert not capture.audit_lifecycle_outputs(tmp_path, plan, expected_device="cpu")["passed"]
    prefix = capture.audit_lifecycle_outputs(
        tmp_path, plan, expected_device="cpu", allow_prefix=True
    )
    assert (
        prefix["valid_prefix"] and not prefix["known_required_failure"] and not prefix["complete"]
    ), prefix


def test_raw_nonfinite_failure_is_retained_and_not_hidden_by_missing_B(tmp_path, monkeypatch):
    install_cpu(monkeypatch)
    original = capture._probe_model

    def model(device):
        value = original(device)
        recurrent = value._recurrent_tensor

        def bad(hidden, view):
            output = recurrent(hidden, view)
            value.observed[view.row_count]["layer0/attention"][1, 0, 0] = torch.nan
            return output

        value._recurrent_tensor = bad
        return value

    monkeypatch.setattr(capture, "_probe_model", model)
    plan = capture.build_lifecycle_plan()
    result = capture.run_lifecycle_evaluation(plan, "LIFE-A-triton", tmp_path, device="cpu")
    assert not result["passed"] and "nonfinite" in result["errors"][0]["message"]
    report = capture.audit_lifecycle_outputs(
        tmp_path, plan, expected_device="cpu", allow_prefix=True
    )
    assert report["valid_prefix"] and report["known_required_failure"] and not report["passed"], (
        report
    )
