"""Bounded kernel evidence tests with CPU substitutes; no CUDA qualification."""

import time
from copy import deepcopy

import pytest
import torch

from vllm_lt.kernels.paged_attention import torch_paged_attention
from vllm_lt.validation import m3_capture_kernels as capture
from vllm_lt.validation.schema import read_json


@pytest.fixture(autouse=True)
def no_cuda(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("CPU kernel evidence test attempted CUDA work")

    for name in (
        "is_available",
        "device_count",
        "current_device",
        "init",
        "_lazy_init",
        "synchronize",
    ):
        monkeypatch.setattr(torch.cuda, name, forbidden)


def cpu_kernels(device):
    assert device.type == "cpu"

    def scatter(keys, values, blocks, offsets, k, v, active):
        live = active.nonzero().flatten()
        keys[blocks[live], offsets[live]] = k[live]
        values[blocks[live], offsets[live]] = v[live]

    return torch_paged_attention, scatter


@pytest.fixture
def completed(tmp_path, monkeypatch):
    monkeypatch.setattr(capture, "_device_kernels", cpu_kernels)
    plan = capture.build_kernel_plan()
    for row in plan["execution_order"]:
        result = capture.run_kernel_evaluation(
            plan,
            row["evaluation_id"],
            tmp_path,
            device="cpu",
            deadline_ns=time.perf_counter_ns() + 600 * 10**9,
        )
        assert result["passed"], result
    return tmp_path, plan


def test_exact_three_layouts_six_rows_and_prelaunch_input_identity():
    plan = capture.build_kernel_plan()
    assert [(r["live_rows"], r["lengths"]) for r in plan["layouts"]] == [
        ([], []),
        ([1], [512]),
        ([1, 3], [16, 17]),
    ]
    assert len(plan["execution_order"]) == 6
    assert all(r["row_count"] == 4 and r["table_width"] == 32 for r in plan["layouts"])
    assert plan["checker"]["status"] == "unavailable" and plan["checker"]["execution_count"] == 0
    assert plan["resource_estimates"]["output_payload_bytes"] == 122880
    for value in plan["input_identities"].values():
        assert value["A"]["logical_input_hashes"] == value["B"]["logical_input_hashes"]
    bad = deepcopy(plan)
    bad["layouts"][1]["lengths"] = [511]
    bad["kernel_plan_sha256"] = capture.base._digest(
        {k: v for k, v in bad.items() if k != "kernel_plan_sha256"}
    )
    with pytest.raises(ValueError, match="matrix"):
        capture.validate_kernel_plan(bad)


def test_cpu_substitution_preserves_full_pool_and_padded_output_evidence(completed):
    root, plan = completed
    report = capture.audit_kernel_outputs(root, plan, expected_device="cpu")
    assert report["passed"] and len(report["completed_evaluations"]) == 6, report
    assert (
        sum(
            (root / "outputs" / r["evaluation_id"] / (r["layout_id"] + ".bin")).stat().st_size
            for r in plan["execution_order"]
        )
        == 122880
    )


@pytest.mark.parametrize("change", ["guard", "missing", "input", "cleanup", "extra_raw"])
def test_offline_audit_rejects_missing_or_contradictory_evidence(completed, change):
    root, plan = completed
    path = root / "evaluations" / plan["execution_order"][-1]["evaluation_id"] / "result.json"
    result = read_json(path)
    if change == "guard":
        result["cache_chunks"][-1]["actual_sha256"] = "0" * 64
    elif change == "missing":
        result["cache_chunks"].pop()
    elif change == "input":
        result["input_hashes"]["query"] = "0" * 64
    elif change == "cleanup":
        result["completion_confirmed"] = False
    else:
        (root / "outputs" / "extra.bin").write_bytes(b"0")
    capture.base._write(path, result)
    report = capture.audit_kernel_outputs(root, plan, expected_device="cpu")
    assert not report["passed"] and report["errors"]


def test_partial_failure_retains_result_and_does_not_count_completion(tmp_path, monkeypatch):
    def fail(device):
        raise RuntimeError("injected kernel preparation failure")

    monkeypatch.setattr(capture, "_device_kernels", fail)
    plan = capture.build_kernel_plan()
    result = capture.run_kernel_evaluation(
        plan,
        plan["execution_order"][0]["evaluation_id"],
        tmp_path,
        device="cpu",
        deadline_ns=time.perf_counter_ns() + 600 * 10**9,
    )
    assert not result["passed"] and "injected" in result["errors"][0]["message"]
    assert list((tmp_path / "evaluations").rglob("result.json"))
    assert not capture.audit_kernel_outputs(tmp_path, plan, expected_device="cpu")["passed"]
    prefix = capture.audit_kernel_outputs(tmp_path, plan, expected_device="cpu", allow_prefix=True)
    assert (
        prefix["valid_prefix"] and not prefix["known_required_failure"] and not prefix["complete"]
    )


def test_retained_nonfinite_output_is_known_failure_even_with_unexecuted_suffix(
    tmp_path, monkeypatch
):
    def kernels(device):
        attention, scatter = cpu_kernels(device)

        def bad(*args):
            value = attention(*args)
            if value.numel():
                value[0, 0, 0] = torch.nan
            return value

        return bad, scatter

    monkeypatch.setattr(capture, "_device_kernels", kernels)
    plan = capture.build_kernel_plan()
    for row in plan["execution_order"][:2]:
        result = capture.run_kernel_evaluation(
            plan,
            row["evaluation_id"],
            tmp_path,
            device="cpu",
            deadline_ns=time.perf_counter_ns() + 600 * 10**9,
        )
    assert not result["passed"]
    report = capture.audit_kernel_outputs(tmp_path, plan, expected_device="cpu", allow_prefix=True)
    assert report["valid_prefix"] and report["known_required_failure"] and not report["passed"], (
        report
    )
    assert len(report["completed_evaluations"]) == 1
