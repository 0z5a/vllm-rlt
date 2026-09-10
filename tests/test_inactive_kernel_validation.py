"""CPU execution and portable evidence tests; these do not qualify CUDA kernels."""

import copy
import json
import time
import weakref

import pytest
import torch

from vllm_lt.validation import m3_inactive_kernels as kernels
from vllm_lt.validation.diagnostics import SpoolBudget, TensorSpool


@pytest.fixture(autouse=True)
def no_cuda(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Kernel planning, CPU execution and auditing must not touch CUDA")

    for name in ("is_available", "device_count", "current_device", "_lazy_init", "synchronize"):
        monkeypatch.setattr(torch.cuda, name, forbidden)


@pytest.fixture
def plan():
    return kernels.build_kernel_plan()


def test_frozen_plan_matches_input_identities_even_with_changed_default_dtype(plan):
    assert len(plan["layouts"]) == 13
    assert len(plan["execution_order"]) == 52
    assert plan["checker"]["status"] == "unavailable"
    assert plan["checker"]["execution_count"] == 0
    assert plan["resource_estimates"]["cache_bytes"] == 83_886_080
    for identity in plan["input_identities"].values():
        assert identity["A"]["logical_input_hashes"] == identity["B"]["logical_input_hashes"]
    assert (
        plan["input_identities"]["L10"]["A"]["input_hashes"]
        != (plan["input_identities"]["L10"]["B"]["input_hashes"])
    )
    old = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        assert kernels.build_kernel_plan() == plan
    finally:
        torch.set_default_dtype(old)
    changed = copy.deepcopy(plan)
    changed["layouts"][12]["lengths"][1] = 510
    changed["kernel_plan_sha256"] = kernels._digest(
        {key: value for key, value in changed.items() if key != "kernel_plan_sha256"}
    )
    with pytest.raises(ValueError, match="frozen"):
        kernels.validate_kernel_plan(changed)


def test_cpu_compact_masked_pair_roundtrips_whole_pool_and_active_output(tmp_path, plan):
    results = []
    for implementation in ("A", "B"):
        result = kernels.run_kernel_evaluation(
            plan,
            f"K-{implementation}-torch-L05",
            tmp_path,
            device="cpu",
            deadline_ns=time.perf_counter_ns() + 60 * 10**9,
        )
        assert result["passed"], result["errors"]
        assert result["cache_references_released"]
        assert len(result["cache_chunks"]) == 40
        assert sum(chunk["size_bytes"] for chunk in result["cache_chunks"]) == 83_886_080
        assert all(
            chunk["actual_sha256"] == chunk["expected_sha256"] for chunk in result["cache_chunks"]
        )
        results.append(result)
    assert results[0]["cache_chunks"] == results[1]["cache_chunks"]
    assert results[1]["active_exact_equal"]
    assert results[1]["output_stats"]["inactive_zero_count"] == 3 * 16 * 128
    assert results[1]["output_stats"]["inactive_negative_zero_count"] == 0
    # A valid prefix remains incomplete; two CPU rows cannot qualify the full matrix.
    assert not kernels.audit_kernel_outputs(tmp_path, plan)["complete"]


@pytest.mark.parametrize("failure", ["attention", "export"])
def test_failure_releases_cache_aliases_and_preserves_result(tmp_path, plan, monkeypatch, failure):
    from vllm_lt.kernels import paged_attention

    references = []
    original_attention = paged_attention.torch_paged_attention
    original_close = TensorSpool.close

    def attention(*args):
        references.extend(weakref.ref(tensor._base) for tensor in args[1:3])
        if failure == "attention":
            raise RuntimeError("injected attention failure")
        return original_attention(*args)

    def close(spool):
        original_close(spool)
        raise OSError("injected export failure")

    monkeypatch.setattr(paged_attention, "torch_paged_attention", attention)
    if failure == "export":
        monkeypatch.setattr(TensorSpool, "close", close)
    result = kernels.run_kernel_evaluation(
        plan,
        "K-A-torch-L02",
        tmp_path,
        device="cpu",
        deadline_ns=time.perf_counter_ns() + 60 * 10**9,
    )
    assert result["status"] == "failed" and not result["passed"]
    assert result["cache_references_released"] and all(ref() is None for ref in references)
    assert json.loads((tmp_path / "evaluations/K-A-torch-L02/result.json").read_text()) == result
    assert any(f"injected {failure}" in error["message"] for error in result["errors"])


def test_expired_case_and_cpu_triton_rejection_leave_marked_incomplete_evidence(tmp_path, plan):
    expired = kernels.run_kernel_evaluation(
        plan, "K-A-torch-L00", tmp_path, device="cpu", deadline_ns=time.perf_counter_ns() - 1
    )
    assert expired["status"] == "incomplete" and not expired["passed"]
    rejected = kernels.run_kernel_evaluation(
        plan,
        "K-A-triton-L00",
        tmp_path,
        device="cpu",
        deadline_ns=time.perf_counter_ns() + 60 * 10**9,
    )
    assert rejected["status"] == "failed" and not rejected["passed"]
    assert "reserved CUDA" in rejected["errors"][0]["message"]
    assert not (tmp_path / "outputs").exists()


def _handwritten_evidence(root, plan):
    """Synthetic outputs exercise audit semantics, not production attention accuracy."""
    chunks = {}
    for layout in plan["layouts"]:
        rows = []
        for component in ("key", "value"):
            for start in range(0, 160, 8):
                tensor = kernels._cache_chunk(
                    plan, layout, component, start, start + 8, written=True
                )
                digest = kernels._tensor_hash(tensor)
                rows.append(
                    dict(
                        component=component,
                        block_start=start,
                        block_stop=start + 8,
                        shape=[8, 2, 16, 16, 128],
                        dtype="float32",
                        size_bytes=2 * 1024**2,
                        actual_sha256=digest,
                        expected_sha256=digest,
                    )
                )
        chunks[layout["layout_id"]] = rows
    for sequence, row in enumerate(plan["execution_order"]):
        layout = next(item for item in plan["layouts"] if item["layout_id"] == row["layout_id"])
        implementation, evaluation_id = row["implementation_id"], row["evaluation_id"]
        live = (
            list(range(len(layout["live_rows"]))) if implementation == "A" else layout["live_rows"]
        )
        count = len(live) if implementation == "A" else layout["row_count"]
        output = torch.zeros(count, 16, 128)
        for logical, physical in enumerate(live):
            output[physical] = (logical + 1) / 8
        metadata = dict(evaluation_id=evaluation_id, layout_id=layout["layout_id"])
        spool = TensorSpool(
            root / "outputs", evaluation_id, layout["layout_id"], SpoolBudget(), "x"
        )
        spool.write("attention_output", output, metadata)
        spool.close()
        marker = dict(
            schema_version=1,
            kernel_plan_sha256=plan["kernel_plan_sha256"],
            evaluation=row,
            started_ns=1 + sequence * 100,
            deadline_ns=90 + sequence * 100,
        )
        result = dict(
            **marker,
            artifact_type="m3_kernel_result",
            finished_ns=80 + sequence * 100,
            status="complete",
            passed=True,
            errors=[],
            cache_references_released=True,
            cache_chunks=chunks[layout["layout_id"]],
            **plan["input_identities"][layout["layout_id"]][implementation],
            output_stats=kernels._output_stats(output, live),
            output_sha256=kernels._tensor_hash(output),
        )
        if implementation == "B":
            result.update(
                reference_evaluation_id=evaluation_id.replace("K-B-", "K-A-"),
                active_exact_equal=True,
            )
        folder = root / "evaluations" / evaluation_id
        folder.mkdir(parents=True)
        kernels._write(folder / "started.json", marker)
        kernels._write(folder / "result.json", result)


def test_offline_raw_audit_rejects_changed_metadata_guards_outputs_and_coverage(tmp_path, plan):
    _handwritten_evidence(tmp_path, plan)
    report = kernels.audit_kernel_outputs(tmp_path, plan)
    assert report["complete"] and report["passed"], report["errors"]
    assert len(report["completed_evaluations"]) == 52

    extra = tmp_path / "outputs/unplanned.bin"
    extra.write_bytes(b"extra")
    assert (
        "raw output files" in kernels.audit_kernel_outputs(tmp_path, plan)["errors"][0]["message"]
    )
    extra.unlink()

    index = tmp_path / "outputs/K-A-torch-L00/L00.index.json"
    original = index.read_text()
    changed = json.loads(original)
    record = changed["records"][0]
    record["metadata"]["evaluation_id"] = "K-A-torch-L01"
    record["record_sha256"] = kernels._digest(
        {key: value for key, value in record.items() if key != "record_sha256"}
    )
    kernels._write(index, changed)
    assert "metadata" in kernels.audit_kernel_outputs(tmp_path, plan)["errors"][0]["message"]
    index.write_text(original)

    result_path = tmp_path / "evaluations/K-A-torch-L00/result.json"
    original_result = result_path.read_text()
    changed = json.loads(original_result)
    changed["cache_chunks"][1]["actual_sha256"] = "0" * 64
    kernels._write(result_path, changed)
    assert "whole-cache" in kernels.audit_kernel_outputs(tmp_path, plan)["errors"][0]["message"]
    result_path.write_text(original_result)

    # Rehashing all local raw-output records must not hide an active A/B mismatch.
    index = tmp_path / "outputs/K-B-torch-L02/L02.index.json"
    changed_index = json.loads(index.read_text())
    changed_output = torch.full((1, 16, 128), 0.25)
    payload = changed_output.view(torch.uint8).numpy().tobytes()
    (index.parent / "L02.bin").write_bytes(payload)
    record = changed_index["records"][0]
    record["sha256"] = kernels._tensor_hash(changed_output)
    record["record_sha256"] = kernels._digest(
        {key: value for key, value in record.items() if key != "record_sha256"}
    )
    kernels._write(index, changed_index)
    result_path = tmp_path / "evaluations/K-B-torch-L02/result.json"
    changed_result = json.loads(result_path.read_text())
    changed_result["output_sha256"] = record["sha256"]
    kernels._write(result_path, changed_result)
    assert (
        "raw compact reference"
        in kernels.audit_kernel_outputs(tmp_path, plan)["errors"][0]["message"]
    )
    result_path.unlink()
    assert not kernels.audit_kernel_outputs(tmp_path, plan)["complete"]
