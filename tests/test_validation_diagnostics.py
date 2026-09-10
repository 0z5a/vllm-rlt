"""Hand-recorded tensor evidence, corruption cases, and independent numeric checks."""

import hashlib
import json
import math

import pytest
import torch

from vllm_lt.validation.diagnostics import (
    BudgetExceededError,
    DiagnosticDump,
    NonfiniteComparisonError,
    SpoolBudget,
    TensorSpool,
    compare,
    comparison_json,
)

POLICY = {
    "policy_id": "original-logits-v1",
    "logits": {
        "float32": {"atol": 0.001, "rtol": 0.0001},
        "bfloat16": {"atol": 0.25, "rtol": 0.02},
    },
    "diagnostic_only": ["layer_output", "populated_kv", "gate_logits", "bf16_fp32_sensitivity"],
}


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_raw_roundtrip_is_typed_independent_and_append_only(tmp_path, dtype):
    budget = SpoolBudget()
    spool = TensorSpool(tmp_path, "oracle-fp32", "fixture-1", budget, "group-1")
    first = torch.arange(12, dtype=torch.float32).reshape(3, 4).to(dtype).T
    expected = first.clone()
    metadata = {"fixture_id": "fixture-1", "positions": (3,), "depth": 2, "layer": 1}
    entry = spool.write("p3/d2/l1/key", first, metadata)
    prefix = spool.data_path.read_bytes()
    first.fill_(99)
    spool.write("p3/d2/gate", torch.tensor(0.125, dtype=dtype), {})
    assert spool.data_path.read_bytes().startswith(prefix)
    torch.testing.assert_close(spool.read("p3/d2/l1/key"), expected, atol=0, rtol=0)
    spool.close()
    reader = TensorSpool.open(tmp_path, "oracle-fp32", "fixture-1")
    assert reader.index["p3/d2/l1/key"]["metadata"]["positions"] == [3]
    copied_metadata = reader.read_metadata("p3/d2/l1/key")
    copied_metadata["positions"].append(9)
    assert reader.read_metadata("p3/d2/l1/key")["positions"] == [3]
    assert entry["size_bytes"] == expected.numel() * expected.element_size()
    actual = reader.read("p3/d2/l1/key")
    assert actual.dtype == dtype and actual.device.type == "cpu"
    assert actual.shape == expected.shape
    actual.zero_()
    torch.testing.assert_close(reader.read("p3/d2/l1/key"), expected, atol=0, rtol=0)
    assert reader.read("p3/d2/gate").shape == ()
    assert budget.written_bytes == 13 * expected.element_size()


def test_bfloat16_bit_patterns_survive_without_float_conversion(tmp_path):
    patterns = torch.tensor([0, 0x8000, 0x3F80, 0x7F80, 0x7FC1, 0xFFFF], dtype=torch.uint16)
    tensor = patterns.view(torch.bfloat16)
    spool = TensorSpool(tmp_path, "bf16", "fixture", SpoolBudget(), "group")
    spool.write("bits", tensor, {})
    spool.close()
    actual = TensorSpool.open(tmp_path, "bf16", "fixture").read("bits")
    assert torch.equal(actual.view(torch.uint16), patterns)


def test_empty_tensor_and_empty_spool(tmp_path):
    spool = TensorSpool(tmp_path, "empty", "fixture", SpoolBudget(), "group")
    spool.write("empty", torch.empty(2, 0), {})
    spool.close()
    reader = TensorSpool.open(tmp_path, "empty", "fixture")
    assert reader.read("empty").shape == (2, 0)
    assert reader.data_path.stat().st_size == 0


def test_duplicate_and_invalid_metadata_do_not_append_or_claim_bytes(tmp_path):
    budget = SpoolBudget()
    spool = TensorSpool(tmp_path, "oracle", "fixture", budget, "group")
    value = torch.ones(2)
    spool.write("same", value, {})
    with pytest.raises(ValueError, match="Duplicate"):
        spool.write("same", value, {})
    with pytest.raises(ValueError):
        spool.write("bad-metadata", value, {"bad": float("nan")})
    assert budget.written_bytes == spool.data_path.stat().st_size == 8
    spool.close()
    with pytest.raises(ValueError, match="closed"):
        spool.write("later", value, {})
    with pytest.raises(FileExistsError):
        TensorSpool(tmp_path, "oracle", "fixture", budget, "group")


def test_shared_budget_counts_groups_and_both_dtype_passes(tmp_path):
    budget = SpoolBudget(total_bytes=16, per_group_bytes=12)
    fp32 = TensorSpool(tmp_path, "fp32", "fixture", budget, "same-group")
    bf16 = TensorSpool(tmp_path, "bf16", "fixture", budget, "same-group")
    fp32.write("row", torch.ones(2), {})
    bf16.write("row", torch.ones(2, dtype=torch.bfloat16), {})
    with pytest.raises(BudgetExceededError, match="group"):
        bf16.write("overflow", torch.ones(2, dtype=torch.bfloat16), {})
    fp32.close()
    bf16.close()
    second = TensorSpool(tmp_path, "fp32", "fixture-2", budget, "second-group")
    second.write("row", torch.ones(1), {})
    with pytest.raises(BudgetExceededError, match="Cumulative"):
        second.write("overflow", torch.ones(1), {})
    second.close()
    assert budget.written_bytes == 16
    assert budget.group_written_bytes == {"same-group": 12, "second-group": 4}


def test_snapshot_and_read_budget_fail_before_allocating_staging(tmp_path, monkeypatch):
    spool = TensorSpool(tmp_path, "oracle", "fixture", SpoolBudget(), "group", max_tensor_bytes=8)
    tensor = torch.ones(3)
    original_to = torch.Tensor.to

    def forbidden(*args, **kwargs):
        raise AssertionError("staging allocation occurred before budget validation")

    monkeypatch.setattr(torch.Tensor, "to", forbidden)
    with pytest.raises(BudgetExceededError, match="single-boundary"):
        spool.write("oversized", tensor, {})
    monkeypatch.setattr(torch.Tensor, "to", original_to)
    spool.write("row", tensor[:2], {})
    spool.close()
    reader = TensorSpool.open(tmp_path, "oracle", "fixture", matching_bytes=7)
    with pytest.raises(BudgetExceededError, match="read"):
        reader.read("row")


@pytest.mark.parametrize(
    "corruption",
    ["payload", "truncated", "trailing", "shape", "dtype", "offset", "size_bytes", "duplicate"],
)
def test_corrupt_evidence_is_rejected(tmp_path, corruption):
    spool = TensorSpool(tmp_path, "oracle", "fixture", SpoolBudget(), "group")
    spool.write("row", torch.arange(6, dtype=torch.float32).reshape(2, 3), {})
    spool.close()
    if corruption == "payload":
        payload = bytearray(spool.data_path.read_bytes())
        payload[0] ^= 1
        spool.data_path.write_bytes(payload)
        with pytest.raises(ValueError, match="SHA-256"):
            TensorSpool.open(tmp_path, "oracle", "fixture").read("row")
        return
    if corruption in ("truncated", "trailing"):
        payload = spool.data_path.read_bytes()
        spool.data_path.write_bytes(payload[:-1] if corruption == "truncated" else payload + b"x")
    else:
        document = json.loads(spool.index_path.read_text())
        if corruption == "duplicate":
            document["records"].append(document["records"][0])
        else:
            document["records"][0][corruption] = {
                "shape": [3, 2],
                "dtype": "int32",
                "offset": 4,
                "size_bytes": 20,
            }[corruption]
        spool.index_path.write_text(json.dumps(document))
    with pytest.raises(ValueError):
        TensorSpool.open(tmp_path, "oracle", "fixture")


def test_diagnostics_match_hand_computed_error_and_kv_coordinate():
    reference = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    actual = torch.tensor([[1.0, 2.0], [3.0, 8.0]])
    stats = compare(actual, reference, "populated_kv", POLICY)
    assert stats["max_abs_error"] == 4
    assert stats["rms_error"] == 2
    assert stats["reference_rms"] == math.sqrt(30 / 4)
    assert stats["max_error_coordinate"] == [1, 1]
    assert stats["allclose"] is None and stats["bounds"] is None
    assert not stats["numeric_required"] and not stats["exact_equal"]
    assert stats["status"] == "diagnostic_only" and stats["passed"]
    assert (
        stats["policy_sha256"]
        == hashlib.sha256(
            json.dumps(POLICY, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    )


def test_logit_near_tie_cannot_excuse_actual_top1_difference():
    reference = torch.tensor([1.0, 0.99995, 0.0])
    actual = torch.tensor([1.0, 1.00005, 0.0])
    stats = compare(actual, reference, "logits", POLICY)
    assert stats["allclose"]
    assert not stats["top1_equal"] and not stats["passed"]
    assert stats["actual_top1_ids"] == [1]
    assert stats["reference_top1_ids"] == [0]
    assert stats["actual_top2_margins"] == [float(actual[1]) - 1]
    assert stats["reference_top2_margins"] == [1 - float(reference[1])]


def test_logit_tolerance_is_original_and_reference_relative():
    reference = torch.tensor([1000.0, 1.0, 0.0])
    within = torch.tensor([1000.1, 1.0, 0.0])
    outside = torch.tensor([1000.2, 1.0, 0.0])
    assert compare(within, reference, "logits", POLICY)["passed"]
    failed = compare(outside, reference, "logits", POLICY)
    assert failed["top1_equal"] and not failed["allclose"] and not failed["passed"]
    assert failed["bounds"] == {"atol": 0.001, "rtol": 0.0001}


def test_bf16_logit_policy_and_cross_dtype_sensitivity():
    reference = torch.tensor([5.0, 2.0, 1.0], dtype=torch.bfloat16)
    actual = torch.tensor([5.25, 2.0, 1.0], dtype=torch.bfloat16)
    assert compare(actual, reference, "logits", POLICY)["passed"]
    with pytest.raises(ValueError, match="matching dtypes"):
        compare(actual.float(), reference, "logits", POLICY)
    sensitivity = compare(actual.float(), reference, "bf16_fp32_sensitivity", POLICY)
    assert sensitivity["allclose"] is None and sensitivity["max_abs_error"] == 0.25


def test_nonfinite_failure_retains_finite_json_stats_and_coordinates():
    actual = torch.tensor([[float("nan"), 1.0], [float("inf"), -float("inf")]])
    reference = torch.tensor([[0.0, float("inf")], [2.0, 3.0]])
    with pytest.raises(NonfiniteComparisonError) as captured:
        compare(actual, reference, "layer_output", POLICY)
    stats = captured.value.stats
    assert stats["actual_finite_count"] == 1
    assert stats["reference_finite_count"] == 3
    assert stats["first_nonfinite_actual_coordinate"] == [0, 0]
    assert stats["first_nonfinite_reference_coordinate"] == [0, 1]
    assert stats["max_abs_error"] is None and stats["rms_error"] is None
    assert stats["passed"] is False and stats["status"] == "nonfinite"
    json.dumps(stats, allow_nan=False)


def test_scalar_gate_exact_copy_and_large_chunked_comparison():
    scalar = compare(torch.tensor(0.5), torch.tensor(0.25), "gate_logits", POLICY)
    assert scalar["max_abs_error"] == 0.25 and scalar["max_error_coordinate"] == []
    reference = torch.zeros(4, 24, 4, 16, 128)
    actual = reference.clone()
    assert compare(actual, reference, "populated_kv", POLICY)["exact_equal"]
    actual[3, 23, 3, 15, 127] = 2.0
    stats = compare(actual, reference, "populated_kv", POLICY)
    assert stats["max_error_coordinate"] == [3, 23, 3, 15, 127]
    assert stats["rms_error"] == math.sqrt(4 / actual.numel())


def test_matching_budget_rejected_before_tensor_copy(monkeypatch):
    value = torch.ones(4)

    def forbidden(*args, **kwargs):
        raise AssertionError("comparison allocated before budget validation")

    monkeypatch.setattr(torch.Tensor, "to", forbidden)
    with pytest.raises(BudgetExceededError, match="live tensor matching"):
        compare(value, value, "layer_output", POLICY, matching_bytes=32)


def test_index_record_cap_precedes_tensor_staging(tmp_path, monkeypatch):
    spool = TensorSpool(tmp_path, "oracle", "fixture", SpoolBudget(), "group")
    tensor = torch.ones(1)

    def forbidden(*args, **kwargs):
        raise AssertionError("tensor staging should not occur")

    monkeypatch.setattr(torch.Tensor, "to", forbidden)
    with pytest.raises(BudgetExceededError, match="index record"):
        spool.write("row", tensor, {"large": "x" * 1024})
    spool.close()
    assert spool.data_path.stat().st_size == 0


def test_comparison_envelope_cap_includes_jsonl_newline():
    encoded = comparison_json({"value": "x" * 2035})
    assert len(encoded.encode()) == 2048 and encoded.endswith("\n")
    with pytest.raises(BudgetExceededError, match="2048"):
        comparison_json({"value": "x" * 2036})
    with pytest.raises(ValueError):
        comparison_json({"value": float("inf")})


def test_spool_validates_shape_even_with_rehashed_record(tmp_path):
    spool = TensorSpool(tmp_path, "oracle", "fixture", SpoolBudget(), "group")
    spool.write("row", torch.ones(2), {})
    spool.close()
    document = json.loads(spool.index_path.read_text())
    record = document["records"][0]
    record["shape"] = [3]
    record.pop("record_sha256")
    record["record_sha256"] = hashlib.sha256(
        json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    spool.index_path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="offset, size"):
        TensorSpool.open(tmp_path, "oracle", "fixture")


def test_dump_selection_limits_and_shared_ledger(tmp_path):
    shared = SpoolBudget(total_bytes=28, per_group_bytes=16)
    dump = DiagnosticDump(
        tmp_path,
        ["pre-1", "pre-2"],
        budget=shared,
        per_fixture_bytes=8,
        total_bytes=16,
        max_snapshot_bytes=8,
    )
    assert not dump.observe_failure("nonfinite", finite=False)
    assert dump.observe_failure("first-failure")
    assert dump.observe_failure("second-failure")
    assert not dump.observe_failure("third-failure")
    assert dump.selected_fixture_ids == ["pre-1", "pre-2", "first-failure", "second-failure"]
    value = torch.ones(2)
    assert dump.can_write("pre-1", 8)
    dump.write("pre-1", "fp32/oracle/row", value, {})
    assert not dump.can_write("pre-1", 1)
    with pytest.raises(BudgetExceededError):
        dump.write("pre-1", "bf16/oracle/row", torch.ones(1, dtype=torch.bfloat16), {})
    dump.write("first-failure", "fp32/native/row", value, {})
    assert not dump.can_write("pre-2", 1)
    with pytest.raises(BudgetExceededError):
        dump.write("pre-2", "fp32/native/row", value, {})
    assert shared.written_bytes == dump.written_bytes == 16
    assert dict(dump.fixture_written_bytes) == {"pre-1": 8, "first-failure": 8}
    dump.close()
    with pytest.raises(ValueError, match="closed"):
        dump.write("second-failure", "later", value, {})
    reader = TensorSpool.open(tmp_path, "diagnostic", "pre-1")
    assert torch.equal(reader.read("fp32/oracle/row"), value)


def test_preselected_fixture_identity_is_immutable_and_distinct_from_failures(tmp_path):
    original = ["first", "second"]
    dump = DiagnosticDump(tmp_path, original)
    original.append("changed-input")
    dump.observe_failure("new-failure")
    assert dump.preselected_fixture_ids == ("first", "second")
    assert dump.selected_fixture_ids == ["first", "second", "new-failure"]
    with pytest.raises(AttributeError):
        dump.preselected_fixture_ids = ("replacement", "second")
    dump.close()
