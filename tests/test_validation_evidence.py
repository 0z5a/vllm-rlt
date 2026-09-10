"""Independent evidence-stream scenarios using hand-authored reference artifacts."""

import copy
import hashlib
import json

import pytest
import torch

from vllm_lt.validation import evidence
from vllm_lt.validation.diagnostics import (
    BudgetExceededError,
    DiagnosticDump,
    NonfiniteComparisonError,
    SpoolBudget,
    TensorSpool,
)

POLICY = {
    "policy_id": "q1-original",
    "logits": {
        "float32": {"atol": 0.001, "rtol": 0.0001},
        "bfloat16": {"atol": 0.25, "rtol": 0.02},
    },
}
FIXTURE = "Q1-L256-F0"


def metadata(index=0, *, operation="logits", depth=4):
    return {
        "fixture_id": FIXTURE,
        "operation": operation,
        "positions": [255 + index],
        "depth": depth,
        "layer": None,
        "output_index": index,
        "history_sha256": hashlib.sha256(f"prefix-{index}".encode()).hexdigest(),
    }


def traces():
    return [
        {
            "output_index": i,
            "exit_depth": 4,
            "actual_token_id": 0,
            "emitted_token_id": 0,
            "history_sha256": metadata(i)["history_sha256"],
            "gate_logits": [0.0, 1.0, 2.0, 3.0],
            "gate_probabilities": [0.2] * 4,
            "cumulative_probabilities": [0.2, 0.4, 0.6, 0.8],
        }
        for i in range(9)
    ]


def make_stream(
    tmp_path,
    records,
    *,
    family="official",
    expected=None,
    comparison_id="candidate--fidelity",
    selected=False,
    kind="same_dtype_fidelity",
    dtype="float32",
    shared_dumps=None,
):
    root = tmp_path
    spool = TensorSpool(root / "spools", "reference", FIXTURE, SpoolBudget(), "group")
    for meta, value in records:
        spool.write(evidence.boundary_key(meta), value, meta)
    spool.close()
    reference_case = root / "cases" / "reference"
    reference_case.mkdir(parents=True)
    (reference_case / "result.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "traces": {FIXTURE: traces()},
                "case": {
                    "exit_policy": {
                        "mode": "live" if family == "live_gate" else "fixed",
                        "threshold": 0.7 if family == "live_gate" else 1.0,
                    }
                },
            }
        )
    )
    comparison = {
        "comparison_id": comparison_id,
        "family": family,
        "dtype": dtype,
        "reference_case_id": "reference",
        "candidate_case_id": "candidate",
        "fixture_id": FIXTURE,
        "comparison_kind": kind,
        "expected_boundary_records": len(records) if expected is None else expected,
        "counts_are_upper_bounds": family == "live_gate",
    }
    dumps = (
        shared_dumps
        if shared_dumps is not None
        else DiagnosticDump(
            root / "dumps", [FIXTURE, "other"] if selected else ["other-1", "other-2"]
        )
    )
    stream = evidence.ComparisonStream(root, comparison, POLICY, dumps, "plan-hash")
    return stream, dumps


def read_records(stream):
    stream.file.flush()
    return [json.loads(line) for line in stream.path.read_text().splitlines()]


def test_teacher_forced_matching_and_complete_coverage(tmp_path):
    value = torch.tensor([2.0, 1.0, 0.0])
    records = [(metadata(i), value) for i in range(9)]
    stream, dumps = make_stream(tmp_path, records)
    for meta, actual in records:
        stream.observe(meta, actual)
    summary = stream.finish()
    assert summary["complete"] and summary["count"] == summary["compared_count"] == 9
    assert summary["required_failures"] == 0
    assert all(evidence.required_pass(row) for row in read_records_closed(stream))
    assert [row["observation_index"] for row in read_records_closed(stream)] == list(range(1, 10))
    assert summary["sha256"] == hashlib.sha256(stream.path.read_bytes()).hexdigest()
    dumps.close()


def read_records_closed(stream):
    return [json.loads(line) for line in stream.path.read_text().splitlines()]


@pytest.mark.parametrize("invalid_index", [0, -1, True, 1.5, "2"])
def test_invalid_observation_index_is_rejected_before_boundary_consumption(tmp_path, invalid_index):
    value = torch.tensor([2.0, 1.0])
    stream, dumps = make_stream(tmp_path, [(metadata(), value)])
    with pytest.raises(ValueError, match="observation_index"):
        stream.observe(metadata(), value, observation_index=invalid_index)
    assert not stream.seen and stream.summary["count"] == 0 and stream.size == 0
    stream.observe(metadata(), value, observation_index=7)
    row = read_records(stream)[0]
    assert row["observation_index"] == 7
    assert "observation_index" not in row["metadata"]
    assert row["key"] == evidence.boundary_key(metadata())
    stream.close()
    dumps.close()


def test_interleaved_observation_indices_increase_per_stream_and_can_match_other_stream(tmp_path):
    value = torch.tensor([2.0, 1.0])
    records = [(metadata(i), value) for i in range(3)]
    first, first_dumps = make_stream(tmp_path / "first", records)
    second, second_dumps = make_stream(tmp_path / "second", records)
    first.observe(metadata(0), value, observation_index=4)
    second.observe(metadata(0), value, observation_index=4)
    for invalid_index in (4, 3):
        with pytest.raises(ValueError, match="strictly increasing"):
            first.observe(metadata(1), value, observation_index=invalid_index)
    assert first.summary["count"] == 1 and len(first.seen) == 1
    first.observe(metadata(1), value, observation_index=9)
    second.observe(metadata(1), value, observation_index=9)
    first.observe(metadata(2), value, observation_index=12)
    assert [row["observation_index"] for row in read_records(first)] == [4, 9, 12]
    assert [row["observation_index"] for row in read_records(second)] == [4, 9]
    first.close()
    second.close()
    first_dumps.close()
    second_dumps.close()


@pytest.mark.parametrize(
    "field,value", [("fixture_id", "wrong"), ("output_index", 1), ("history_sha256", "wrong")]
)
def test_correspondence_requires_fixture_output_and_identical_history(tmp_path, field, value):
    meta = metadata()
    stream, dumps = make_stream(tmp_path, [(meta, torch.tensor([2.0, 1.0]))])
    changed = {**meta, field: value}
    with pytest.raises(ValueError):
        stream.observe(changed, torch.tensor([2.0, 1.0]))
    assert stream.summary["count"] == 0
    stream.close()
    dumps.close()


@pytest.mark.parametrize("family", ["official", "live_gate"])
def test_duplicate_missing_and_over_budget_boundaries_are_not_written(tmp_path, family):
    value = torch.tensor([2.0, 1.0])
    stream, dumps = make_stream(tmp_path, [(metadata(), value)], expected=1, family=family)
    stream.observe(metadata(), value)
    with pytest.raises(ValueError, match="duplicate"):
        stream.observe(metadata(), value)
    with pytest.raises(BudgetExceededError, match="upper bound"):
        stream.observe(metadata(1), value)
    assert stream.summary["count"] == 1
    stream.close()
    dumps.close()


def test_missing_reference_and_missing_final_coverage_fail(tmp_path):
    value = torch.tensor([2.0, 1.0])
    stream, dumps = make_stream(tmp_path, [(metadata(i), value) for i in range(9)])
    with pytest.raises(ValueError, match="missing reference"):
        stream.observe({**metadata(), "positions": [999]}, value)
    with pytest.raises(ValueError, match="coverage"):
        stream.finish()
    assert not stream.summary["complete"]
    stream.close()
    dumps.close()


def test_official_stream_cannot_substitute_hidden_rows_for_final_logits(tmp_path):
    meta = metadata(operation="loop_hidden")
    stream, dumps = make_stream(tmp_path, [(meta, torch.ones(2))])
    with pytest.raises(ValueError, match="final four-loop logits"):
        stream.observe(meta, torch.ones(2))
    stream.close()
    dumps.close()


@pytest.mark.parametrize("depth,passes", [(1, True), (2, True), (3, True), (4, False)])
def test_original_top1_exception_ends_at_final_loop(tmp_path, depth, passes):
    meta = {**metadata(depth=depth), "output_index": None}
    reference = torch.tensor([1.0, 0.99995])
    candidate = torch.tensor([1.0, 1.00005])
    stream, dumps = make_stream(tmp_path, [(meta, reference)], family="original")
    stream.observe(meta, candidate)
    record = read_records(stream)[0]
    assert record["stats"]["allclose"] and not record["stats"]["top1_equal"]
    assert not record["stats"]["passed"]
    assert evidence.required_pass(record) is passes
    assert record["top1_required"] is (depth == 4)
    assert stream.finish()["required_failures"] == int(not passes)
    dumps.close()


def test_live_exit_divergence_with_same_tokens_makes_later_boundaries_incomparable(tmp_path):
    value = torch.tensor([2.0, 1.0])
    stream, dumps = make_stream(
        tmp_path, [(metadata(i), value) for i in range(9)], family="live_gate"
    )
    for index in range(9):
        stream.observe(metadata(index, depth=2 if index == 3 else 4), value)
    actual = traces()
    actual[3]["exit_depth"] = 2
    for field in ("gate_logits", "gate_probabilities", "cumulative_probabilities"):
        actual[3][field] = actual[3][field][:2]
    actual[3]["cumulative_probabilities"] = [0.2, 0.8]
    summary = stream.finish(actual)
    records = read_records_closed(stream)
    assert [row["status"] for row in records] == ["compared"] * 3 + ["incomparable"] * 6
    assert summary["behavior_failures"] == 1
    assert summary["first_behavior_failure"]["output_index"] == 3
    assert summary["behavior"][3]["status"] == "diverged"
    assert all(row["status"] == "incomparable" for row in summary["behavior"][4:])
    gate = summary["behavior"][3]["gate_comparison"]
    assert gate["diagnostic_only"] and gate["threshold"] == 0.7
    assert gate["per_depth"][1]["cdf_delta"] == pytest.approx(0.4)
    assert gate["per_depth"][1]["actual_threshold_distance"] == pytest.approx(0.1)
    assert gate["per_depth"][1]["reference_threshold_distance"] == pytest.approx(-0.3)
    dumps.close()


def test_live_first_logit_divergence_is_compared_and_persisted(tmp_path):
    expected = torch.tensor([1.0, 0.99995])
    stream, dumps = make_stream(
        tmp_path, [(metadata(i), expected) for i in range(2)], family="live_gate"
    )
    stream.observe(metadata(0), torch.tensor([1.0, 1.00005]))
    stream.observe(metadata(1), expected)
    records = read_records(stream)
    assert records[0]["status"] == "compared" and not evidence.required_pass(records[0])
    assert records[1]["status"] == "incomparable"
    assert stream.summary["first_failure"] == records[0]
    stream.close()
    dumps.close()


def test_unexplained_live_history_mismatch_does_not_hide_a_failure(tmp_path):
    expected = traces()
    candidate = copy.deepcopy(expected)
    candidate[3]["history_sha256"] = "different-with-no-earlier-token-or-exit-change"
    with pytest.raises(ValueError, match="before a recorded"):
        evidence.compare_behavior(candidate, expected, live=True)
    stream, dumps = make_stream(tmp_path, [(metadata(), torch.ones(2))], family="live_gate")
    with pytest.raises(ValueError, match="supplied input history"):
        stream.observe({**metadata(), "history_sha256": "other"}, torch.ones(2))
    stream.close()
    dumps.close()


def test_forced_route_gate_deltas_remain_diagnostic():
    expected = traces()
    candidate = copy.deepcopy(expected)
    candidate[2]["gate_logits"][1] += 0.25
    candidate[2]["gate_probabilities"][1] += 0.125
    candidate[2]["cumulative_probabilities"][1] += 0.125
    records = evidence.compare_behavior(candidate, expected, live=False, route_control="forced")
    row = records[2]
    assert row["status"] == "matched" and row["route_control"] == "forced"
    gate = row["gate_comparison"]
    assert gate["diagnostic_only"] and gate["threshold"] is None
    assert gate["per_depth"][1]["logit_delta"] == 0.25
    assert gate["per_depth"][1]["probability_delta"] == 0.125
    assert gate["per_depth"][1]["actual_threshold_distance"] is None


@pytest.mark.parametrize("incomparable", [False, True])
def test_nonfinite_boundary_record_survives_before_stop(tmp_path, incomparable):
    expected = torch.tensor([2.0, 1.0])
    stream, dumps = make_stream(
        tmp_path, [(metadata(1), expected)], family="live_gate" if incomparable else "official"
    )
    meta = metadata(1, depth=2 if incomparable else 4)
    with pytest.raises(NonfiniteComparisonError):
        stream.observe(meta, torch.tensor([float("nan"), 1.0]))
    record = read_records(stream)[0]
    assert not record["stats"]["all_finite"] and not evidence.required_pass(record)
    assert record["stats"]["actual_finite_count"] == 1
    assert stream.summary["first_failure"] == record
    assert not stream.summary["complete"]
    if incomparable:
        assert record["stats"]["reference_available"] is False
    stream.close()
    dumps.close()


def test_representative_ouro_kv_dump_and_json_records_fit_frozen_caps(tmp_path):
    meta = {
        **metadata(8, operation="populated_kv", depth=None),
        "positions": [260, 261, 262, 263],
        "component": "keys",
        "axes": ["depth", "layer", "position", "kv_head", "head_dim"],
        "depth_range": [1, 4],
        "layer_range": [0, 23],
    }
    values = torch.zeros(4, 24, 4, 16, 128, dtype=torch.bfloat16)
    comparison_id = "main-bfloat16-native-triton-no_refill-" + "g" * 160
    stream, dumps = make_stream(
        tmp_path,
        [(meta, values)],
        family="main",
        comparison_id=comparison_id,
        selected=True,
        dtype="bfloat16",
    )
    stream.observe(meta, values)
    stream.close()
    dumps.close()
    assert all(len(line) + 1 <= 2048 for line in stream.path.read_bytes().splitlines())
    dump = TensorSpool.open(tmp_path / "dumps", "diagnostic", FIXTURE)
    assert len(dump.index) == 2
    for key, entry in dump.index.items():
        assert len(key) == 64
        assert len(json.dumps(entry, sort_keys=True, separators=(",", ":")).encode()) + 2 <= 1024
        assert (
            entry["metadata"]["comparison_sha256"]
            == hashlib.sha256(comparison_id.encode()).hexdigest()
        )
        assert torch.equal(dump.read(key), values)


def test_pair_budget_accounts_for_contiguous_copy_before_reference_read(tmp_path, monkeypatch):
    meta = metadata(operation="layer_output")
    value = torch.ones(2, 2).T
    stream, dumps = make_stream(tmp_path, [(meta, value)], family="main")
    monkeypatch.setattr(evidence, "MATCHING_BYTES", 8 * 1024**2 + 40)
    monkeypatch.setattr(
        stream.reference, "read", lambda key: pytest.fail("read preceded pair budget check")
    )
    with pytest.raises(BudgetExceededError, match="Comparison/dump pair"):
        stream.observe(meta, value)
    assert not stream.summary["count"]
    stream.close()
    dumps.close()


def test_live_upper_bound_does_not_allow_missing_final_prediction_coverage(tmp_path):
    value = torch.tensor([2.0, 1.0])
    stream, dumps = make_stream(tmp_path, [(metadata(), value)], family="live_gate", expected=100)
    stream.observe(metadata(), value)
    with pytest.raises(ValueError, match="nine output indices"):
        stream.finish(traces())
    assert not stream.summary["complete"]
    stream.close()
    dumps.close()


@pytest.mark.parametrize("divergence_index,kv_status", [(7, "incomparable"), (8, "compared")])
def test_last_unused_prediction_does_not_invalidate_computed_kv(
    tmp_path, divergence_index, kv_status
):
    value = torch.tensor([2.0, 1.0])
    kv_meta = {**metadata(8, operation="populated_kv", depth=None), "component": "keys"}
    kv = torch.zeros(4, 2, 1, 2, 8)
    records = [(metadata(i), value) for i in range(9)] + [(kv_meta, kv)]
    stream, dumps = make_stream(tmp_path, records, family="live_gate")
    actual = traces()
    actual[divergence_index]["actual_token_id"] = 1
    actual[divergence_index]["emitted_token_id"] = 1
    for index in range(9):
        meta = metadata(index)
        if index > divergence_index:
            meta["history_sha256"] = "changed-prefix-after-consumed-divergent-output"
            actual[index]["history_sha256"] = meta["history_sha256"]
        stream.observe(meta, value.flip(0) if index == divergence_index else value)
    if divergence_index < 8:
        kv_meta = {**kv_meta, "history_sha256": actual[8]["history_sha256"]}
    stream.observe(kv_meta, kv)
    summary = stream.finish(actual)
    assert read_records_closed(stream)[-1]["status"] == kv_status
    assert summary["behavior_failures"] == 1
    assert summary["first_behavior_failure"]["output_index"] == divergence_index
    dumps.close()


def test_kv_axis_metadata_must_match_before_elementwise_comparison(tmp_path):
    meta = {
        **metadata(8, operation="populated_kv", depth=None),
        "component": "keys",
        "axes": ["depth", "layer", "position", "kv_head", "head_dim"],
        "depth_range": [1, 4],
        "layer_range": [0, 1],
    }
    value = torch.zeros(4, 2, 1, 2, 8)
    stream, dumps = make_stream(tmp_path, [(meta, value)], family="main")
    with pytest.raises(ValueError, match="layer_range"):
        stream.observe({**meta, "layer_range": [1, 2]}, value)
    assert stream.summary["count"] == 0
    stream.close()
    dumps.close()


def test_gate_evidence_must_be_finite_even_on_later_incomparable_outputs():
    expected = traces()
    actual = copy.deepcopy(expected)
    actual[1]["actual_token_id"] = 1
    actual[2]["gate_logits"][0] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        evidence.compare_behavior(actual, expected, live=True)


def test_preselected_quota_survives_fp32_failure_and_cross_dtype_sensitivity(tmp_path):
    dumps = DiagnosticDump(tmp_path / "dumps", [FIXTURE, "other"])
    reference = torch.tensor([2.0, 1.0])
    fp32, _ = make_stream(
        tmp_path / "fp32",
        [(metadata(), reference)],
        family="main",
        comparison_id="fp32-fidelity",
        shared_dumps=dumps,
    )
    fp32.observe(metadata(), reference.flip(0))
    assert fp32.summary["required_failures"] == 1
    assert dumps.written_bytes == 0 and not dumps.failure_fixture_ids
    fp32.close()
    sensitivity, _ = make_stream(
        tmp_path / "sensitivity",
        [(metadata(), reference)],
        family="main",
        comparison_id="bf16-sensitivity",
        kind="bf16_fp32_sensitivity",
        dtype="bfloat16",
        shared_dumps=dumps,
    )
    sensitivity.observe(metadata(), reference.to(torch.bfloat16))
    assert dumps.written_bytes == 0
    sensitivity.close()
    meta = metadata(operation="layer_output")
    bf16_reference = torch.tensor([1.0, 0.1, 0.01, 0.001], dtype=torch.bfloat16)
    bf16_actual = bf16_reference.flip(0)
    fidelity, _ = make_stream(
        tmp_path / "bf16",
        [(meta, bf16_reference)],
        family="main",
        comparison_id="bf16-fidelity",
        dtype="bfloat16",
        shared_dumps=dumps,
    )
    fidelity.observe(meta, bf16_actual)
    fidelity.close()
    assert dumps.written_bytes == 2 * bf16_actual.numel() * bf16_actual.element_size()
    dumps.close()
    reader = TensorSpool.open(tmp_path / "dumps", "diagnostic", FIXTURE)
    assert len(reader.index) == 2
    for key, entry in reader.index.items():
        expected = bf16_actual if entry["metadata"]["side"] == "actual" else bf16_reference
        actual = reader.read(key)
        assert actual.dtype == torch.bfloat16
        assert torch.equal(actual.view(torch.uint8), expected.contiguous().view(torch.uint8))


def test_failure_selected_fixture_captures_from_failure_forward_across_dtypes(tmp_path):
    dumps = DiagnosticDump(tmp_path / "dumps", ["pre-1", "pre-2"])
    value = torch.tensor([2.0, 1.0])
    stream, _ = make_stream(
        tmp_path / "fp32",
        [(metadata(i), value) for i in range(3)],
        family="main",
        comparison_id="fp32-failure",
        shared_dumps=dumps,
    )
    stream.observe(metadata(0), value)
    assert dumps.written_bytes == 0
    stream.observe(metadata(1), value.flip(0))
    assert dumps.written_bytes == 16
    stream.observe(metadata(2), value)
    assert dumps.written_bytes == 32
    assert dumps.failure_fixture_ids == [FIXTURE]
    assert FIXTURE not in dumps.preselected_fixture_ids
    stream.close()
    bf16 = value.to(torch.bfloat16)
    later, _ = make_stream(
        tmp_path / "bf16",
        [(metadata(), bf16)],
        family="main",
        comparison_id="bf16-later",
        dtype="bfloat16",
        shared_dumps=dumps,
    )
    later.observe(metadata(), bf16)
    later.close()
    assert dumps.written_bytes == 40
    dumps.close()
    reader = TensorSpool.open(tmp_path / "dumps", "diagnostic", FIXTURE)
    assert len(reader.index) == 6
    fp32_indices = [
        entry["metadata"]["output_index"]
        for entry in reader.index.values()
        if entry["dtype"] == "float32"
    ]
    assert sorted(fp32_indices) == [1, 1, 2, 2]
