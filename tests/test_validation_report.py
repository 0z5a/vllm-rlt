"""Offline audits of actual tiny CPU driver artifacts, including preserved raw tensors.

The tiny plans deliberately bypass only the production 168-case plan validator;
that exact schema is covered separately. All writer, comparison, spool, trace,
summary and report code below runs unchanged, without CUDA or checkpoint loading.
"""

import copy
import hashlib
import json
import math
import shutil
import time
from pathlib import Path

import pytest
import torch

from vllm_lt.models import OuroConfig, OuroForCausalLM
from vllm_lt.validation import report, runner
from vllm_lt.validation.common import (
    digest,
    read_json,
    write_json,
)
from vllm_lt.validation.diagnostics import DiagnosticDump, SpoolBudget, compare
from vllm_lt.validation.schema import (
    _comparisons,
    _resolve_cases,
)


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("offline reporting must not discover CUDA or load checkpoint weights")

    for name in ("is_available", "device_count", "current_device", "_lazy_init"):
        monkeypatch.setattr(torch.cuda, name, forbidden)
    monkeypatch.setattr(OuroForCausalLM, "from_pretrained", forbidden)

    def validate_tiny_plan(plan):
        assert plan["plan_sha256"] == digest(
            {key: value for key, value in plan.items() if key != "plan_sha256"}
        )

    monkeypatch.setattr(report, "validate_plan_integrity", validate_tiny_plan)


def _tiny_plan(dtype, *, family="main", preselected=False):
    root = Path(__file__).resolve().parents[1]
    suite = read_json(root / "benchmarks/fixtures/ouro-q1.json")
    contract = read_json(root / "benchmarks/fixtures/ouro-q1-contract.json")
    config = OuroConfig.tiny()
    for fixture in suite["fixtures"] + suite["feasibility_fixtures"]:
        fixture["prompt_token_ids"] = [
            token % config.vocab_size for token in fixture["prompt_token_ids"][:3]
        ]
        fixture["continuation_input_ids"] = [
            token % config.vocab_size for token in fixture["continuation_input_ids"]
        ]
    for prompt in suite["original_reproduction"]["prompt_token_ids"]:
        prompt[:] = [token % config.vocab_size for token in prompt]
    contract["engine"]["cache"]["block_size"] = 2
    contract["engine"]["scheduler"]["max_num_batched_tokens"] = 3
    order, fixtures = _resolve_cases(suite, contract, config)
    comparisons = _comparisons(order, fixtures, contract, config)
    fixture_id = "Q1-L16-F2" if preselected else "Q1-L16-F0"
    candidate = next(
        case
        for case in order
        if case["implementation"] == "native"
        and case["family"] == family
        and case["dtype"] == dtype
        and case["backend"] == "torch"
        and case["fixture_ids"] == [fixture_id]
    )
    reference = next(
        case
        for case in order
        if case["implementation"] == "oracle"
        and case["family"] == family
        and case["dtype"] == dtype
        and case["fixture_ids"] == [fixture_id]
    )
    contract["dtypes"] = [dtype]
    contract["controls"]["gpu_ids"] = [0]  # Fictitious recorded CPU-test environment only.
    official = {
        "dependencies": {
            "transformers": "4.55.0",
            "torch": "cpu-test",
            "safetensors": "cpu-test",
            "huggingface-hub": "cpu-test",
        },
        "optional_kernels_present": False,
    }
    plan = {
        "suite": suite,
        "contract": contract,
        "model_config": config.to_dict(),
        "execution_order": [reference, candidate],
        "comparison_order": [
            row for row in comparisons if row["candidate_case_id"] == candidate["case_id"]
        ],
        "dependencies": {"python": "cpu-test", "torch_cuda_build": None, "official": official},
    }
    plan["plan_sha256"] = digest(plan)
    return plan, config


def _generate(
    output, *, dtype="float32", family="main", preselected=False, fail=False, shared=False
):
    output.mkdir(parents=True)
    plan, config = _tiny_plan(dtype, family=family, preselected=preselected)
    if shared:
        extra = copy.deepcopy(plan["comparison_order"][0])
        extra["comparison_id"] += "-second-reference-view"
        plan["comparison_order"].append(extra)
        plan["plan_sha256"] = digest(
            {key: value for key, value in plan.items() if key != "plan_sha256"}
        )
    torch.manual_seed(41)
    model = OuroForCausalLM(config).to(dtype=getattr(torch, dtype))
    with torch.no_grad():
        model.model.early_exit_gate.weight.zero_()
        model.model.early_exit_gate.bias.fill_(math.log(0.4 / 0.6))
    budget = SpoolBudget()
    dumps = DiagnosticDump(
        output / "dumps",
        plan["contract"]["diagnostics"]["preselected_dump_fixture_ids"],
        budget=budget,
    )
    write_json(output / "plan.json", plan)
    for index, case in enumerate(plan["execution_order"]):
        if fail and index == 1:
            with torch.no_grad():
                model.lm_head.weight.mul_(30)  # Deliberate finite negative fixture.
        result = runner.execute_case(
            model, plan, case, output, budget, dumps, time.monotonic() + 60
        )
        assert result["status"] == "complete"
    dumps.close()
    controls = plan["contract"]["controls"]
    manifest = {
        "schema_version": 1,
        "artifact_type": "validation_manifest",
        "plan_sha256": plan["plan_sha256"],
        "status": "complete",
        "failures": [],
        "completed_cases": [case["case_id"] for case in plan["execution_order"]],
        "official_provenance": plan["dependencies"]["official"],
        "environment": {
            "cuda_visible_devices": "0",
            "logical_device": "cuda:0",
            "account": "cpu-test",
            "scheduler": [{"gpu_id": 0, "type": "RUN", "user": "cpu-test"}],
            "actual_torch_threads": {
                "intraop": controls["cpu_threads"],
                "interop": controls["interop_threads"],
            },
            "arithmetic": plan["contract"]["arithmetic"],
            "cpu_affinity": [0],
            "numa_status": ["Cpus_allowed_list: 0", "Mems_allowed_list: 0"],
            "python": "cpu-test",
            "torch_cuda_version": None,
            "software": {name: "cpu-test" for name in ("torch", "safetensors", "huggingface-hub")},
        },
        "teardown_after_workspace_release": {"allocated_bytes": 0, "reserved_bytes": 0},
        "elapsed_s": 1,
        "tensor_written_bytes": budget.written_bytes,
        "artifact_disk_bytes": 0,
        "diagnostic_dumps": {
            "selected_fixture_ids": dumps.selected_fixture_ids,
            "written_bytes": dumps.written_bytes,
            "fixture_written_bytes": dict(dumps.fixture_written_bytes),
        },
    }
    write_json(output / "manifest.json", manifest)
    return output


@pytest.fixture(scope="module")
def evidence_sources(tmp_path_factory):
    # Generation itself uses CPU tensors only; every report test additionally forbids CUDA APIs.
    base = tmp_path_factory.mktemp("report-real-cpu-writers")
    return {
        "pass": _generate(base / "pass"),
        "shared": _generate(base / "shared", shared=True),
        "live": _generate(base / "live", family="live_gate"),
        "bf16": _generate(base / "bf16", dtype="bfloat16", preselected=True),
        "fail": _generate(base / "fail", fail=True),
    }


@pytest.fixture
def evidence(tmp_path, evidence_sources):
    def copy_artifacts(kind="pass"):
        output = tmp_path / kind
        shutil.copytree(evidence_sources[kind], output)
        return output

    return copy_artifacts


def _stream(output):
    path = next((output / "comparisons").glob("*.jsonl"))
    return path, [json.loads(line) for line in path.read_text().splitlines()]


def _rewrite_stream(path, records):
    path.write_text("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in records))
    summary_path = path.with_suffix(".summary.json")
    summary = read_json(summary_path)
    summary.update(
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(), size_bytes=path.stat().st_size
    )
    write_json(summary_path, summary)


@pytest.mark.parametrize("kind", ["pass", "live", "bf16"])
def test_real_driver_evidence_passes_offline_audit_and_keeps_manual_gate(evidence, kind):
    result = report.build_report(evidence(kind))
    assert result["errors"] == []
    assert result["evidence_status"] == "complete"
    assert result["runtime_qualification"] == "passed"
    assert result["milestone_status"] == "manual_diagnosis_required"
    assert len(result["raw_evidence"]["retained_references"]) == 1
    if kind == "bf16":
        assert result["raw_evidence"]["diagnostic_dumps"][0]["records"] > 0


def test_finite_negative_is_complete_evidence_failed_qualification_with_paired_dump(evidence):
    result = report.build_report(evidence("fail"))
    assert result["errors"] == []
    assert result["evidence_status"] == "complete"
    assert result["runtime_qualification"] == "failed"
    assert result["runtime_by_dtype"]["float32"]["numerical_required_failures"] > 0
    assert result["raw_evidence"]["diagnostic_dumps"][0]["records"] > 0


@pytest.mark.parametrize("artifact", ["comparison", "reference_bin", "reference_index"])
def test_missing_required_artifact_cannot_qualify(evidence, artifact):
    output = evidence()
    patterns = {
        "comparison": "comparisons/*.jsonl",
        "reference_bin": "spools/*/*.bin",
        "reference_index": "spools/*/*.index.json",
    }
    next(output.glob(patterns[artifact])).unlink()
    result = report.build_report(output)
    assert result["runtime_qualification"] == "inconclusive"
    assert result["evidence_status"] != "complete"


@pytest.mark.parametrize("kind", ["pass", "bf16"])
def test_raw_payload_corruption_is_detected_without_loading_tensors(evidence, kind):
    output = evidence(kind)
    path = next(output.glob("dumps/*/*.bin" if kind == "bf16" else "spools/*/*.bin"))
    with path.open("r+b") as stream:
        byte = stream.read(1)
        stream.seek(0)
        stream.write(bytes([byte[0] ^ 1]))
    result = report.build_report(output)
    assert result["runtime_qualification"] == "inconclusive"
    assert any("payload SHA256" in row["message"] for row in result["errors"])


@pytest.mark.parametrize("tamper", ["duplicate", "numeric_flag", "observation_order", "top1_flag"])
def test_rehashed_comparison_still_requires_independent_evidence_consistency(evidence, tamper):
    output = evidence()
    path, records = _stream(output)
    if tamper == "duplicate":
        records.append(copy.deepcopy(records[-1]))
    elif tamper == "numeric_flag":
        records[0]["stats"]["numeric_required"] = True
    elif tamper == "observation_order":
        records[1]["observation_index"] = records[0]["observation_index"]
    else:
        row = next(row for row in records if row["metadata"]["operation"] == "logits")
        row["stats"]["actual_top1_ids"][0] = (row["stats"]["actual_top1_ids"][0] + 1) % 64
    _rewrite_stream(path, records)
    result = report.build_report(output)
    assert result["evidence_status"] == "invalid"
    assert result["runtime_qualification"] == "inconclusive"


def test_summary_pass_flag_cannot_hide_real_finite_failure(evidence):
    output = evidence("fail")
    path = next(output.glob("comparisons/*.summary.json"))
    summary = read_json(path)
    summary["required_failures"] = 0
    write_json(path, summary)
    result = report.build_report(output)
    assert any("summary required_failures" in row["message"] for row in result["errors"])


def test_deleted_dumps_remain_required_even_if_manifest_counters_are_rewritten(evidence):
    output = evidence("bf16")
    shutil.rmtree(output / "dumps")
    manifest = read_json(output / "manifest.json")
    manifest["tensor_written_bytes"] -= manifest["diagnostic_dumps"]["written_bytes"]
    manifest["diagnostic_dumps"].update(written_bytes=0, fixture_written_bytes={})
    write_json(output / "manifest.json", manifest)
    result = report.build_report(output)
    assert any("required paired diagnostic capture" in row["message"] for row in result["errors"])


def test_original_early_loop_top1_difference_is_diagnostic_but_depth_four_fails(evidence):
    plan = read_json(evidence() / "plan.json")
    policy = plan["contract"]["comparison_policy"]
    actual, reference = torch.tensor([1.0, 1.0001]), torch.tensor([1.0001, 1.0])
    stats = compare(actual, reference, "logits", policy)
    assert stats["allclose"] and not stats["passed"]
    comparison = {
        "family": "original",
        "dtype": "float32",
        "comparison_kind": "same_dtype_fidelity",
    }
    record = {
        "stats": stats,
        "metadata": {"operation": "logits", "depth": 1},
        "top1_required": False,
    }
    assert report._audit_stats(record, comparison, [2], policy)
    record["metadata"]["depth"] = 4
    record["top1_required"] = True
    assert not report._audit_stats(record, comparison, [2], policy)


def test_live_last_unconsumed_token_keeps_kv_comparable_but_exit_change_does_not():
    reference = [{"actual_token_id": 1, "exit_depth": 4} for _ in range(9)]
    actual = copy.deepcopy(reference)
    actual[8]["actual_token_id"] = 2
    metadata = {"operation": "populated_kv", "output_index": 8}
    assert not report._must_be_incomparable(metadata, "key", {"key": True}, actual, reference)
    actual[8]["exit_depth"] = 3
    assert report._must_be_incomparable(metadata, "key", {"key": True}, actual, reference)
    actual = copy.deepcopy(reference)
    actual[2]["exit_depth"] = 3
    metadata = {"operation": "attention_input", "output_index": 3}
    assert report._must_be_incomparable(metadata, "key", {"key": True}, actual, reference)


def test_write_report_is_offline_and_records_limits(evidence):
    output = evidence()
    result = report.write_report(output)
    assert read_json(output / "qualification.json") == result
    assert "manual diagnosis" in (output / "qualification.md").read_text()
    assert any("does not recompute elementwise" in line for line in result["limitations"])


def test_same_global_observation_can_be_shared_across_comparison_streams(evidence):
    result = report.build_report(evidence("shared"))
    assert result["errors"] == []
    assert result["runtime_qualification"] == "passed"
    assert result["counts"]["verified_comparisons"] == 2


def test_cross_stream_global_observation_conflict_is_rejected(evidence):
    output = evidence("shared")
    path, records = _stream(output)
    for row in records:
        row["observation_index"] += 10000  # Still strictly increasing within this stream.
    _rewrite_stream(path, records)
    result = report.build_report(output)
    assert any("different global observation indices" in row["message"] for row in result["errors"])
    assert result["runtime_qualification"] == "inconclusive"


def test_observation_union_must_cover_exactly_one_through_case_count(evidence):
    output = evidence()
    path, records = _stream(output)
    for row in records:
        row["observation_index"] += 1
    _rewrite_stream(path, records)
    result = report.build_report(output)
    assert any("global observation coverage" in row["message"] for row in result["errors"])


def test_missing_dump_half_is_rejected_even_with_updated_byte_accounting(evidence):
    output = evidence("bf16")
    path = next(output.glob("dumps/*/*.index.json"))
    index = read_json(path)
    removed = index["records"].pop()
    assert removed["metadata"]["side"] == "reference"
    index["size_bytes"] -= removed["size_bytes"]
    write_json(path, index)
    with path.with_name(path.name.removesuffix(".index.json") + ".bin").open("r+b") as stream:
        stream.truncate(index["size_bytes"])
    manifest = read_json(output / "manifest.json")
    manifest["tensor_written_bytes"] -= removed["size_bytes"]
    manifest["diagnostic_dumps"]["written_bytes"] -= removed["size_bytes"]
    manifest["diagnostic_dumps"]["fixture_written_bytes"][index["fixture_id"]] -= removed[
        "size_bytes"
    ]
    write_json(output / "manifest.json", manifest)
    result = report.build_report(output)
    assert any("missing final reference" in row["message"] for row in result["errors"])


def test_retained_index_metadata_hash_is_verified(evidence):
    output = evidence()
    path = next(output.glob("spools/*/*.index.json"))
    index = read_json(path)
    index["records"][0]["metadata"]["history_sha256"] = "f" * 64
    write_json(path, index)
    result = report.build_report(output)
    assert any("record SHA-256 mismatch" in row["message"] for row in result["errors"])
