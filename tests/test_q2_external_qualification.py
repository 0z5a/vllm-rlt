"""Final timing disposition cannot promote a contaminated but token-equivalent run."""

import hashlib
import json

import pytest

from vllm_lt.benchmarks.q2_external_qualification import build_qualification


def write(path, value):
    path.write_text(json.dumps(value))
    return {
        "size_bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


@pytest.fixture
def inputs(tmp_path):
    report = tmp_path / "report.json"
    chronology = tmp_path / "chronology.json"
    marker = {"size_bytes": 1, "sha256": "a" * 64}
    base = {
        "schema_version": 1,
        "artifact_type": "q2_external_report",
        "complete": True,
        "evidence_status": "complete",
        "decision": "passed",
        "passed": True,
        "errors": [],
        "missing": [],
        "failures": [],
        "plan_sha256": "b" * 64,
        "equivalence": {"complete": True, "passed": True},
        "runs": [
            {"run_id": name, "token_ids": list(range(64))}
            for name in ("N-feas", "O-feas", "N-warm", "O-warm", "N1", "O1", "O2", "N2")
        ],
        "artifacts": {
            "json_files": {
                f"runs/N2/{name}.json": marker for name in ("started", "result", "completed")
            }
        },
    }
    binding = write(report, base)
    review = {
        "schema_version": 1,
        "artifact_type": "q2_external_manual_timing_disposition",
        "manual_timing_decision": "inconclusive",
        "performance_qualified": False,
        "primary_report_decision": "passed",
        "primary_report_evidence_status": "complete",
        "reason": "CPU preparation overlaps N2 on the same binding.",
        "records": {
            "primary_report": binding,
            "external_N2_start": {**marker, "mtime_ns": 10},
            "quality_plan": {"mtime_ns": 11},
            "quality_probe_log": {"mtime_ns": 12},
            "external_N2_result": {**marker, "mtime_ns": 13},
            "external_N2_completion": marker,
        },
        "shared_affinity": {"cpu_ids": list(range(8)), "numactl_show": {"membind": "0"}},
    }
    write(chronology, review)
    return report, chronology, base, review


def test_known_contamination_supersedes_passed_report_without_rewriting_it(inputs, tmp_path):
    report, chronology, _, _ = inputs
    before = report.read_bytes(), chronology.read_bytes()
    result = build_qualification(report, chronology, tmp_path / "final.json")
    assert result["generation_equivalence"] == "passed"
    assert result["timing_controls"] == "invalid"
    assert result["performance_qualification"] == result["decision"] == "inconclusive"
    assert result["passed"] is False and result["performance_qualified"] is False
    assert result["supersedes"] == hashlib.sha256(before[0]).hexdigest()
    assert (report.read_bytes(), chronology.read_bytes()) == before
    with pytest.raises(FileExistsError):
        build_qualification(report, chronology, report)


@pytest.mark.parametrize("mutation", ["report_hash", "case_hash", "clock", "status", "affinity"])
def test_unbound_or_noncontaminated_chronology_is_rejected(inputs, tmp_path, mutation):
    report, chronology, _, review = inputs
    if mutation == "report_hash":
        review["records"]["primary_report"]["sha256"] = "c" * 64
    elif mutation == "case_hash":
        review["records"]["external_N2_result"]["sha256"] = "c" * 64
    elif mutation == "clock":
        review["records"]["quality_probe_log"]["mtime_ns"] = 14
    elif mutation == "status":
        review["performance_qualified"] = True
    else:
        review["shared_affinity"]["cpu_ids"] = [56]
    write(chronology, review)
    output = tmp_path / "final.json"
    with pytest.raises(ValueError):
        build_qualification(report, chronology, output)
    assert not output.exists()


def test_failed_subchecks_cannot_qualify_generation(inputs, tmp_path):
    report, chronology, base, review = inputs
    base["errors"] = ["invalid generation record"]
    review["records"]["primary_report"] = write(report, base)
    write(chronology, review)
    result = build_qualification(report, chronology, tmp_path / "final.json")
    assert result["generation_equivalence"] == "inconclusive"
    assert result["passed"] is False
