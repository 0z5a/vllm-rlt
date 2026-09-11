"""Versioned CPU-only timing invalidation; preserve the original Q2 report.

This command publishes a reviewed control violation, not a new timing experiment.
It cannot qualify timing: passing generation and a historical passed subreport
remain separate from the final inconclusive performance decision.
Run this file directly to avoid importing the package's Torch entry points.
"""

import argparse
import hashlib
import json
from pathlib import Path


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _pairs(items):
    result = {}
    for key, value in items:
        _require(key not in result, f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _read(path):
    path = Path(path)
    _require(path.is_file() and not path.is_symlink(), "input must be an ordinary file")
    _require(path.stat().st_size <= 4 * 1024**2, "input exceeds 4 MiB")
    raw = path.read_bytes()
    value = json.loads(raw, object_pairs_hook=_pairs)
    _require(isinstance(value, dict), "input must be a JSON object")
    return value, {"size_bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def build_qualification(report_path, chronology_path, output_path):
    """Bind the preserved report to its reviewed CPU-overlap evidence, fail closed."""
    output = Path(output_path)
    if output.exists():
        raise FileExistsError(output)
    report, report_file = _read(report_path)
    chronology, chronology_file = _read(chronology_path)
    _require(
        report.get("artifact_type") == "q2_external_report" and report.get("schema_version") == 1,
        "unsupported original report",
    )
    _require(
        chronology.get("artifact_type") == "q2_external_manual_timing_disposition"
        and chronology.get("schema_version") == 1,
        "unsupported chronology record",
    )
    _require(
        chronology.get("manual_timing_decision") == "inconclusive"
        and chronology.get("performance_qualified") is False,
        "chronology must explicitly invalidate timing qualification",
    )
    records = chronology["records"]
    _require(
        {k: records["primary_report"][k] for k in report_file} == report_file,
        "chronology is bound to a different original report",
    )
    _require(
        chronology["primary_report_decision"] == report["decision"]
        and chronology["primary_report_evidence_status"] == report["evidence_status"],
        "chronology differs from the preserved report decision",
    )
    for label, relative in (
        ("external_N2_start", "runs/N2/started.json"),
        ("external_N2_result", "runs/N2/result.json"),
        ("external_N2_completion", "runs/N2/completed.json"),
    ):
        _require(
            {k: records[label][k] for k in ("size_bytes", "sha256")}
            == report["artifacts"]["json_files"][relative],
            f"chronology differs from recorded {relative}",
        )
    clocks = [
        records[key]["mtime_ns"]
        for key in ("external_N2_start", "quality_plan", "quality_probe_log", "external_N2_result")
    ]
    _require(all(type(value) is int and value >= 0 for value in clocks), "invalid chronology clock")
    _require(clocks == sorted(clocks), "chronology does not establish activity during N2")
    affinity = chronology["shared_affinity"]
    _require(affinity["cpu_ids"] == list(range(8)), "unexpected reviewed shared CPU binding")
    _require(affinity["numactl_show"]["membind"] == "0", "unexpected shared memory binding")
    _require(isinstance(chronology.get("reason"), str) and chronology["reason"], "missing reason")
    clean = (
        report.get("complete") is True
        and report.get("evidence_status") == "complete"
        and report.get("decision") == "passed"
        and report.get("passed") is True
        and all(report.get(key) == [] for key in ("errors", "missing", "failures"))
    )
    generation = "inconclusive"
    if clean and report.get("equivalence") == {"complete": True, "passed": True}:
        runs = report.get("runs", [])
        ids = {row["run_id"] for row in runs}
        expected = {"N-feas", "O-feas", "N-warm", "O-warm", "N1", "O1", "O2", "N2"}
        _require(len(runs) == 8 and ids == expected, "generation coverage differs")
        tokens = runs[0]["token_ids"]
        _require(
            len(tokens) == 64
            and all(type(token) is int and token >= 0 for token in tokens)
            and all(row["token_ids"] == tokens for row in runs),
            "actual generation token equivalence differs",
        )
        generation = "passed"
    result = {
        "schema_version": 1,
        "artifact_type": "q2_external_final_qualification",
        "decision": "inconclusive",
        "passed": False,
        "generation_equivalence": generation,
        "timing_controls": "invalid",
        "performance_qualification": "inconclusive",
        "performance_qualified": False,
        "supersedes": report_file["sha256"],
        "supersedes_scope": "Final timing qualification only; original report bytes are preserved.",
        "plan_sha256": report["plan_sha256"],
        "original_report": report_file,
        "chronology_evidence": chronology_file,
        "reason": chronology["reason"],
        "affected_run_ids": ["N2"],
        "issue_closure": False,
        "gpu_runs_added": 0,
        "limitations": [
            "File mtimes establish overlapping case activity, not a causal slowdown amount.",
            "This consumes the frozen report and reviewed chronology; it is not a raw auditor.",
            "Token equivalence or missing control errors cannot establish valid timing.",
        ],
    }
    # Exclusive creation prevents replacing either historical evidence or another decision.
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--chronology", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = build_qualification(args.report, args.chronology, args.output)
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "generation_equivalence",
                    "timing_controls",
                    "performance_qualification",
                    "passed",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
