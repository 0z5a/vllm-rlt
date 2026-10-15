"""Keep every original W5 case and pair only complete same-protocol raw jobs."""

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import TypedDict, cast

from prepare_w5_workloads import CASES


class RawFile(TypedDict):
    path: str
    sha256: str


class Audit(TypedDict):
    status: str
    job_id: str
    protocol_sha256: str
    job_sha256: str
    raw_files: list[RawFile]
    raw_trials: int
    requests_per_trial: int
    same_policy_signature_sha256: str
    measured_elapsed_s: list[float]
    measured_tps: list[float]
    actual_A_peak: int
    actual_S_peak: int
    target_A_reached: bool


class Comparison(TypedDict):
    matched_pairs: int
    ratios: list[float]
    mean_ratio: float
    median_ratio: float
    minimum_ratio: float
    maximum_ratio: float


ARMS = ("A_P4D4_off", "C_P4D3_off", "D_P4D3_on")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def matched_ratios(baseline: list[float], candidate: list[float]) -> Comparison:
    assert len(baseline) == len(candidate) == 10
    assert all(math.isfinite(x) and x > 0 for x in baseline + candidate)
    ratios = [a / b for a, b in zip(baseline, candidate, strict=True)]
    return {
        "matched_pairs": 10,
        "ratios": ratios,
        "mean_ratio": statistics.mean(ratios),
        "median_ratio": statistics.median(ratios),
        "minimum_ratio": min(ratios),
        "maximum_ratio": max(ratios),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audits", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert args.audits.is_dir()
    expected = {
        f"{c.case_id}-fresh{f}-{a}" for c in CASES for f in (0, 1) for a in ARMS
    }
    audits: dict[str, Audit] = {}
    provenance: list[dict[str, str]] = []
    for path in sorted(args.audits.glob("*.json")):
        audit = cast(Audit, json.loads(path.read_text()))
        if audit["job_id"] not in expected:
            raise ValueError(
                "Unexpected job cannot count toward the original seven W5 cases"
            )
        assert audit["job_id"] not in audits
        assert audit["status"] == "PASS_COMPLETE_SINGLE_W5_JOB_RAW_ONLY"
        assert audit["raw_trials"] == len(audit["raw_files"]) == 10
        assert len(audit["measured_elapsed_s"]) == len(audit["measured_tps"]) == 5
        measured_elapsed, measured_tps = [], []
        for index, row in enumerate(audit["raw_files"]):
            raw = Path(row["path"])
            assert digest(raw) == row["sha256"]
            trial = json.loads(raw.read_text())
            assert trial["index"] == index
            assert trial["phase"] == ("warmup" if index < 5 else "measured")
            if index >= 5:
                measured_elapsed.append(trial["elapsed_s"])
                measured_tps.append(trial["tps"])
        assert measured_elapsed == audit["measured_elapsed_s"]
        assert measured_tps == audit["measured_tps"]
        job_path = Path(audit["raw_files"][0]["path"]).parent.parent / "case.json"
        assert digest(job_path) == audit["job_sha256"]
        job = json.loads(job_path.read_text())
        assert job["status"] == "COMPLETE" and job["job_id"] == audit["job_id"]
        assert job["protocol_sha256"] == audit["protocol_sha256"]
        audits[audit["job_id"]] = audit
        provenance.append({"path": str(path), "sha256": digest(path)})
    rows: list[dict[str, object]] = []
    for case in CASES:
        labels = [f"{case.case_id}-fresh{f}-{a}" for f in (0, 1) for a in ARMS]
        present = [label for label in labels if label in audits]
        missing = [label for label in labels if label not in audits]
        row: dict[str, object] = {
            "case_id": case.case_id,
            "B": case.batch,
            "C": case.concurrency,
            "expected_requests_per_trial": 4 * case.concurrency,
            "jobs_verified": len(present),
            "required_jobs": 6,
            "status": "NOT_RUN" if not present else "INCOMPLETE",
            "missing_jobs": missing,
        }
        if not missing:
            group = [audits[label] for label in labels]
            assert len({a["protocol_sha256"] for a in group}) == 1
            assert all(a["requests_per_trial"] == 4 * case.concurrency for a in group)
            for arm in ARMS:
                assert (
                    audits[f"{case.case_id}-fresh0-{arm}"][
                        "same_policy_signature_sha256"
                    ]
                    == audits[f"{case.case_id}-fresh1-{arm}"][
                        "same_policy_signature_sha256"
                    ]
                )
            elapsed = {
                arm: [
                    v
                    for f in (0, 1)
                    for v in audits[f"{case.case_id}-fresh{f}-{arm}"][
                        "measured_elapsed_s"
                    ]
                ]
                for arm in ARMS
            }
            baseline, off, guided = (elapsed[a] for a in ARMS)
            row.update(
                status="RAW_COMPLETE_REQUIRES_PROCESS_RESOURCE_OFFBOX_PROOF",
                protocol_sha256=group[0]["protocol_sha256"],
                off_vs_full=matched_ratios(baseline, off),
                guided_vs_full=matched_ratios(baseline, guided),
                guided_vs_off_elapsed_overhead_percent=100
                * (matched_ratios(guided, off)["mean_ratio"] - 1),
                measured_tps_mean={
                    a: statistics.mean(
                        v
                        for f in (0, 1)
                        for v in audits[f"{case.case_id}-fresh{f}-{a}"]["measured_tps"]
                    )
                    for a in ARMS
                },
                actual_A_peaks=[a["actual_A_peak"] for a in group],
                actual_S_peaks=[a["actual_S_peak"] for a in group],
                target_A_reached_each_job=[a["target_A_reached"] for a in group],
            )
        rows.append(row)
    args.output.mkdir(exist_ok=False)
    result = {
        "status": "INCOMPLETE"
        if len(audits) < 42
        else "FULL7_RAW_ONLY_REQUIRES_WHOLE_BINDINGS",
        "source_sha256": digest(Path(__file__)),
        "case_rows": rows,
        "required_cases": 7,
        "required_fresh_jobs": 42,
        "required_trials_including_warmup": 420,
        "jobs_verified": len(audits),
        "audit_provenance": provenance,
        "arrival": "4C closed-loop; matched workloads with reversed configuration blocks",
        "resource_model_source_gate_and_same_GPU": (
            "Must be independently proven by the actual owning controller "
            "and offbox archives; raw status alone is insufficient"
        ),
        "quality_or_goal_completion": "NOT_PROVEN",
    }
    target = args.output / "matrix.json"
    target.write_text(json.dumps(result, indent=2) + "\n")
    with (args.output / "RESULTS.md").open("x") as stream:
        stream.write(
            "# Re:Loop original W5 raw matrix\n\n"
            "| Case | B | C | Requests/trial | Verified jobs/6 | "
            "Guided speed vs full | Raw status |\n"
            "|---|---:|---:|---:|---:|---:|---|\n"
        )
        for row in rows:
            speed = "NOT_RUN"
            if "guided_vs_full" in row:
                comparison = cast(Comparison, row["guided_vs_full"])
                speed = f"{100 * (comparison['mean_ratio'] - 1):+.4f}%"
            stream.write(
                f"|{row['case_id']}|{row['B']}|{row['C']}|{row['expected_requests_per_trial']}|{row['jobs_verified']}/6|{speed}|{row['status']}|\n"
            )
        stream.write(
            "\nAll seven original cases are retained. Ratios use the ten matched "
            "measured repetitions, not a ratio of independent TPS means or "
            "adjacent-trial interleaving. Two fresh starts are two environment "
            "repeats. Different cases may require independently frozen "
            "hardware/pool protocols; every within-case arm/fresh comparison "
            "must share one protocol. Whole process natural0/absence, full "
            "source/checkpoint/gate/resource and offbox provenance are separate "
            "mandatory completion evidence. No quality, target-residency or "
            "global-goal claim follows from a raw-only status.\n"
        )
    print(
        json.dumps(
            {
                "status": result["status"],
                "matrix_sha256": digest(target),
                "cases": 7,
                "jobs_verified": len(audits),
            }
        )
    )


if __name__ == "__main__":
    main()
