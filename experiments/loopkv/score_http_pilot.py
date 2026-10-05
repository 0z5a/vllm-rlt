"""Audit independent native HTTP cohorts and apply the preregistered SLO rule."""

import argparse
import hashlib
import json
import math
import statistics
from collections import defaultdict
from itertools import pairwise
from pathlib import Path


def quantile(values, fraction):
    return sorted(values)[math.ceil(len(values) * fraction) - 1]


def read_hashed(path, expected):
    payload = path.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == expected, path.name
    return json.loads(payload)


def audit_cohort(raw, workload, case, sampling):
    expected = {row["request_id"]: row for row in workload}
    completed = {row["request_id"]: row for row in raw["requests"]}
    assert len(completed) == len(raw["requests"]) == len(workload)
    assert completed.keys() == expected.keys()
    assert raw["arrival_mode"] == "closed_loop"
    assert 0 < raw["peak_client_concurrency"] <= case["concurrency"]
    samples = defaultdict(list)
    for rid, row in completed.items():
        times = row["token_times_s"]
        assert len(times) == len(row["token_ids"]) == expected[rid]["max_tokens"]
        assert len(row["exit_depths"]) == len(times)
        assert all(sampling["min_loops"] <= d <= sampling["max_loops"] for d in row["exit_depths"])
        assert row["finish_reason"] == "length" and times == sorted(times)
        assert row["planned_s"] == expected[rid]["arrival_s"]
        assert 0 <= row["sent_s"] <= times[0] <= times[-1] <= row["finished_s"] <= raw["seconds"]
        assert [chunk["time_s"] for chunk in row["chunks"] for _ in range(chunk["tokens"])] == times
        samples["ttft"].append(times[0] - row["sent_s"])
        samples["request"].append(row["finished_s"] - row["sent_s"])
        samples["per_request_p99_itl"].append(quantile([b - a for a, b in pairwise(times)], 0.99))
    work = raw["work"]
    histogram = {int(k): v for k, v in work["recurrent_batch_histogram"].items()}
    assert work["submitted_prefill_tokens"] == sum(len(row["prompt_token_ids"]) for row in workload)
    assert work["submitted_recurrent_rows"] == sum(k * v for k, v in histogram.items())
    assert work["submitted_recurrent_rows"] == sum(work["submitted_recurrent_depth_rows"].values())
    assert work["recurrent_calls"] == sum(histogram.values())
    assert math.isclose(
        work["effective_batch_mean"],
        work["submitted_recurrent_rows"] / work["recurrent_calls"],
    )
    assert 0 < work["observed_peak_residents_after_step"] <= case["batch"]
    return completed, {name: quantile(values, 0.99) for name, values in samples.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    receipt = json.loads((args.results / "controller.json").read_text())
    assert receipt["plan_sha256"] == hashlib.sha256(args.plan.read_bytes()).hexdigest()
    assert receipt["returncode"] == 0
    assert json.loads((args.results / "DONE.json").read_text())["returncode"] == 0
    assert len(plan["runs"]) == len(receipt["workers"]) == 12
    assert len({row["pid"] for row in receipt["workers"]}) == 12
    groups = defaultdict(list)
    references = {}
    checkpoint = None
    for spec, worker in zip(plan["runs"], receipt["workers"]):
        assert worker["run"] == spec and worker["returncode"] == 0
        experiment = read_hashed(
            args.plan.parent / Path(spec["experiment_plan"]).name,
            spec["experiment_sha256"],
        )
        case = next(row for row in experiment["cases"] if row["name"] == spec["case"])
        arm = next(row for row in experiment["arms"] if row["name"] == spec["arm"])
        assert not arm["alias"] and not arm["compact"] and case["slo"] is None
        workload = read_hashed(
            args.plan.parent / Path(case["workload"]).name, case["workload_sha256"]
        )
        warmup_workload = read_hashed(
            args.plan.parent / Path(case["warmup"]["workload"]).name,
            case["warmup"]["workload_sha256"],
        )
        folder = args.results / spec["name"]
        metadata = json.loads((folder / "plan.json").read_text())
        assert metadata["plan"] == experiment and metadata["pid"] == worker["pid"]
        assert metadata["source_sha"] == spec["source_sha"]
        assert metadata["selection"] == {"case": spec["case"], "arm": spec["arm"]}
        assert metadata["lock_scope"] == "outer_quiet"
        if checkpoint is None:
            checkpoint = metadata["checkpoint"]
        assert checkpoint == metadata["checkpoint"]
        assert json.loads((folder / "DONE.json").read_text())["returncode"] == 0
        folder /= spec["case"] + "-" + spec["arm"]
        raw = json.loads((folder / "requests.json").read_text())
        summary = json.loads((folder / "summary.json").read_text())
        warmup = json.loads((folder / "warmup0.json").read_text())
        assert len(warmup_workload) == 128 and len(workload) == 512
        audit_cohort(warmup, warmup_workload, case, experiment["sampling"])
        assert summary["case"] == case and summary["arm"] == arm
        completed, p99 = audit_cohort(raw, workload, case, experiment["sampling"])
        output_tokens = sum(len(row["token_ids"]) for row in completed.values())
        assert summary["work"] == raw["work"]
        assert summary["peak_client_concurrency"] == raw["peak_client_concurrency"]
        key = spec["case"], spec["arm"]
        reference_name, reference = references.setdefault(key, (spec["name"], completed))
        differences = {
            field + "_different_requests": sum(
                row[field] != reference[rid][field] for rid, row in completed.items()
            )
            for field in ("token_ids", "exit_depths")
        }
        for name, value in p99.items():
            assert math.isclose(value, summary["latency_seconds"][name]["0.99"], abs_tol=1e-9)
        assert summary["complete_requests"] == 512 and summary["output_tokens"] == output_tokens
        assert summary["seconds"] == raw["seconds"]
        assert math.isclose(summary["tokens_s"], output_tokens / raw["seconds"], rel_tol=1e-12)
        groups[spec["case"], spec["arm"]].append(
            {
                "run": spec["name"],
                "pid": worker["pid"],
                "p99": p99,
                "seconds": raw["seconds"],
                "requests_s": 512 / raw["seconds"],
                "output_tokens": output_tokens,
                "graphs": summary["graphs"],
                "work": raw["work"],
                "output_reference_run": reference_name,
                **differences,
            }
        )
    calibrated = []
    for (case, arm), runs in sorted(groups.items()):
        assert len(runs) == 3
        slo = {
            target: math.ceil(1.5 * max(run["p99"][metric] for run in runs) / step) * step
            for target, metric, step in [
                ("ttft_s", "ttft", 0.1),
                ("request_s", "request", 0.1),
                ("request_p99_itl_s", "per_request_p99_itl", 0.01),
            ]
        }
        rate = statistics.median(run["requests_s"] for run in runs)
        calibrated.append(
            {
                "case": case,
                "native_arm": arm,
                "slo": slo,
                "open_loop_rates_requests_s": [scale * rate for scale in (0.5, 0.8, 1.0, 1.2)],
                "runs": runs,
            }
        )
    args.out.write_text(
        json.dumps(
            {
                "scope": (
                    "preregistered native calibration only; excluded from candidate performance"
                ),
                "plan_sha256": receipt["plan_sha256"],
                "checkpoint": checkpoint,
                "measured_requests": 6144,
                "warmup_requests": 1536,
                "calibrated": calibrated,
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps({"audited_independent_workers": 12, "calibrated_points": len(calibrated)}))


if __name__ == "__main__":
    main()
