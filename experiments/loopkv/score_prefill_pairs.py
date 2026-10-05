"""Independently audit warmed generation pairs and write the speed table."""

import argparse
import hashlib
import json
import random
import statistics
from collections import Counter
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("run", type=Path)
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
manifest = json.loads((args.run / "plan.json").read_text())
plan = manifest["plan"]
assert json.loads((args.run / "DONE.json").read_text())["returncode"] == 0
rows = []
all_requests = 0
for case in plan["cases"]:
    ref = args.run / f"{case['name']}-reference"
    reference = json.loads((ref / "reference-requests.json").read_text())
    ref_row = json.loads((ref / "reference.json").read_text())
    steps = json.loads((ref / "steps.json").read_text())
    histogram = Counter(s["rows"] for s in steps if s["stage"] == "recurrent")
    assert ref_row["work"]["recurrent_batch_histogram"] == {str(k): v for k, v in histogram.items()}
    assert ref_row["work"]["submitted_recurrent_rows"] == sum(k * v for k, v in histogram.items())
    assert (
        ref_row["work"]["submitted_prefill_tokens"]
        == sum(s["rows"] for s in steps if s["stage"] == "prefill")
        == case["requests"] * plan["prompt_length"]
    )
    assert ref_row["work"]["observed_peak_residents_after_step"] == max(
        s["resident_requests"] for s in steps
    )
    assert ref_row["work"]["submitted_recurrent_depth_rows"] == {
        str(depth): case["requests"] * (plan["outputs"] - 1)
        for depth in range(1, plan["loops"] + 1)
    }
    assert set(reference) == {str(i) for i in range(case["requests"])}
    for rid, output in reference.items():
        assert (
            output["request_id"] == rid
            and output["finished"]
            and output["finish_reason"] == "length"
        )
        assert len(output["token_ids"]) == plan["outputs"]
        assert output["exit_depths"] == [plan["loops"]] * plan["outputs"]
    all_requests += len(reference)
    for storage in plan["storage"]:
        ratios, samples, off, on = [], [], [], []
        for pair, order in enumerate(plan["orders"]):
            results = {}
            for enabled in order:
                label = "on" if enabled else "off"
                folder = args.run / f"{case['name']}-{storage}-pair{pair}-{label}"
                for phase in ("warmup", "measured"):
                    row = json.loads((folder / f"{phase}.json").read_text())
                    path = folder / f"{phase}-requests.json"
                    assert json.loads(path.read_text()) == reference
                    assert row["work"] == ref_row["work"]
                    assert row["free_blocks"] == case["blocks"]
                    assert (
                        row["case"] == case
                        and row["storage"] == storage
                        and row["enabled"] == enabled
                        and row["phase"] == phase
                    )
                    assert (
                        row["output_tokens"] == case["requests"] * plan["outputs"]
                        and row["seconds"] > 0
                    )
                    if storage == "compact":
                        assert row["compact_live_records"] == row["compact_reserved_records"] == 0
                    all_requests += len(reference)
                    row["requests_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
                    samples.append(row)
                results[label] = row
            receipt = json.loads(
                (args.run / f"{case['name']}-{storage}-pair{pair}.json").read_text()
            )
            ratio = results["off"]["seconds"] / results["on"]["seconds"]
            assert receipt["speedup"] == ratio and receipt["order"] == order
            ratios.append(ratio)
            off.append(results["off"]["seconds"])
            on.append(results["on"]["seconds"])
        rng = random.Random(20261006)
        bootstrap = sorted(
            statistics.median(rng.choices(ratios, k=len(ratios))) for _ in range(10000)
        )
        tokens = case["requests"] * plan["outputs"]
        rows.append(
            {
                "case": case,
                "storage": storage,
                "pairs": len(ratios),
                "ratios": ratios,
                "median_paired_speedup": statistics.median(ratios),
                "paired_bootstrap95": [bootstrap[250], bootstrap[9749]],
                "baseline_median_tokens_s": tokens / statistics.median(off),
                "candidate_median_tokens_s": tokens / statistics.median(on),
                "baseline_seconds": off,
                "candidate_seconds": on,
                "work": ref_row["work"],
                "samples": samples,
            }
        )
report = {
    "scope": manifest["scope"],
    "manifest": manifest,
    "rows": rows,
    "all_requests": all_requests,
    "all_tokens": all_requests * plan["outputs"],
    "all_output_objects_and_work_exact": True,
}
args.out.write_text(json.dumps(report, indent=2) + "\n")
table = [
    "| B / C | Storage | Baseline tok/s | Candidate tok/s | "
    "Paired improvement | Paired bootstrap 95% |",
    "|---|---|---:|---:|---:|---:|",
]
for row in rows:
    lo, hi = row["paired_bootstrap95"]
    table.append(
        f"| {row['case']['batch']} / {row['case']['requests']} | {row['storage']} | "
        f"{row['baseline_median_tokens_s']:.2f} | {row['candidate_median_tokens_s']:.2f} | "
        f"{(row['median_paired_speedup'] - 1) * 100:+.2f}% | "
        f"{(lo - 1) * 100:+.2f}% to {(hi - 1) * 100:+.2f}% |"
    )
args.out.with_suffix(".md").write_text("\n".join(table) + "\n")
print("\n".join(table))
print(json.dumps({"requests": all_requests, "tokens": report["all_tokens"], "exact": True}))
