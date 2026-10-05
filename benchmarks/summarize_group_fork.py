"""Audit all cold ABBA observations and retain every workload in the result table."""

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


def summarize(directory: Path, parent_manifest: Path) -> dict:
    parent = json.loads(parent_manifest.read_text())["files"]
    candidate = json.loads((directory / "candidate-source.json").read_text())
    receipts = json.loads((directory / "processes.json").read_text())
    identities = [
        f"s{seed}-{label}"
        for seed in (31, 32, 33)
        for label in ("parent-a1", "fork-b1", "fork-b2", "parent-a2")
    ]
    assert [r["identity"] for r in receipts] == identities
    assert all(r["returncode"] == 0 for r in receipts)
    grouped = defaultdict(list)
    rows = []
    for identity in identities:
        observations = [
            json.loads(line) for line in (directory / f"{identity}.jsonl").read_text().splitlines()
        ]
        assert len(observations) == 40
        seen = set()
        for row in observations:
            key = (row["family"], row["prompt_length"], row["group_size"], row["concurrency"])
            assert key not in seen
            seen.add(key)
            assert f"s{row['seed']}-{row['label']}" == identity
            assert row["source_sha256"] == (candidate if row["fork"] else parent)
            assert row["threads"] == 1 and row["capacity_fallbacks"] == 0
            assert row["peak_active_requests"] == min(row["concurrency"], row["group_size"])
            assert (
                row["prefill_tokens"]
                + row["ordinary_prefix_hit_tokens"]
                + row["group_reused_tokens"]
                == row["prompt_length"] * row["group_size"]
            )
            if row["fork"]:
                assert row["prefill_tokens"] == row["prompt_length"]
                assert row["group_forks"] == row["group_size"] - 1
            assert len(row["outputs"]) == row["group_size"]
            assert len(row["request_completion_seconds"]) == row["group_size"]
            assert all(len(out["tokens"]) == len(out["scores"]) == 8 for out in row["outputs"])
            grouped[key].append(row)
        rows.extend(observations)
    assert len(grouped) == 40 and len(rows) == 480
    max_error = 0.0
    results = []
    for key, observations in sorted(grouped.items()):
        ratios = []
        for seed in (31, 32, 33):
            cohort = [r for r in observations if r["seed"] == seed]
            assert len(cohort) == 4
            reference = cohort[0]["outputs"]
            for row in cohort[1:]:
                for actual, expected in zip(row["outputs"], reference, strict=True):
                    for field in ("id", "tokens", "depths", "stop", "version"):
                        assert actual[field] == expected[field], (key, seed, field)
                    for a, b in zip(actual["scores"], expected["scores"], strict=True):
                        error = abs(a - b)
                        assert error <= 1e-5 + 3e-5 * abs(b), (key, seed, a, b)
                        max_error = max(max_error, error)
            times = [
                statistics.mean(r["wall_seconds"] for r in cohort if r["fork"] == arm)
                for arm in (False, True)
            ]
            ratios.append(times[0] / times[1])
        logs = [math.log(ratio) for ratio in ratios]
        mean = statistics.mean(logs)
        half = 4.302652729911275 * statistics.stdev(logs) / math.sqrt(3)
        entry = dict(zip(("family", "prompt_length", "group_size", "concurrency"), key))
        entry.update(
            ratio=math.exp(mean),
            ci95=[math.exp(mean - half), math.exp(mean + half)],
            seed_ratios=ratios,
        )
        for arm, label in ((False, "parent"), (True, "fork")):
            subset = [r for r in observations if r["fork"] == arm]
            entry[label + "_seconds"] = statistics.mean(r["wall_seconds"] for r in subset)
            for field in (
                "prefill_tokens",
                "peak_used_pages_including_group_snapshot",
                "peak_shared_pages_including_cache_owners",
                "scheduler_steps",
            ):
                values = {r[field] for r in subset}
                assert len(values) == 1, (key, label, field, values)
                entry[label + "_" + field] = values.pop()
        results.append(entry)
    return {
        "observations": len(rows),
        "processes": len(receipts),
        "measured_requests": sum(row["group_size"] for row in rows),
        "measured_output_tokens": sum(row["group_size"] * 8 for row in rows),
        "max_cross_arm_score_error": max_error,
        "max_oracle_score_error": max(r["max_reference_score_error"] for r in rows),
        "workloads": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--parent-manifest", type=Path, required=True)
    options = parser.parse_args()
    summary = summarize(options.directory, options.parent_manifest)
    (options.directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = [
        "| Family | P | G | C | Parent ms | Fork ms | Speed ratio [95% CI] | "
        "Prefill A/B | Live pages A/B |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for row in summary["workloads"]:
        lo, hi = row["ci95"]
        lines.append(
            f"| {row['family']} | {row['prompt_length']} | {row['group_size']} | "
            f"{row['concurrency']} | {row['parent_seconds'] * 1000:.2f} | "
            f"{row['fork_seconds'] * 1000:.2f} | {row['ratio']:.3f} [{lo:.3f}, {hi:.3f}] | "
            f"{row['parent_prefill_tokens']}/{row['fork_prefill_tokens']} | "
            f"{row['parent_peak_used_pages_including_group_snapshot']}/"
            f"{row['fork_peak_used_pages_including_group_snapshot']} |"
        )
    (options.directory / "table.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "workloads"}, indent=2))


if __name__ == "__main__":
    main()
