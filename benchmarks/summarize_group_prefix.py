"""Summarize three independent CPU admission-audit processes without pooling arms."""

import argparse
import json
import math
import statistics
from pathlib import Path


def interval(values: list[float]) -> tuple[float, float, float]:
    assert len(values) == 3
    logs = [math.log(value) for value in values]
    mean = statistics.mean(logs)
    half = 4.30265273 * statistics.stdev(logs) / math.sqrt(3)
    return math.exp(mean), math.exp(mean - half), math.exp(mean + half)


def summarize(directory: Path) -> str:
    rows = []
    for seed in (31, 32, 33):
        cohort = [
            json.loads(line)
            for line in (directory / f"observations-s{seed}.jsonl").read_text().splitlines()
        ]
        assert len(cohort) == 108 and {row["seed"] for row in cohort} == {seed}
        rows.extend(cohort)
    table = [
        "| Family | P | C | Cold prefill tokens | Warm prefill tokens | "
        "Disabled/cold ratio [95% CI] | Disabled/warm ratio [95% CI] | "
        "Disabled/(primer+warm) [95% CI] |",
        "| --- | ---: | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for family in ("ouro", "nanbeige"):
        for length in (15, 16, 17):
            for concurrency in (1, 4, 8):
                case = [
                    row
                    for row in rows
                    if (row["family"], row["prompt_tokens"], row["concurrency"])
                    == (family, length, concurrency)
                ]
                assert len(case) == 18
                assert {row["peak_live_requests"] for row in case} == {concurrency}
                for row in case:
                    assert row["prefill_tokens"] + row["prefix_hit_tokens"] == length * 8
                    assert row["prefix_queries"] == 8 and row["used_pages_after"] == 0
                    assert len(row["outputs"]) == 8 and row["max_score_error"] < 1e-5
                prefill = {
                    arm: {row["prefill_tokens"] for row in case if row["arm"] == arm}
                    for arm in ("disabled", "cold", "warm")
                }
                assert all(len(values) == 1 for values in prefill.values())
                assert prefill["disabled"] == {length * 8}
                counts = {arm: next(iter(values)) for arm, values in prefill.items()}
                ratios = []
                for arm, field in (
                    ("cold", "wall_seconds"),
                    ("warm", "wall_seconds"),
                    ("warm", "primer_plus_group_seconds"),
                ):
                    paired = []
                    for seed in (31, 32, 33):
                        baseline = [
                            row["wall_seconds"]
                            for row in case
                            if row["seed"] == seed and row["arm"] == "disabled"
                        ]
                        candidate = [
                            row[field] for row in case if row["seed"] == seed and row["arm"] == arm
                        ]
                        assert len(baseline) == len(candidate) == 2
                        paired.append(statistics.mean(baseline) / statistics.mean(candidate))
                    mean, low, high = interval(paired)
                    ratios.append(f"{mean:.3f} [{low:.3f}, {high:.3f}]")
                table.append(
                    f"| {family} | {length} | {concurrency} | {counts['cold']} | "
                    f"{counts['warm']} | " + " | ".join(ratios) + " |"
                )
    print(
        json.dumps(
            {
                "observations": len(rows),
                "measured_requests": sum(len(row["outputs"]) for row in rows),
                "measured_output_tokens": sum(
                    len(output["tokens"]) for row in rows for output in row["outputs"]
                ),
                "max_score_error": max(row["max_score_error"] for row in rows),
            }
        )
    )
    return "\n".join(table) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(summarize(args.directory))
