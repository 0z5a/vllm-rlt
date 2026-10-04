"""Paired trial statistics specified by the execution protocol."""

import math
import statistics
from dataclasses import dataclass


@dataclass(frozen=True)
class TrialPair:
    trial_id: str
    baseline_tokens_s: float
    candidate_tokens_s: float
    baseline_tokens: int
    candidate_tokens: int
    failures: int = 0
    captures_during_measurement: int = 0


@dataclass(frozen=True)
class Speedup:
    pairs: int
    median_ratio: float
    lower_bound: float | None
    coverage: float | None


def paired_speedup(pairs: list[TrialPair]) -> Speedup:
    if not pairs or len({pair.trial_id for pair in pairs}) != len(pairs):
        raise ValueError("expected distinct nonempty paired trials")
    ratios = []
    for pair in pairs:
        if pair.failures or pair.captures_during_measurement:
            raise ValueError("failed requests or timed captures invalidate a performance pair")
        if pair.baseline_tokens <= 0 or pair.baseline_tokens != pair.candidate_tokens:
            raise ValueError("paired trials must complete the same positive output token count")
        if not all(
            math.isfinite(value) and value > 0
            for value in (pair.baseline_tokens_s, pair.candidate_tokens_s)
        ):
            raise ValueError("throughputs must be finite and positive")
        ratios.append(pair.candidate_tokens_s / pair.baseline_tokens_s)
    ratios.sort()
    n = len(ratios)
    # P(Binomial(n, .5) <= k-1) <= .05, calculated with exact integers.
    tail = 0
    index = 0
    for k in range(1, n + 1):
        tail += math.comb(n, k - 1)
        if 20 * tail <= 2**n:
            index = k
        else:
            break
    coverage = 1 - sum(math.comb(n, j) for j in range(index)) / 2**n if index else None
    return Speedup(n, statistics.median(ratios), ratios[index - 1] if index else None, coverage)


def speedup_table(cases: dict[str, list[TrialPair]]) -> str:
    lines = [
        "| Case | Pairs | Baseline tok/s (median) | Candidate tok/s (median) | "
        "Paired speedup | Throughput change | One-sided ≥95% lower bound |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, pairs in cases.items():
        result = paired_speedup(pairs)
        baseline = statistics.median(pair.baseline_tokens_s for pair in pairs)
        candidate = statistics.median(pair.candidate_tokens_s for pair in pairs)
        bound = f"{result.lower_bound:.4f}×" if result.lower_bound is not None else "pilot only"
        lines.append(
            f"| {name} | {result.pairs} | {baseline:.3f} | {candidate:.3f} | "
            f"{result.median_ratio:.4f}× | {100 * (result.median_ratio - 1):+.2f}% | {bound} |"
        )
    return "\n".join(lines) + "\n"
