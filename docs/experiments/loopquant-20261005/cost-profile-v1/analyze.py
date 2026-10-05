"""Summarize saved Nsight CSV/SQLite artifacts without treating sums as a critical path."""

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import sqlite3


def union_ns(intervals: list[tuple[int, int]]) -> int:
    total, end = 0, 0
    for start, stop in sorted(intervals):
        total += max(0, stop - max(start, end))
        end = max(end, stop)
    return total


def category(name: str) -> str:
    if name == "quantize_rows":
        return "fused_fp8_producer"
    if name == "_paged_attention_kernel":
        return "paged_attention"
    if "float_e4m3_t" in name:
        return "fp8_gemm"
    if "gemm" in name.lower() or "splitKreduce" in name:
        return "other_gemm_and_epilogue"
    if "reduce_kernel" in name:
        return "other_reductions"
    return "other_kernels"


def summarize(folder: Path, policy: str) -> dict:
    report = folder / f"{policy}-cuda_gpu_kern_sum.csv"
    lines = report.read_text().splitlines()
    first = next(
        i
        for i, line in enumerate(lines)
        if line.startswith("Time (%),Total Time (ns),")
    )
    rows = list(csv.DictReader(io.StringIO("\n".join(lines[first:]))))
    csv_sum = sum(int(row["Total Time (ns)"]) for row in rows)
    db = sqlite3.connect(f"file:{folder / (policy + '.sqlite')}?mode=ro", uri=True)
    names = dict(db.execute("SELECT id, value FROM StringIds"))
    windows = db.execute(
        "SELECT start, end, text FROM NVTX_EVENTS WHERE text LIKE 'engine-%'"
    ).fetchall()
    assert len(windows) == 1 and windows[0][1] is not None
    begin, end, label = windows[0]
    kernels = db.execute(
        "SELECT start, end, demangledName FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start"
    ).fetchall()
    assert all(begin <= start <= stop <= end for start, stop, _ in kernels)
    assert csv_sum == sum(stop - start for start, stop, _ in kernels)
    labels = {identifier: category(name) for identifier, name in names.items()}
    groups = {}
    for start, stop, identifier in kernels:
        group = groups.setdefault(
            labels[identifier], dict(calls=0, sum_ns=0, intervals=[])
        )
        group["calls"] += 1
        group["sum_ns"] += stop - start
        group["intervals"].append((start, stop))
    for group in groups.values():
        group["union_ns"] = union_ns(group.pop("intervals"))
        group["mean_ns"] = group["sum_ns"] / group["calls"]
    producer = [row for row in rows if row["Name"] == "quantize_rows"]
    api = folder / f"{policy}-cuda_api_sum.csv"
    api_lines = api.read_text().splitlines()
    api_first = next(
        i
        for i, line in enumerate(api_lines)
        if line.startswith("Time (%),Total Time (ns),")
    )
    api_rows = list(csv.DictReader(io.StringIO("\n".join(api_lines[api_first:]))))
    api_top = sorted(
        api_rows, key=lambda row: int(row["Total Time (ns)"]), reverse=True
    )[:8]
    result = dict(
        policy=policy.upper(),
        nvtx_label=label,
        nvtx_range_ns=end - begin,
        kernel_calls=len(kernels),
        kernel_sum_ns=csv_sum,
        kernel_union_ns=union_ns([(start, stop) for start, stop, _ in kernels]),
        groups=groups,
        producer_csv=producer,
        cuda_api_top=api_top,
        csv_sha256=hashlib.sha256(report.read_bytes()).hexdigest(),
        sql_scope="kernel intervals inside engine NVTX range; sum verified against nsys stats; category unions can overlap",
    )
    db.close()
    return result


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("folder", type=Path)
args = parser.parse_args()
print(
    json.dumps(
        dict(
            scope="instrumented cost only, not speedup or removable critical-path time",
            policies=[summarize(args.folder, p) for p in ["bf16", "dyn", "sh1"]],
        ),
        indent=2,
    )
)
