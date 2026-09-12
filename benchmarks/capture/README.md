# CUDA graph benchmark and validation tooling

This directory contains the graph A/B controller, profiling adapter, offline
reporter, and numerical, kernel and lifecycle validation. It runs from a source
checkout and is excluded from the installed `vllm_lt` package. Shared benchmark
and oracle utilities remain in `vllm_lt.benchmarks` and `vllm_lt.validation`.

| Module | Purpose |
| --- | --- |
| `runner.py` | Controller and reserved-device workers |
| `schema.py` | CPU-only planning, source hashing and contract validation |
| `runtime.py` | Eager/replay adapter and profiler attribution |
| `report.py` | Offline correctness, timing and profile audits |
| `validation.py` | Numerical comparisons and graph setup audits |
| `kernels.py` | Held-input kernel validation |
| `lifecycle.py` | Executor ownership, teardown and failure validation |

From the repository root, with the prepared development/validation environment:

```bash
python -m benchmarks.capture --help
python -m benchmarks.capture source-probe
python -m benchmarks.capture probe --help
python -m benchmarks.capture report --run-dir /absolute/path/to/results
```

`probe` requires two clean source checkouts, a prepared local model, a contract,
an explicit physical GPU ID and verified CPU/NUMA affinity. It does not use CUDA.
`run` and `worker` perform GPU work and must execute within a scheduler reservation.
Keep generated plans, results, traces and logs in an external experiment directory
or the ignored `artifacts/` directory. Do not commit generated output.

The two JSON files in `fixtures/` are hand-authored input contracts for the
historical bounded 101-execution experiment, not generated results. Their bytes,
limits and historical four-live-row validation matrix are unchanged. Relocating
the tooling changes source provenance; existing frozen plans must be audited with
their recorded source snapshots. Historical reports retain the commands and paths
used at those snapshots. A new experiment needs a newly frozen plan.

See the [graph design](../../docs/m3-cuda-graphs.md),
[latest benchmark report](../../docs/benchmarks/m3-capture-review-20260912.md), and
[external evidence archive](../../docs/benchmarks/m3-capture-review-evidence-20260912/README.md).
