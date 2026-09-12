# PR15 review evidence — 2026-09-12

Implementation: `af3327d5a30219bcc92c57deb43a10cc7be3040f`.
The root-cause diagnostic uses unchanged predecessor
`65a719d9086fe96f311b2a195d424c2a01c9bbf2`.

[Results and limits](../m3-capture-review-20260912.md) separate the successful
30-measurement performance screen, fixed-concurrency lifecycle plateau and
actual replay attribution from the incomplete profiling protocols.

- `lifecycle-summary.json`: derived directly from CUDA memory snapshots and
  real-model output histories; includes the six baseline diagnostic lifetimes
  and seventeen candidate churn lifetimes.
- `stream-plan.json`, `lifecycle-plan.json`, `timing-plan.json`, `profile-plan.json`:
  frozen hypotheses, controls, source/model hashes, budgets and stopping rules.
- `report.json`: audited timing, per-request latency, padding, setup projections,
  cleanup and launch/kernel attribution. `report.py.txt` recomputes metrics and
  checks host events, inventories and profile correlations from raw artifacts.
- `*-stop.json`: preserve each separate preflight, environment, profiler-time
  and artifact-cap failure. No failed attempt is relabeled as complete.
- The `.py.txt` files preserve the actual experiment drivers without adding
  archived scripts to runtime tooling. They retain their recorded absolute
  paths, not a portable benchmark CLI. `baseline-plan.json`
  supplies the prior plan referenced by the stream diagnostic.

Full local evidence archive: `/home/hsliu2/tmp/vllm-lt-graph-review-20260912/evidence.tar.gz`.
It contains both source snapshots, all attempts, profiles, raw results, logs,
plans, diagnostics and the archive manifest. Checkpoint weights and virtual
environment packages are excluded. The source snapshots and stopped attempts
remain separately named. Raw traces have not been uploaded to GitHub.

SHA-256: `70309cc90bbd55db79c211dc16916f6e5e6eaba3f139f883afb3194e7f5ccf30`. Archive size: 156,479,663 bytes.
All 846 payload files were read back from the archive and checked
against their recorded sizes and SHA-256 hashes. Compression does not change the
short-profile experiment's failed **raw** artifact-budget decision.

CPU checks: 956 passed, 49 GPU skips, with CUDA discovery/initialization blocked.
The separate reserved-device invocation passed those 49 GPU kernel tests.
Ruff lint/format and whitespace checks passed. The original profiling worker's
interrupted teardown retained traceback-owned tensors; its process exited and
GPU 2 was verified free at zero memory before the next reservation. Later
workers closed successfully. All task-owned reservations have ended.
