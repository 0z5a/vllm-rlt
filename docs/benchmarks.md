# Benchmarks

- [Roadmap M1 specification](roadmap.md#m1-first-benchmark-and-profiler-pr): workload, budget, timing and scheduler contract.
- [M1 baseline — 2026-09-10](benchmarks/m1-20260910.md): frozen configuration, results and evidence.
- [M2 metadata guide](m2-metadata.md): interfaces and repository-tool commands.
- [M2 comparison — 2026-09-11](benchmarks/m2-20260911.md): frozen worker order, execution counts and acceptance results.
- [Q1 validation guide](q1-validation.md) and [Q1 results — 2026-09-11](validation/q1-20260911.md).
- [Initial validation — 2026-09-10](validation/initial-20260910.md): original checkpoint smoke and BF16 failure.

New experiments follow the [precision policy](precision-policy.md). The current
M1/M2 executable contracts retain FP32; BF16 requires versioned harness support.
Run repository tools from the checkout root with `python -m benchmarks --help`
or `python -m benchmarks.ab --help`. Freeze new plans after source changes;
reproduce historical evidence using the source commits recorded in each report.

Review discussions live on [PR 10](https://github.com/hsliuustc0106/vllm-lt/pull/10),
[PR 11](https://github.com/hsliuustc0106/vllm-lt/pull/11), and
[PR 12](https://github.com/hsliuustc0106/vllm-lt/pull/12).
The former local follow-up notes, including their check counts and deferred API
choices, remain in [the pinned archive](https://github.com/hsliuustc0106/vllm-lt/tree/c9e646ace2d8198a52e522ea4d71129f1b143c58/docs/reviews).
