# Benchmarks and validation records

- [M1 baseline — 2026-09-10](benchmarks/m1-20260910.md).
- [M2 metadata comparison — 2026-09-11](benchmarks/m2-20260911.md).
- [Q1 numerical validation — 2026-09-11](validation/q1-20260911.md).
- [Initial validation — 2026-09-10](validation/initial-20260910.md).

The completed M1/Q1/M2 experiment controllers, fixed budgets, contract validators,
fixtures and report builders have been retired. Reproduce those experiments with
[the archived harness](https://github.com/hsliuustc0106/vllm-lt/tree/24030af03085ccb0cf088872a2a6f4bbf26e937e)
or the exact source commits recorded in each report; their old CLI commands do
not run from the current checkout. Results and raw-evidence links remain unchanged.

Reusable helpers remain in `benchmarks.observe`, `benchmarks.profile`, and
`benchmarks.replay`. Independent references and tensor comparison tools remain
in `vllm_lt.validation`. Their tests cover timing, replay semantics, numerical
comparisons and ownership; the experiment-specific contract tests were removed.
New measurements follow the [precision policy](precision-policy.md).
