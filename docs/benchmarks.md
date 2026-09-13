# Benchmarks

Run `python -m vllm_lt.benchmarks --help` for the benchmark CLI.
Prepare the local checkpoint and generate a CPU-only plan before reserving a GPU;
run inference under the scheduler-assigned device and save reports outside Git.

Keep generated plans, traces, logs and results in external experiment storage.
Record source, model, hardware, dtype, controls and run budgets before measuring.
Separate preparation and profiling from measured inference; report variability
and failed acceptance checks alongside successful checks.

Historical evidence is available in the [release assets](https://github.com/hsliuustc0106/vllm-lt/releases).
