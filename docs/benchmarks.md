# Ouro M1 benchmark

The M1 harness measures the synchronous FP32 Ouro engine on one reserved GPU.
It follows [issue #3](https://github.com/hsliuustc0106/vllm-lt/issues/3).
A completed baseline is a measurement milestone; it does not require a speedup.
The [2026-09-10 baseline report](benchmarks/m1-20260910.md) records the first
completed matrix, acceptance checks, measured variability, and next target.

Prepare the pinned `ByteDance/Ouro-1.4B` checkpoint at revision
`574fa66cb8bf5abdc979642d01cf2b79b16bfab1` before running. Use a clean, committed
checkout and an environment with PyTorch, Triton, and the package dependencies.
The probe reads local configuration and hashes files; it neither loads weights
nor queries CUDA. Downloading and tokenization are outside the experiment.
The committed fixture records its tokenizer and synthetic replay provenance.

```bash
python -m vllm_lt.benchmarks probe \
  --suite benchmarks/fixtures/ouro-m1.json \
  --contract benchmarks/fixtures/ouro-m1-contract.json \
  --model-path /path/to/prepared/ouro-1.4b \
  --output artifacts/m1-plan
```

Inspect scheduler status and select an available exact physical GPU ID. Replace
`<available-id>` below with that verified ID. The scheduler assigns visibility;
the runner checks the assignment and never chooses devices.

```bash
gpu run --gpu-ids <available-id> --timeout 2h --note "vllm-lt M1 baseline" -- \
  python -m vllm_lt.benchmarks run \
  --plan artifacts/m1-plan/plan.json \
  --output artifacts/m1-run

python -m vllm_lt.benchmarks report \
  --run-dir artifacts/m1-run \
  --output artifacts/m1-report
```

Every output directory must be new. Input or source drift rejects the frozen
plan. Exit 0 means the requested contract completed, 2 means invalid input,
and 1 means execution/report coverage failed. The report is CPU only and
recomputes metrics from raw records without the checkpoint or original checkout.

The plan contains seven cells: W1 single-request 128/64, W2 eight-request
128/64, W3 single-request 512/32, and mixed-length W4/W5 replay in both scheduler
modes. W4 has heterogeneous depths 2/3/4; W5 is a four-loop control. W1–W3
perform fixed-four-loop generation. W4/W5 impose synthetic token IDs and exits
after real gate, coda, sampling, and KV work. Their outputs do not measure model
accuracy or adaptive quality.

The fixed budget is seven feasibility executions, seven warmups, fourteen
measured executions, and two profiler executions. W4/W5 each use paired order
refill-1, no-refill-1, no-refill-2, refill-2. There is no retry or resume.
A 600-second guard applies to each workload and a two-hour deadline covers the
whole process, including preparation and teardown. The scheduler enforces the
outer limit even if a device call blocks. An incomplete suite remains visible.

One model stays resident. Each execution creates and destroys a fresh engine
and 6 GiB KV pool outside its timer. Allocator and compiled caches persist.
No shared cache is cleared; process-local cuBLAS cleanup happens only at final
teardown. Active requests and reserved KV pages must return to zero after each
execution; persistent model/pool/allocator bytes are reported separately.

Timing begins immediately before enqueue after observer setup. Each step has
one host return timestamp shared by its outputs. Throughput includes prefill
and queueing; TPOT excludes the first prediction. The final device synchronization
is a separate boundary. The CUDA stream span can include host-induced gaps and
is not summed kernel busy time. Timing mode adds only host records/counters and
boundary events. Feasibility checks read finite hidden/logit values; they are
excluded from measurements.

Profiling starts at the first prelude and stops after at least sixteen subsequent
outputs, then drains outside capture. W4 can include naturally interleaved
prefill. CPU scope durations are inclusive and overlap; do not sum them as
independent costs. Chrome traces must contain actual kernel and memcpy events.
Profile-only snapshots distinguish reserved pages from populated physical pages;
logical KV copy bytes are payload accounting, not measured DRAM traffic.

The experiment directory contains its plan, environment/source manifest,
per-run start markers, raw events/results, and profiler traces/metadata. Failed
runs retain available records and are ineligible for comparison. The offline
report verifies hashes, replay/work accounting, cleanup, and matched pairs;
it shows both observations and their range. Two runs are a bounded screen,
and overlapping variation is inconclusive. Publish compact records under
`docs/benchmarks/`; retain full local artifacts under ignored `artifacts/`.
