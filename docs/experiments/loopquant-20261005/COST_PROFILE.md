# Native FP8 cost profile — Ouro1.4B / RTX 5090

The verified profile does not establish a scale-only performance win. The entire
fused FP8 producer occupies about 4% of accumulated GPU kernel duration and
includes mandatory work. Removing its reduction cannot be equated with removing
that entire duration. The separate uninstrumented pilot remains negative.

## Capture and verification

Source `290be852825c7cd7a1592ebb9853dfe9f774ca01`; model revision
`574fa66cb8bf5abdc979642d01cf2b79b16bfab1`. BF16, DYN and SH1 use the same
closed trace: C32, batch-token cap 512, prompt 256/output 32, fixed R4;
32 warmup requests precede 64 captured requests. All three finish 64 requests
and 2,048 output tokens, with zero failures, residual KV, timed Graph captures
and fallbacks. Graph replay counts are 362/362/361. Scheduling is asynchronous;
identical request traces do not force identical kernel invocation counts.

Existing Nsight Systems 2025.3.1.0 captured one GPU while both GPUs and heavy IO
were reserved. The finite profiler and application processes exited naturally.
All three profiles and 15 stats commands returned 0. The 108,192,114-byte offbox
archive has SHA256
`96c30f6244117369af59be29dcd0dced536ffc68dbba5901002f6edc482da24a`;
all 28 files passed independent size and SHA checks. Native reports and SQLite
files remain in that archive. [Completion](cost-profile-v1/complete.json),
[file inventory](cost-profile-v1/files.json), [exact commands and exit codes](cost-profile-v1/receipt.json).

The actual command template, with each resolved invocation in the receipt:

```sh
nsys profile --sample=none --cpuctxsw=none --trace=cuda,nvtx,cublas \
  --cuda-graph-trace=node --capture-range=cudaProfilerApi \
  --capture-range-end=stop --kill=none --wait=all \
  --output <policy> python probe.py --case <policy>.json --output <policy>.json
nsys stats -r cuda_gpu_kern_sum --format csv <policy>.nsys-rep
nsys stats -r cuda_api_sum --format csv <policy>.nsys-rep
nsys stats -r cuda_gpu_mem_time_sum --format csv <policy>.nsys-rep
nsys stats -r cuda_kern_exec_sum --format csv <policy>.nsys-rep
nsys stats -r cublas_api_sum --format csv <policy>.nsys-rep
```

## Measured profile output

Kernel sums come from `cuda_gpu_kern_sum`; interval unions and enclosing NVTX
ranges come from exported SQLite. The [analyzer](cost-profile-v1/analyze.py)
verifies every kernel lies inside its `engine-*` range and that the SQL duration
sum equals `nsys stats` exactly. Local recomputation matches the node result.
[Machine-readable results](cost-profile-v1/analysis.json).

| Policy | Instrumented NVTX range (ms) | Kernel calls | Kernel duration sum (ms) | Kernel interval union (ms) |
|---|---:|---:|---:|---:|
| BF16 | 6816.621 | 703,116 | 3687.713 | 3679.906 |
| DYN | 7114.366 | 532,718 | 3387.669 | 3380.336 |
| SH1 | 6965.301 | 531,481 | 3382.810 | 3375.461 |

| Kernel category, duration sum (ms) | BF16 | DYN | SH1 |
|---|---:|---:|---:|
| Fused norm/scale/cast/pack producer | 0.000 | 140.594 | 133.662 |
| FP8 GEMM | 0.000 | 1669.094 | 1670.397 |
| Other GEMM and epilogue | 1826.595 | 36.210 | 33.985 |
| Paged attention | 817.428 | 816.650 | 818.613 |
| Other reductions | 177.636 | 101.028 | 101.003 |
| Remaining kernels | 866.054 | 624.092 | 625.150 |

The FP8 GEMMs are the installed SM120 CUTLASS rowwise `_scaled_mm` path.
The `quantize_rows` producer includes normalization where fused, row scaling,
conversion and packing. Its DYN/SH1 stats are:

| Policy | Producer calls | Total (ms) | Mean per call (µs) | Share of kernel-duration sum |
|---|---:|---:|---:|---:|
| DYN | 47,040 | 140.594 | 2.9888 | 4.150% |
| SH1 | 46,944 | 133.662 | 2.8473 | 3.951% |

Mean producer duration is 4.74% lower in SH1.
This single instrumented observation has different invocation counts and shape
schedules; it is not a paired causal estimate, confidence interval or E2E speedup.
[BF16 kernel stats](cost-profile-v1/bf16-cuda_gpu_kern_sum.csv),
[DYN kernel stats](cost-profile-v1/dyn-cuda_gpu_kern_sum.csv),
[SH1 kernel stats](cost-profile-v1/sh1-cuda_gpu_kern_sum.csv).

## Host API evidence

| CUDA API duration sum (ms) | BF16 | DYN | SH1 |
|---|---:|---:|---:|
| cudaEventSynchronize | 1022.347 | 1258.625 | 1238.454 |
| cudaGraphLaunch_v10000 | 712.747 | 634.546 | 612.498 |
| cudaLaunchKernel | 659.912 | 588.771 | 585.932 |

[BF16 API stats](cost-profile-v1/bf16-cuda_api_sum.csv),
[DYN API stats](cost-profile-v1/dyn-cuda_api_sum.csv),
[SH1 API stats](cost-profile-v1/sh1-cuda_api_sum.csv).
These API durations can overlap GPU execution. Kernel categories and their
unions can also overlap each other. Neither sum is a removable critical path.
The larger instrumented NVTX ranges despite lower FP8 GPU kernel sums call for
host/scheduling analysis; these data alone do not establish its causal share.
No Amdahl E2E bound is inferred from the 4% kernel-sum fraction.

## Separate uninstrumented speed comparison

The prior one-trial smoke pilot at this C32/B512 shape measured:

| Policy | Output tok/s | Speedup vs BF16 | Throughput change vs BF16 | Speedup vs DYN |
|---|---:|---:|---:|---:|
| BF16 | 387.146 | 1.0000× | +0.00% | 1.0651× |
| DYN | 363.494 | 0.9389× | −6.11% | 1.0000× |
| SH1 | 361.615 | 0.9341× | −6.59% | 0.9948× |

[Pilot details and raw results](PILOT_E2E.md). No steady-state, repeated-trial
confidence or locked-quality qualification is claimed for that pilot.

## Decision

G1 is `inconclusive`. Initial dev quality is close across SH1 and LOOP,
but the measured producer cost and negative pilot do not justify full QAT for
a scale-only FP8 speed claim. The mixed-depth batching and lower-format routes
have not yet been measured sufficiently to resolve G1. Continue the fixed F0
RTN/GPTQ diagnostics and native family qualification; do not expand the full
QAT grid on this evidence. G0 numerical failures, locked G2 and formal G3 remain
separate and unchanged. The scale/conversion critical-path figure remains
unresolved; a kernel-sum chart must not substitute for that required evidence.
