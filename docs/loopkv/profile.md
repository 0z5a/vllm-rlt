# Rectangular alias: first profile

The copy path shrinks, but the additional reader work exceeds the measured
copy-kernel saving in this diagnostic pair. The independent three-pair pilot
remains **0.964×**; profiling is not substituted for that throughput result.

Full Ouro-1.4B, B32/C64/P128/D256, original gate at 0.5, BF16 eager Triton,
RTX 5090 GPU `GPU-209bed64-f11a-c1f3-fe87-8a69538d8d79`. Source is
`37630710efb926fae8bdd6b576f77c28a3952e63`; model pin and workload match
[the pilot](results.md). Each arm uses a fresh process and a full warmup.
All 16,384 output tokens and exit depths match. Both profilers naturally exit 0.

## Commands and authoritative output

Capture wraps `experiments.loopkv.capture --measure --profile-range`:

```sh
nsys profile --kill=none --capture-range-end=stop \
  --capture-range=cudaProfilerApi --sample=none --cpuctxsw=none \
  --trace=cuda,nvtx -o ARM-trace python -m experiments.loopkv.capture ...
nsys stats --report cuda_gpu_kern_sum,cuda_api_sum,cuda_gpu_mem_time_sum \
  --format csv --output ARM-stats ARM-trace.nsys-rep
```

Nsight Systems 2025.3.1 reports the following aggregate device durations.
Milliseconds below are the raw nanoseconds divided by 1,000,000.

| GPU activity | Native count | Alias count | Native ms | Alias ms | Change, ms |
|---|---:|---:|---:|---:|---:|
| `_paged_attention_kernel` | 35,472 | 35,472 | 1,609.503 | 1,807.846 | +198.343 |
| BF16 elementwise direct-copy kernel, instance 12 | 59,520 | 1,468 | 117.370 | 1.621 | −115.749 |
| int fill kernels, all three variants | 0 | 17,856 | 0 | 12.244 | +12.244 |
| CUDA H2D memcpy | 4,292 | 7,248 | 1.304 | 2.512 | +1.208 |
| CUDA D2D memcpy | 52,822 | 52,822 | 41.153 | 41.194 | +0.041 |

Sources: [native kernels](evidence/nsight-pilot-v1/native-stats_cuda_gpu_kern_sum.csv),
[alias kernels](evidence/nsight-pilot-v1/alias-stats_cuda_gpu_kern_sum.csv),
[native transfers](evidence/nsight-pilot-v1/native-stats_cuda_gpu_mem_time_sum.csv),
[alias transfers](evidence/nsight-pilot-v1/alias-stats_cuda_gpu_mem_time_sum.csv).
Full demangled names and unrounded values remain in those CSVs.

`cudaLaunchKernel` calls decline from 1,811,438 to 1,771,242, while their summed
host API duration increases from 6,822.427 to 7,470.375 ms. Each additional
metadata staging copy also records/queries an event: both API counts rise from
2,956 to 5,912. Sources: [native APIs](evidence/nsight-pilot-v1/native-stats_cuda_api_sum.csv)
and [alias APIs](evidence/nsight-pilot-v1/alias-stats_cuda_api_sum.csv).

## Interpretation and next experiment

The strided promotion path launches a BF16 copy kernel, so unchanged D2D memcpy
counts do not mean that promotion was retained. The alias reader adds two
staging transfers per prepared batch and resolves every history position through
the exit table, including full-depth prompt positions. The profile therefore
motivates a direct-reader prefix and avoiding metadata for query depths that
cannot encounter an alias. This is a hypothesis to test, not a measured fix.

Summed kernel time is 8.127 s versus 8.243 s. These durations are not critical-path
fractions and cannot be subtracted from E2E time to predict a speedup. Profiled
E2E times are 32.126 s and 34.190 s, substantially slower than the unprofiled
pilot. No CPU-stack attribution, DRAM traffic measurement or formal confidence
interval is inferred from this single diagnostic pair. Raw `.nsys-rep` and
SQLite artifacts remain in the task's remote `evidence/profile-main-v1` directory.
