# Terminal output collection: CPU mechanism experiment

This measures output materialization only on macOS/arm64, Torch 2.13.0. It excludes model execution, CUDA, serving, reward, optimizer, publication and recovery. The shared host was not reserved. No full-model or full-RL speedup follows from these ratios.

Three independent process starts; one warmup per arm/length, then cumulative/final/final/cumulative. Tokens, scores and depths compare exactly within every block. Times are geometric means over observations. The interval resamples the three process-level log ratios 10,000 times; with only three clusters it is descriptive and does not establish a stable speed claim.

| Response tokens | Cumulative (ms) | Terminal (ms) | Collector ratio | 95% cluster interval | Score elements before → after |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 3.062 | 0.115 | 26.67× | [23.22, 32.59] | 8,256 → 128 |
| 512 | 49.716 | 0.245 | 202.94× | [181.23, 230.30] | 131,328 → 512 |
| 2048 | 693.714 | 0.911 | 761.84× | [641.91, 914.42] | 2,098,176 → 2,048 |
| 8192 | 12621.990 | 5.077 | 2486.06× | [1751.78, 3161.43] | 33,558,528 → 8,192 |

Full official-model E2E, GPU asynchronous collection, high-concurrency serving, memory savings and reward convergence: **NOT_RUN**.

Validation: 171 distinct CPU tests passed across terminal output, engine, speculative decoding, async pipeline, profiler and serving suites. The interface suites used existing pure-Python aiohttp dependencies because an iCloud-evicted compiled extension blocked the default import. No packages were changed. GPU async test remains unrun.
