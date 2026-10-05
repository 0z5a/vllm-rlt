# Frozen family serving inputs

Five family-specific performance banks are prepared from calibration tokens only.
Each trace has512 requests with unique IDs. These concatenated token-bank inputs
are synthetic performance workloads and are not used to assess generation quality.
All files were independently hash-checked; every input plus output budget fits
its declared context. [Preparation](family-serving-v1/prepare.py) and
[validation](family-serving-v1/validation.json) identify the exact artifacts.
Full request files remain local under `serving-families-v1`, identified by hashes.

| Family | Fixed loops | Primary input/output | Long input/output | Decode input/output | Mixed decode depths |
|---|---:|---:|---:|---:|---|
| Huginn | 32 | 1024/128 | 3968/128 | 512/512 | 8,16,32; native qualification pending |
| Nanbeige | 2 | 1024/128 | 4096/128 | 512/512 | 1,2; native qualification pending |
| HRM-Text | 2 H cycles, each3 L cycles | 1024/128 | 3968/128 | 512/512 | Unsupported by registered fixed-depth path |
| LoopFormer | 8 | 512/128 | 896/128 | 512/512 | Unsupported by registered fixed-depth path |
| Parcae | 8 | 1024/128 | 1920/128 | 512/512 | Unsupported by registered fixed-depth path |

Every family also has256/32 smoke input and a shuffled mixed-length trace.
Mixed-length is not evidence of mixed-depth execution. The model's complete
registered recurrence is used for prefill; requested depths apply to decoding.
Context-limited configurations have their own IDs and are not reported as the
plan's4096/128 workload. OOM points keep their original concurrency and batch cap.

Unit-rate exponential arrivals (seed37) are frozen for512 requests. An actual
open-arrival run must first lock its offered rates using independent BF16
measurements, divide these times by the selected req/s, and freeze SLO and the
paired-trial schedule before candidate results. Rates and SLO are not yet frozen.
Formal comparisons still require G2, at least512 requests, approximately60s
steady state, five independent pairs, queue traces, and all failure accounting.

No performance measurement is claimed by this preparation. Existing Ouro1.4B and
Ouro2.6B continue to use the previously frozen shared Ouro prompt bank.
