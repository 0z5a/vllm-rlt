# LAST_EXITED alias: current validation

Status: rectangular storage prototype; the first official-checkpoint pilot regresses.
Reference: `ecb1f8b505b7e831815b40aec3b4598619cca23a`.

When a token exits early, the candidate retains every executed depth and changes
only the source chosen for later queries. Skipped-depth payload remains unwritten.
The physical pool is still rectangular, so payload reservation savings are zero.

## Correctness

| Check | Result | Scope |
|---|---:|---|
| CPU runtime, cache and engine suite | 91 passed, 1 CUDA skip | Includes 36 address subcases; Torch 2.13.0 |
| CUDA attention outputs | 46/46 bitwise equal | RTX 5090, BF16, Torch 2.12.1+cu130 |
| CUDA logical K/V exports | 64/64 exact | Full-depth prompt and heterogeneous decode exits |
| Skipped-depth poison | 18/18 retained | Physical written flags remain false |
| Official Ouro-1.4B token/exit parity | 64/64 requests, 16,384 tokens per run | B32/C64/P128/D256, natural gate; full pinned weights |
| Standalone checkpoint reader | 3/3 dtypes exact | FP32/FP16/BF16 parameter bits against safetensors |
| CUDA tiny-model token/exit parity | 6/6 batch cases | B1/4/16/32/64/128, C=2B, up to 256 requests |

The CUDA tests use a two-layer randomly initialized model. They validate the real
CUDA execution path but do not establish pretrained quality or E2E speed.
`tests/loopkv/test_alias_runtime.py` contains the payload/attention and CPU engine
regressions. The standalone CUDA gate and raw evidence are retained in the task
artifact directory and will accompany the completed model measurements.

## Speed comparison

| Checkpoint / workload | Baseline tokens/s | Alias tokens/s | Speedup | Status |
|---|---:|---:|---:|---|
| Ouro-1.4B, P128/D256, B32/C64, natural gate | 704.88 | 679.28 | 0.964× | Median of 3 pilot pairs; regression |
| Ouro-2.6B and Thinking checkpoints | — | — | — | Pending |
| Nanbeige4.2-3B full-depth control | — | — | — | Pending |
| Huginn-0125 full-depth control | — | — | — | Pending model-adapter integration |

The throughput above is finite-batch engine E2E after one complete warmup;
model loading is excluded. Both arms submit the same 64 requests, include queueing
and full drain, and generate exactly 256 tokens per request with EOS ignored.
This is a controlled workload, not HTTP serving or a quality estimate.
No capacity, quality, all-model completion, or statistically confirmed speedup
claim is made.
Compact allocation, optimized readers, async/Graph, online admission/restore,
quality evaluation and external baselines remain open. Natural and delayed gate
tracks must use their own paired references; fixed depth remains a negative control.

## First pilot evidence

Official `ByteDance/Ouro-1.4B` revision
`574fa66cb8bf5abdc979642d01cf2b79b16bfab1`; BF16, original Ouro gate at 0.5,
eager Triton, four CPU threads. Each arm starts a fresh process from frozen
`37630710efb926fae8bdd6b576f77c28a3952e63`, performs one full warmup, then
measures submissions through full GPU drain without per-step diagnostics.
Arm order is balanced and shuffled with seed 1705. GPU UUID:
`GPU-209bed64-f11a-c1f3-fe87-8a69538d8d79`; RTX 5090, Torch 2.12.1+cu130.

| Pilot pair | Order | Native tokens/s | Alias tokens/s | Alias speedup |
|---|---|---:|---:|---:|
| 1 | alias / native | 704.88 | 679.28 | 0.964× |
| 2 | native / alias | 694.98 | 689.39 | 0.992× |
| 3 | native / alias | 707.74 | 677.62 | 0.957× |

All paired prompt IDs, generated token IDs and exit depths match exactly.
Speedup is native elapsed time / alias elapsed time; its median is 0.964×
(range 0.957–0.992×). Three pilot pairs do not replace seven formal pairs or
serve as a P99 sample. The regression remains in the report.
[Raw pair summaries](evidence/ouro-1.4b-pilot-pairs.json) and
[M0 accounting](evidence/m0-main-summary.json) retain both arms.

The independent diagnostic run observed effective recurrent batch mean 29.67,
P10/P50/P90 = 24/32/32 and maximum residency 32. Decode exits were 12,706 at
loop 2 and 3,614 at loop 3 (mean 2.2214). Reference promotion payload was
5,706,743,808 algorithmic bytes across the whole drained run; this is not
measured DRAM traffic or a live-memory peak. Both arenas reserve 9 GiB.
Removing those copies did not offset the prototype's remaining costs here.
Profile attribution and the compact-layout experiment are the next gates.

## Reproduction

```sh
python -m pytest tests/loopkv tests/test_kv_cache.py tests/test_prepared_kv.py tests/test_engine.py -q
CUDA_VISIBLE_DEVICES=0 python -m pytest tests/loopkv/test_alias_runtime.py --run-gpu -m gpu -q
```

The default capture entry is diagnostic and includes trace-collection overhead.
Only `--measure` disables step tracing and warms up a complete workload before
timing all submissions through full drain. It measures finite-batch engine E2E;
HTTP serving and long-running latency measurements are separate. Formal pairs need frozen inputs,
independent initialization, matched policy/backend/work and recorded GPU UUIDs.
