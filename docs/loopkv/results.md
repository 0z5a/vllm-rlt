# LAST_EXITED alias: current validation

Status: rectangular storage prototype; official-checkpoint performance pending.
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
| Official Ouro-1.4B token/exit parity | 2/2 requests, 16 tokens | B2/P8/D8, natural gate; full pinned weights |
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
| Ouro-1.4B, natural gate, multi-B/C | — | — | — | Smoke passed; main-point qualification in progress |
| Ouro-2.6B and Thinking checkpoints | — | — | — | Pending |
| Nanbeige4.2-3B full-depth control | — | — | — | Pending |
| Huginn-0125 full-depth control | — | — | — | Pending model-adapter integration |

No throughput, memory-capacity, quality, or all-model completion claim is made.
Compact allocation, optimized readers, async/Graph, online admission/restore,
quality evaluation and external baselines remain open. Natural and delayed gate
tracks must use their own paired references; fixed depth remains a negative control.

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
