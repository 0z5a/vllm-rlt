# Compact records: first implementation

Status: CPU and CUDA gates passed; official-checkpoint results pending.
Incremental reference: rectangular alias branch at
`9894e0e` (the exact parent is recorded in Git).

`CacheConfig(compact_last_exited=True)` selects a synchronous eager cache.
The arena retains the native `[page, layer, offset, head, dim]` strides, but
individual offsets across all layers are allocated as records. Prompt and decode
share this pool without rounding their boundary to a page. Each request owns an
`[R + 1, capacity]` int32 map: executed physical record IDs followed by exit depth.
Skipped version entries remain -1; the reader resolves the canonical executed
depth and loads its payload directly. There is no dense CUDA reconstruction.

Admission reserves the full worst-future block budget while record allocation
occurs only when a traversal is prepared. This version therefore cannot increase
online admission. It demonstrates reusable physical records inside an unchanged
arena; neither reserved GPU bytes nor maximum admitted concurrency is claimed
to improve. Those are separate capacity experiments.

The current scope is one execution stream. Allocation identity rejects stale
descriptors after request-ID reuse. The engine rejects async, Graph, speculative
rollback, transfer, prefix sharing and preemption before constructing the cache.
Cross-stream retirement and stable graph metadata remain future work.

| Check | Result | Scope |
|---|---:|---|
| Runtime/cache/engine regression suite | 100 passed, 2 CUDA skips, 36 subtests | CPU, Torch 2.13.0 |
| Heterogeneous executed versions | 23 records vs 32 logical versions | Three full-depth prompt tokens plus five decode tokens |
| Skipped versions | No allocated record; device-map entry stays -1 | CPU payload/attention equality |
| Free and reuse | All records and credits returned; stale descriptor rejected | Repeated request ID |
| Full tiny-model engine | B1/4/16, C=2B token/exit parity | Conservative admission and refills |
| CUDA general reader | 46 bitwise attention, 64 exact payload checks | RTX 5090, BF16; no dense K/V reconstruction |
| CUDA skipped entries | 18 checks passed | No physical record; map entry remains -1 |
| CUDA full tiny-model engine | B1/4/16/32/64/128, C=2B all exact | Up to 256 requests; tiny weights only |
| Rectangular alias on updated kernel | All original CUDA gates passed | Shared-kernel regression check |

Reproduce the standalone GPU gate with
`python -m experiments.loopkv.gpu_gate --storage compact --out result.json`.
The official checkpoint harness accepts `--compact`; `--measure` remains the
finite-batch E2E mode, and neither mode stands in for HTTP serving.

Raw CUDA evidence: [compact](evidence/compact-gpu-v1.json) and
[rectangular regression](evidence/compact-branch-alias-regression-v1.json).
