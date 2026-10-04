# Compact records: first implementation

Status: CPU validation passed; CUDA and official-checkpoint results pending.
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
| CUDA general reader | Pending | No performance claim |

Reproduce the standalone GPU gate with
`python -m experiments.loopkv.gpu_gate --storage compact --out result.json`.
The official checkpoint harness accepts `--compact`; `--measure` remains the
finite-batch E2E mode, and neither mode stands in for HTTP serving.
