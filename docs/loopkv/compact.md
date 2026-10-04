# Compact records: first implementation

Status: CPU/CUDA gates and full official-checkpoint token/exit parity passed; first timed pilot regresses.
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

## Official Ouro-1.4B diagnostic

A full-weight P128/D256, B32/C64 natural-gate run completed 64 requests and
16,384 output tokens. Prompt IDs, output IDs and exit depths exactly match the
native reference, including all 36,254 executed decode loops. The run is
instrumented diagnostic evidence, not a throughput comparison.

| Metric | Rectangular storage | Compact records |
|---|---:|---:|
| Peak allocator-owned payload units | 9.000 GiB | 6.056 GiB (33,072 records) |
| Reserved payload arena | 9.000 GiB | 9.000 GiB |
| Maximum active requests | 32 | 32 |
| Records remaining after drain | — | 0 |
| Free admission blocks after drain | 3,072 | 3,072 |
| E2E speedup, separate timed pilot | Reference | 0.974× median of 3 pairs |

Rectangular ownership is derived from the observed peak 32 resident requests,
each reserving 96 pages. Compact ownership counts actual assigned records.
These metrics count allocator-owned units, not only initialized bytes. Compact
persistent GPU address metadata peaked at 441,728 bytes; runtime peak allocated
memory was 12,574,126,080 bytes. Reserved memory did not improve in this version.
[Raw summary and hashes](evidence/compact-official-main-v1.json) preserve the
measurement scope. [Additional CUDA checks](evidence/compact-gpu-v2.json) verify
all unused payload records remain poisoned and stale descriptors are rejected
after request-ID reuse.

## First timed pilot

All six fresh processes completed the full 64-request workload after their own
complete warmup. Every pair has exact prompt IDs, output IDs and exit depths.
The frozen source is `74ea87a5fdbb9fa7b0a9736e9a4fcfbf977b8b66`; model,
inputs, hardware and gate match the diagnostic above. This table compares the
whole compact stack to Native-S. The incremental alias-to-compact comparison
remains outstanding.

| Pair | Order | Native tokens/s | Compact tokens/s | Speedup |
|---|---|---:|---:|---:|
| 1 | compact / native | 699.31 | 685.80 | 0.981× |
| 2 | native / compact | 701.27 | 682.72 | 0.974× |
| 3 | native / compact | 692.54 | 660.33 | 0.953× |

Median paired ratio is **0.974×**, range 0.953–0.981×. It is a throughput
regression despite fewer occupied payload records; reservation remains 9 GiB.
Three pilot pairs do not establish a formal confidence interval or serving
latency. [Raw comparisons](evidence/compact-pilot-pairs.json) retain all pairs.
[Rectangular Nsight attribution](profile.md) motivates the subsequent reader
experiment, but is not a profile of compact storage.
