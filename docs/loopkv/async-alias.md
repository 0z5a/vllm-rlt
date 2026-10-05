# Resident routing for rectangular aliases

Status: CPU and finite tiny-model CUDA scheduling/lifetime tests pass; official-weight qualification is pending.
The parent change enables rectangular aliases with the existing resident asynchronous
runner for delayed or frozen-trace policies. Compact async, static workspaces,
transfer, prefix reuse and preemption remain unsupported. The Graph extension is
documented in [alias-graphs.md](alias-graphs.md). Huginn retains
its existing synchronous-only restriction.

Routing banks now own per-row tables for every loop depth and snapshot query
depths. The existing metadata kernel gathers them from resident request tables.
Their lifetime follows the bank's final-reader event, so a bank cannot be
overwritten while its previous core still reads it. Async attention uses the
general canonical-depth resolver for every tile until resident first-alias
positions are implemented; it does not reuse the synchronous prefix shortcut.

New allocations record an event after their exit-map reset. The first routing
bank waits for that event on the copy stream before publishing its ready event
to the core stream. Batched finalized-depth publication waits for the request's
last core event and records the following boundary event. Existing request
retirement waits for the final reader before freeing the allocation and slot.

| Validation | Current result |
|---|---|
| CPU cache/model/async regression before explicit reset event | 162 passed, 43 CUDA/official-input skips, 36 subtests |
| CPU async/reset regression after explicit reset event | 31 passed, 2 CUDA skips |
| Capture workload counters | 3 CPU tests passed; sync/async mixed-depth accounting and call-weighted percentiles |
| Frozen mixed-depth trace with held coda delivery | Exact output/exit objects versus native synchronous |
| Cancel and reuse same request ID across three rounds | All three requests complete; all block credits return |
| CUDA UVA/H2D banks and native Graph control | 48 tiny-model arms pass at B1–128/C2B, two request rounds |
| CUDA delayed allocation reset and final bank reader | Both H2D and mapped-host modes pass with explicit cross-stream delays |
| Official Native-D/Native-A/candidate-A quality and E2E | Pending |

| Fixed-policy throughput comparison | Native-A tokens/s | Alias-A tokens/s | Speedup |
|---|---:|---:|---:|
| Official checkpoint E2E | — | — | Not measured |

The CUDA gate naturally exited0 on RTX5090/Torch2.12.1+cu130 at source
`28fd63584f68b0574b1a4d6e51ce4b312cf87eda`. It covers both mapped-host and H2D descriptors, B1–128/C2B, mixed-depth
traces, cancellation/reuse, and a native Graph control. All output/exit objects match native synchronous
execution, and all block credits and state slots return after each round.
[All48 cases, source and raw hash](evidence/async-cuda-v1.json). These tiny-model
correctness results do not establish official-model quality or performance.

Future timed captures collect the same CPU submission counters in both arms:
recurrent batch histograms, rows per depth, prefill tokens and observed residency.
They add no device reads. The reported residency is sampled after each engine
step. Batch percentiles weight core calls and use the nearest-rank definition.
Submitted rows include work discarded before output delivery; exit-depth sums
describe emitted outputs and are reported separately.

Official Ouro1.4B delayed-policy B1–128/C2B diagnostic: all native/alias async request objects match synchronous native across 490 requests per path. Graph paths retain exit-depth differences. [Full scope, counters and provenance](delayed-official.md). No timing or quality claim.
