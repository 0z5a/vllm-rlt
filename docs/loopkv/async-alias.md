# Resident routing for rectangular aliases

Status: CPU scheduling/lifetime tests pass; CUDA qualification is queued.
This branch enables rectangular aliases with the existing resident asynchronous
runner for delayed or frozen-trace policies. Compact async, static workspaces,
Graph, transfer, prefix reuse and preemption remain unsupported. Huginn retains
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
| Frozen mixed-depth trace with held coda delivery | Exact output/exit objects versus native synchronous |
| Cancel and reuse same request ID across three rounds | All three requests complete; all block credits return |
| CUDA UVA/H2D banks and native Graph control | Queued in `experiments.loopkv.async_gate` |
| CUDA delayed allocation reset and final bank reader | Queued with explicit cross-stream delays |
| Official Native-D/Native-A/candidate-A quality and E2E | Pending |

| Fixed-policy throughput comparison | Native-A tokens/s | Alias-A tokens/s | Speedup |
|---|---:|---:|---:|
| Official checkpoint E2E | — | — | Not measured |

CPU scheduling tests cannot establish CUDA event ordering. The forthcoming
CUDA gate covers both mapped-host and H2D descriptors, B1–128/C2B, mixed-depth
traces, cancellation/reuse, and the native Graph control before official runs.
