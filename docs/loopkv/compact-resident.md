# Compact KV in resident asynchronous execution

Compact allocations now reserve their physical record IDs during CPU preparation
and publish them through the same event-owned routing banks as resident state.
Each six-integer descriptor contains the request slot, depth, position, record-map
pointer, map width and physical record. Padding publishes no record. The copy
stream waits for allocation initialization; the execution stream waits for the
bank, and request retirement waits for its final reader before recycling records.

Recurrent Graph entries copy the map pointer, width and query depth into stable
device tensors on every replay. Attention consumes the compact map directly.
Prefill keeps the existing eager record path; compact mode does not allocate the
rectangular static workspace or page-table pool. Worst-future reservation and
optional past-skip credit reclamation retain their existing rules.

| Validation | Result |
|---|---|
| CPU loopkv suite | 102 passed, 2 CUDA cases deselected, 36 subtests passed |
| Cancellation and request ID reuse | Native/alias/compact/credits exact under normal and held coda readiness |
| Complete logical schedule, opposite coda readiness | Native/alias/compact/credits exact over two request lifetimes, three depth policies |
| CUDA bank publication, delayed reader, padding and Graph replay | Prepared; not run |
| Official checkpoint and performance | Not run |

The finite CUDA gate is `python -m experiments.loopkv.async_gate --compact --out
RESULT.json`. It compares seven execution paths at B1/4/16/32/64/128 and C2B,
with mapped-host and H2D descriptors, two lifetimes, P33/P34 and six output tokens.
The separate `replay_gate --compact` runs real delayed-gate computation under
captured admission, batch and coda-delivery decisions.

| Matched E2E | Native tokens/s | Compact tokens/s | Speedup |
|---|---:|---:|---:|
| Official resident async/Graph | — | — | Not measured |

Preemption, prefix sharing, transfer, speculative rollback, FlashAttention and
synchronous static-buffer/Graph combinations remain outside this cache contract.
