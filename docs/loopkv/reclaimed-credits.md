# Reclaim confirmed skipped-version credits

The compact allocator can now return admission credits for versions that a
finalized historical token provably did not execute. Each future token still
reserves every loop depth. This can admit additional requests without predicting
future exits or relying on preemption. The option is disabled by default and
requires compact LAST_EXITED storage.

The reservation includes allocated records, all possible future records and
page-rounding slack. Finalization releases only skipped `(depth, position)`
entries that have no allocated record. Preparing an unwritten entry retains its
credit. Freeing a request returns its exact remaining credits and physical
records; a second free cannot release them again. Finalized sources remain
immutable. Async execution and Graph remain unsupported by this allocator.

| Pressure check | Result |
|---|---|
| 96-record pool, two requests each initially needing 56 credits | Second request initially rejected |
| First request finalizes six positions at one of four depths | 18 confirmed unused credits returned; second request admitted |
| Every remaining position of both requests executes all four depths | 86 physical records fit within 94 committed credits and 96 total records |
| Prepared but unwritten skipped versions | Retain their credits |
| Tiny CPU engine, B limit 8 / C16; first eight requests R2, remaining eight R4 | All outputs and exits match; peak residency exceeds the native/fixed-credit limit of two |
| CUDA pressure gate | Pending |
| Official checkpoint capacity / timing | Pending |

The CPU pressure tests establish reservation and lifetime behavior, not a
pretrained-model throughput or quality result. The standalone CUDA gate is
`python -m experiments.loopkv.credit_gate --out RESULT.json`. Full-checkpoint
timing must include any admission, scheduling and numerical effects of changing
the number of resident requests.

| Official checkpoint speed comparison | Native tokens/s | Candidate tokens/s | Speedup |
|---|---:|---:|---:|
| Reclaimed credits under fixed memory | — | — | Not measured |
