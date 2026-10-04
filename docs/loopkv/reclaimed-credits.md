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
| CUDA pressure gate | Exact payloads, all 16 requests and exits match; peak residency 2 / 2 / 3 for native / fixed compact / reclaimed credits; all records and credits returned |
| Official Ouro-1.4B fixed6GiB arena, B limit64/C128, P128/D128 | All128 requests complete; resident peak32→38; output tokens match, exit depths differ; timing not qualified |

The CPU pressure tests establish reservation and lifetime behavior, not a
pretrained-model throughput or quality result. The standalone CUDA gate is
`python -m experiments.loopkv.credit_gate --out RESULT.json`. Full-checkpoint
timing must include any admission, scheduling and numerical effects of changing
the number of resident requests.

The CUDA run naturally exited zero at source
`cf70584d60913e1664439fb1b4acf603e2dff105`, RTX 5090, Torch 2.12.1+cu130.
It deliberately switches from early-exit histories to full-depth future tokens.
[Raw gate](evidence/credit-gpu-v1.json). This is synthetic-weight correctness;
it establishes neither official-model output equivalence nor a speed benefit.

## Official checkpoint diagnostic

Three separate processes at the same source loaded the verified official
Ouro-1.4B pin `574fa66cb8bf5abdc979642d01cf2b79b16bfab1`. They used the same
128 requests, P128/output limit128, B limit64, original greedy gate0.5 and a
6GiB payload arena (2048 blocks). All three naturally exited zero and returned
all2048 block credits. The compact variants had zero remaining live records.

| Metric | Native | Compact staging | Reclaimed credits |
|---|---:|---:|---:|
| Completed requests / output tokens | 128 / 16,384 | 128 / 16,384 | 128 / 16,384 |
| Maximum resident requests | 32 | 32 | 38 |
| Requests with exact output tokens versus native | 128 | 128 | 128 |
| Requests with exact exit sequences versus native | 128 | 128 | 0 |
| Cached decode loop executions | 36,391 | 36,391 | 36,452 |
| Peak occupied physical payload bytes | — | 4,790,943,744 | 5,523,505,152 |
| CUDA allocator peak reserved bytes | 9,554,624,512 | 9,554,624,512 | 9,554,624,512 |

Compact staging preserves the complete request objects, including exits. Credit
reclamation changes admission and observed batch sizes; its128 output-token
sequences still match, but every request has at least one different exit. This
is not an exact-workload speed comparison. The link between changed batching
and exit decisions requires a matched-admission control before attributing the
difference. The128 requests use a repeated frozen prompt fixture, not128
independent quality questions.

These captures include per-step diagnostics and overlapped other non-timing
work. Their seconds are retained in the raw summary but must not be used as a
speedup result. [Manifests, natural exits, batch histograms and all mismatch
positions](evidence/credit-official-v1.json). Broader quality, quiet timing and
sustained service remain outstanding.

| Official checkpoint speed comparison | Native tokens/s | Candidate tokens/s | Speedup |
|---|---:|---:|---:|
| Reclaimed credits under fixed memory | — | — | Not measured |
