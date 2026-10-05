# Ouro finite HTTP diagnostic

All24 official-model HTTP arms drain successfully. Non-Graph alias and compact
paths match native output tokens and exits. Natural Graph runs retain output
and exit differences, so this experiment does not establish Graph equivalence.

| B / C | Alias async token / exit diffs | Compact async token / exit diffs | Alias Graph token / exit diffs | Compact Graph token / exit diffs |
|---|---:|---:|---:|---:|
| 4 / 8 | 0 / 0 | 0 / 0 | 0 / 4 | 0 / 0 |
| 16 / 32 | 0 / 0 | 0 / 0 | 0 / 5 | 0 / 5 |
| 32 / 64 | 0 / 0 | 0 / 0 | 3 / 44 | 0 / 40 |
| 64 / 128 | 0 / 0 | 0 / 0 | 4 / 63 | 3 / 56 |

Each difference is counted per request against the native path in the same
execution mode. Measured Graph candidate totals are10 token-different and217
exit-different requests across alias and compact. The observed counts do not
isolate numerical changes from differences in naturally selected batching.
Graph cache coverage also differs: at B32 native captures one new entry and
replays1,341 times without fallback; alias replays306/falls back376, and compact
replays327/falls back358. These are recorded behaviors, not matched-work timings.

The frozen source is `37387d1c022dc157f2bf0bb89dd9e00cdac65dec`. It runs official
Ouro-1.4B revision `574fa66cb8bf5abdc979642d01cf2b79b16bfab1` with BF16/Triton,
delayed threshold0.5/min2/max4, greedy seeds17+i and EOS disabled. Each finite
cohort includes controlled P32/64/128/256 inputs and D16/32/64/96 output limits.
All closed-loop arrivals begin together, concurrency is2B, and one full-cohort
warmup precedes each measured arm. Weights load once; engine/cache state is
isolated between arms.

All1,392 measured requests /72,384 output tokens and the same number of warmup
requests pass independent raw-log validation: frozen input hashes, complete
request IDs, output lengths, finish reasons, monotonic timestamps, exact chunk
timestamp expansion, full-drain bounds and recomputed divergence counts.
Recorded SLO goodput is unavailable because no SLO was registered.

| HTTP speed comparison | Native tokens/s | Candidate tokens/s | Speedup |
|---|---:|---:|---:|
| B4/16/32/64, C2B diagnostic | — | — | Not qualified |

The other GPU ran a calibration task, so observed elapsed values and nearest-rank
latencies are diagnostic data only. None of these cohorts meets the registered
2,000-request/120-second formal minimum. This pretokenized localhost TCP adapter
shares a client/server event loop; production text serving and tokenizer costs
are outside its scope. Formal paired HTTP/SLO testing remains outstanding.

All four children/controllers/SSH naturally exited0. The whole GPU1 window was
returned after offbox audit, empty compute/model-reader checks and original-lock
verification. [Raw hashes and diagnostic counters](evidence/http-diagnostic-v2.json)
retain measurements without presenting them as speed gains. Archive SHA256:
`c5c77ea7e64e0dd7e7e4afc25d45ab4e15a14a27bc34a80525b6a368663fddb2`.
