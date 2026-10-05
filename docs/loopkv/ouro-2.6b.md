# Official Ouro 2.6B cache diagnostic

The complete 24-arm diagnostic naturally exited0 at source
`f2b6add3270a02b599f4d2da0e76445e0e8f946d` on RTX5090/Torch2.12.1+cu130.
The official checkpoint revision is `1ed04250da1a9936042725d302e81c8fa2ab5abd`;
all nine assets were independently verified before one weight load. Each arm
uses a fresh engine, original Ouro threshold0.5/min2/max4, greedy ignored EOS,
seed17, P128/D128 and C2B. These repeated fixtures are not independent quality
questions. Concurrent non-timing work was allowed; elapsed times are not speed evidence.

All 1,960 requests and 250,880 output tokens completed. Alias and compact without
reclaimed credits match native complete request objects at every batch limit.
Credits also match at B1–64. At B128, all token sequences still match, but credits
increase observed residency and 252/256 request exit sequences differ. The
submitted recurrent work increases by125 rows; no causal explanation or
same-work speedup is established.

| B limit / C | Native actual peak residents | Credits actual peak residents | Native / credits recurrent rows | Credits exit-different requests | Token-different requests (all candidates) |
|---|---:|---:|---:|---:|---:|
| 1 / 2 | 1 | 1 | 555 / 555 | 0 | 0 |
| 4 / 8 | 4 | 4 | 2,337 / 2,337 | 0 | 0 |
| 16 / 32 | 16 | 16 | 9,360 / 9,360 | 0 | 0 |
| 32 / 64 | 32 | 32 | 18,740 / 18,740 | 0 | 0 |
| 64 / 128 | 64 | 64 | 37,451 / 37,451 | 0 | 0 |
| 128 / 256 | 64 | 75 | 74,833 / 74,958 | 252 | 0 |

The payload arena is capped at24GiB. B128 is a configured scheduler limit: the
worst-future rectangular reservation admits64, not128, concurrent requests.
All B64/B128 arms reserve31,570,526,208 CUDA bytes. Credits at B128 reach75
residents with54,893 peak live records; their live and reserved record counts
both return to zero, as do every arm's pending cache ownership obligations.
All4,096 blocks return after the largest cases. Actual batch histograms,
per-depth work, observed residency and each raw request hash are retained.

| Fixed-policy E2E speed comparison | Native tokens/s | Candidate tokens/s | Speedup |
|---|---:|---:|---:|
| Alias / compact / credits | — | — | Not measured in this diagnostic |

The archive SHA256 independently matches locally and remotely:
`fc1695ca95c223d9125d92fad9925996f1eefc3dacc8be6b45d39a0ce56ec970`.
The controller and child naturally exited0; their PIDs disappeared, GPU1 was
compute-empty and the original GPU1 lock was independently acquired/released.
[Full provenance, counters and request hashes](evidence/ouro-2.6b-official-v1.json).

Quality datasets, delayed/Graph paths, longer contexts and formal paired speed
measurements remain outstanding. The model inputs therefore remain required.
