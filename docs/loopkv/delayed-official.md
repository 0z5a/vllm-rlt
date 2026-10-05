# Official Ouro 1.4B resident-path diagnostic

All 36 arms completed at frozen source `80a9972a14acacc7329f064ef5f3901eaf8e683c`:
B1/4/16/32/64/128, C2B, P128/D128, greedy ignored EOS, seed17, delayed exit
threshold0.5/min2/max4. Each batch compares native/alias synchronous, asynchronous
and Graph execution. One verified official checkpoint was loaded once; each arm
received a fresh engine and cache. This is a repeated-fixture correctness diagnostic,
not a quality dataset or timing experiment. Other non-timing work was allowed.

Every alias arm produced exactly the same token sequences as its matching native
execution mode: 490 requests per path, 2,940 requests and 376,320 output tokens in
total. Alias synchronous/asynchronous request objects also match completely.
Native asynchronous request objects match native synchronous at every batch.
Graph output tokens match, but some exit depths differ, including between the
native and alias Graph paths. Their recurrent work is therefore not interchangeable
for a same-work performance claim.

| B / C | Alias Graph requests with different exit sequences vs native Graph | Native / alias recurrent rows | Native / alias mean recurrent B | Native / alias Graph fallbacks |
|---|---:|---:|---:|---:|
| 1 / 2 | 0 | 821 / 821 | 1.000 / 1.000 | 0 / 0 |
| 4 / 8 | 0 | 3,289 / 3,289 | 2.987 / 2.987 | 0 / 0 |
| 16 / 32 | 14 | 13,166 / 13,165 | 8.830 / 8.661 | 0 / 0 |
| 32 / 64 | 22 | 26,320 / 26,317 | 23.755 / 22.551 | 798 / 784 |
| 64 / 128 | 107 | 52,675 / 52,650 | 37.279 / 42.154 | 855 / 858 |
| 128 / 256 | 0 | 105,594 / 105,594 | 84.408 / 84.408 | 927 / 927 |

The counters include actual submitted recurrent rows. Each arm's row total was
independently checked against the sum of recorded decode exit depths (excluding
the first output, produced from full-depth prefill). The evidence includes batch
histograms, nearest-rank quantiles, per-depth work, Graph captures/replays/fallbacks,
and each raw request file's SHA256. A Graph flag is not evidence that every batch
used replay. B128 alias Graph reserved 29,848,764,416 CUDA bytes; all 8,192 blocks
returned after drain.

| Matched-policy E2E speed comparison | Native tokens/s | Alias tokens/s | Speedup |
|---|---:|---:|---:|
| Synchronous / asynchronous / Graph | — | — | Not measured in this diagnostic |

Changing effective batch and exit depths are observed together; this does not
establish the cause. A serialized matched-schedule numerical oracle, fixed logical
schedule replay, and quality evaluation remain necessary before interpreting a
speed result. Existing synchronous pilot regressions are unchanged by this result.

The controller naturally exited0. GPU0 was independently observed compute-empty
and its original lock acquired/released after completion. The collected archive's
local and remote SHA256 both equal
`d43826b5b0b235c5a4231b8d61cebfc366bb88864291e4be28a8679b1b08c492`.
[Checkpoint pin, frozen plan, all counters and raw request hashes](evidence/delayed-official-v1.json).
