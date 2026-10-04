# Direct prefix and alias tail

Status: CPU/CUDA and official-checkpoint quality passed. Five shapes completed
three independent speed pairs each; no stable throughput gain was established.

The rectangular alias cache now tracks the first skipped position for each
request and query depth. Before that position, every needed version has its own
physical slot. Traversals that cannot see an alias use the original direct
reader and prepare no extra depth metadata. For other traversals, complete
32-token tiles before the first alias use direct page tables; remaining tiles
use the existing canonical-depth resolver. Online softmax order is unchanged.
The boundary tile retains the general resolver, including its valid-token mask.

Depth tables, query depths and first-alias positions share one staging transfer.
A full-depth finalization leaves the initialized -1 exit sentinel, which already
means to use the current depth. Host finalization and immutability checks remain.
The compact allocator is unchanged by this optimization. Async and Graph remain
unsupported by the experimental cache modes.

| Validation | Result |
|---|---|
| CPU cache/runtime/engine regression | 101 passed, 2 CUDA skips, 36 address subtests |
| CUDA attention against original direct reference | 46/46 bitwise equal |
| CUDA mixed prefixes crossing the 32-token boundary | 8/8 bitwise equal |
| CUDA payload / skipped poison | 64/64 exact; 18/18 retained |
| Tiny CUDA full engine, B1/4/16/32/64/128, C=2B | All output tokens and exit depths exact |
| Text-stop streaming boundary tests | 2 passed; separate quality harness |
| Official Ouro-1.4B GSM8K100, EOS and text stops | 100/100 requests and11,473 tokens/exits/stops exact; strict64/100 |

CUDA runtime source `87504bb1d213e6396a0b0e7a9f8ab1fcf3ebe7bf`, RTX 5090,
Torch 2.12.1+cu130 and Triton 3.7.1. [Raw gate](evidence/hybrid-gpu-v1.json).
These tiny-weight tests cover routing and cache lifetime, not pretrained quality.

| Ouro-1.4B P128/D256 B32/C64 implementation | Native tokens/s | Candidate tokens/s | Speedup | Evidence |
|---|---:|---:|---:|---|
| Original rectangular alias | 704.88 | 679.28 | 0.964× | 3 pilot pairs; regression |
| Compact records | 699.31 | 682.72 | 0.974× | 3 pilot pairs; regression |
| Direct-prefix alias reader | 697.40 | 704.09 | 1.006× | 3 pilot pairs; range 0.977–1.010× |

The two completed pilots are distinct campaigns. Their numbers do not form an
incremental alias-to-compact comparison. [The profile](profile.md) motivates
this reader change; its summed kernel durations are not an E2E speed forecast.

## Batch scaling pilot

All 30 fresh-process arms and their controller exited naturally with code zero.
Every paired prompt, output token and exit-depth sequence matches exactly.
Source `7ca67d64599b421d0128f062c48325cc2a857b36`, official Ouro-1.4B pin
`574fa66cb8bf5abdc979642d01cf2b79b16bfab1`, GPU0 RTX 5090 UUID
`GPU-209bed64-f11a-c1f3-fe87-8a69538d8d79`; GPU1 and heavy I/O remained reserved
and idle for this entire timing campaign. Each arm completes one full warmup
before timing all request submissions through drain. Greedy original gate 0.5,
minimum two / maximum four loops, EOS ignored, C=2B and private fixed allocations
are identical within each pair. B is the scheduler limit; these timing captures
omit per-step traces and therefore do not establish the observed B distribution.

| B limit / C | P / output limit | Native median tokens/s | Hybrid median tokens/s | Median paired speedup | Paired range |
|---|---|---:|---:|---:|---:|
| 8 / 16 | 128 / 128 | 218.18 | 216.02 | 0.9901× | 0.9253–1.0450× |
| 16 / 32 | 128 / 128 | 379.65 | 369.45 | 0.9604× | 0.9600–0.9978× |
| 32 / 64 | 128 / 128 | 600.61 | 575.56 | 0.9583× | 0.9472–0.9871× |
| 64 / 128 | 128 / 128 | 894.54 | 807.57 | 0.8965× | 0.8905–0.9587× |
| 32 / 64 | 128 / 256 | 697.40 | 704.09 | 1.0057× | 0.9765–1.0096× |

Speedup is the median of paired native/candidate elapsed-time ratios; it need
not equal the ratio of the two independent median throughput columns. The
P128/D256 point straddles parity, while the four shorter-generation points have
median regressions. Three pairs are screening evidence, not a confidence interval
or a stable tail-latency result. This is finite engine E2E, not HTTP serving.
These comparisons include the whole hybrid alias stack versus Native-S;
incremental reader-only attribution still needs matched source variants.
[Plan, all pairs, run fingerprints and natural exit receipts](evidence/hybrid-batch-pilot-pairs.json).

The quality capture preserves natural EOS and configured ASCII text stops using
a byte table independently checked against all 49,152 official tokenizer entries
and 500 random multi-token sequences. It stops the logical request through the
engine API, preserves its terminal output and records the matched byte offset.
The underlying engine reports `abort` for that logical text-stop action; no OS
process is signaled. Scoring trims at the recorded text boundary. This is a new
100-question frozen GSM8K cohort, not a reproduction of a historical HF score.

[Full quality report](quality.md) preserves the Native-S/full4 controls, original
compact scheduling failure, corrected rerun and all source/input fingerprints.
