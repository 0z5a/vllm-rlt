# Direct prefix and alias tail

Status: CPU and CUDA correctness passed; official-checkpoint quality passed; independent speed
pilots are running.

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
| Direct-prefix alias reader | — | — | — | Pending independent timed pairs |

The two completed pilots are distinct campaigns. Their numbers do not form an
incremental alias-to-compact comparison. [The profile](profile.md) motivates
this reader change; its summed kernel durations are not an E2E speed forecast.

The quality capture preserves natural EOS and configured ASCII text stops using
a byte table independently checked against all 49,152 official tokenizer entries
and 500 random multi-token sequences. It stops the logical request through the
engine API, preserves its terminal output and records the matched byte offset.
The underlying engine reports `abort` for that logical text-stop action; no OS
process is signaled. Scoring trims at the recorded text boundary. This is a new
100-question frozen GSM8K cohort, not a reproduction of a historical HF score.

[Full quality report](quality.md) preserves the Native-S/full4 controls, original
compact scheduling failure, corrected rerun and all source/input fingerprints.
