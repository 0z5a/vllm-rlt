# HRM-Text native execution

HRM-Text uses a bidirectional prompt followed by causal decoding. Its prompt must
be processed as one unit: chunking it changes the KV at every recurrent level.
The scheduler now has an atomic-prefill option, rejects prompts larger than the
configured token budget before admission, and packs only complete prompts.
Explicit per-query read lengths let native, alias and compact caches attend the
whole prompt while preserving the actual RoPE position and write address.
Every requested history position must already be written.

The adapter preserves the author's H/L recurrence and checkpoint names. One
scheduler step executes three low-level module applications followed by one
high-level application. Two such steps retain separate KV for all eight module
applications; it does not share KV between recurrences or add early exit.
The resident state holds both `z_H` and `z_L`. Gate values cannot trigger exit,
and admission requires the checkpoint's full H-cycle count and threshold 1.
This is a fixed-depth negative control for skipped-depth storage savings.

| Official checkpoint property | Value |
|---|---:|
| Parameters | 1,182,795,264 |
| Checkpoint tensors | 131 |
| Layers in each H/L module | 16 |
| H cycles / L applications per H cycle | 2 / 3 |
| KV layers per scheduler step | 64 |
| Independent KV layer versions per token | 128 |
| Hidden width / resident state width | 1,536 / 3,072 |
| BF16 KV payload per fully cached token | 786,432 bytes |

The official public assets are pinned to
`sapientinc/HRM-Text-1B@22097cbcecdd1301afe30a19a3ee61b96a9863e5`.
All four local files are verified; the weight SHA256 is
`f8fe2b2bf6948414e8e8d6538659198726d98f967c55b533b7aabe8a1fa9a584`.
The adapter's complete state-dictionary keys and shapes match all 131 official
header entries. This header audit does not count as official-model execution.
The reference contract is the author's
[H/L recurrence](https://github.com/sapientinc/HRM-Text/blob/aaa948ea674fd84b7bc455c9cfb455ecfefdf914/models/baselines/hrm_nocarry_bp_warmup.py),
[gated attention](https://github.com/sapientinc/HRM-Text/blob/aaa948ea674fd84b7bc455c9cfb455ecfefdf914/models/layers.py)
and checkpoint conversion at that same revision.

| Validation | Result |
|---|---|
| Independent dense FP32 logits, intermediate states and all KV versions | Pass for native, alias, compact and credits |
| Full bidirectional P5 prompt plus two causal decode positions | Pass; no tolerance changes |
| Ragged atomic prefills with chunk size 1, two request-ID lifetimes | Pass in four storage modes, sync and async |
| Strict checkpoint dispatch and standalone mmap reader | Exact tensor reload |
| BF16 cast preserves FP32 RoPE constants | Pass |
| Unwritten full-prefix reads and unsupported shallow requests | Rejected before execution |
| Affected CPU regressions | 214 passed, 27 GPU tests deselected, 36 subtests passed |
| Tiny CUDA FP32 states, logits and all KV against independent CPU oracle | Pass at the same tolerances |
| Tiny BF16 CUDA sync/async/Graph, four storage modes, B1–128/C2B | All 72 arms pass, including two request-ID lifetimes |
| Official checkpoint inference | Pending |

The dense oracle uses functional weights and a complete prefix/causal attention
matrix independent of the scheduler, model forward and paged cache. FP32 state,
logit and KV comparisons use absolute tolerance `3e-6`, relative tolerance `3e-5`.
The real CPU engine also matches dense greedy decoding for variable prompts.

The CUDA gate ran frozen source `224521d0d9cd2bf7fb3c5f37d3ab5f32bde35134`
on RTX5090 #1 with torch 2.12.1+cu130. It completed 11,760 tiny-model requests
and 70,560 output tokens. Each storage variant exactly matches native within
the same execution mode, and every completed request object matches across two
ID lifetimes. Every request completes both H cycles and returns its entire
cache reservation. Graph execution records 96 captures, 1,040 replays and zero
fallbacks across all arms. The controller and child processes exited naturally
with code 0; all eight evidence files were verified after copying off the node.
The archive SHA256 is
`bab1e0d4c4c921fbea30d407c58929733ed9620c2b62fe4caf6a1ef429b6dd55`.
[Raw CUDA cases and completion receipt](evidence/hrm-tiny-cuda.json).
This is tiny-model correctness evidence on a shared node; it does not establish
official-model quality or speed.

| Official model E2E comparison | Native tokens/s | Candidate tokens/s | Speedup |
|---|---:|---:|---:|
| HRM-Text-1B | — | — | Not measured |

Only torch/Triton LAST_EXITED execution is admitted. Prefix reuse, preemption,
speculation, FlashAttention and prefill UVA are not qualified. The implementation
supports synchronous and resident execution with tiny CUDA/Graph qualification.
Full production text serving, task quality and high-concurrency official
checkpoint results remain separate work.
