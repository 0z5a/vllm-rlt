# Looped-family quantization adapters and shared INT4 deployment

This incremental change uses the frozen integration base
`b836c1eb79412669f8cb6f07605b08be4e149f6b`. It combines the existing
[native LoopFormer/HRM stack](https://github.com/0z5a/vllm-rlt/pull/22)
at `4176b1bfc0a6ca0478cc4d459245f819d38e5e7a` with the
[quantization harness](https://github.com/ThinkFlowLab/vllm-rlt/pull/89)
at `8032bbb7321813c17295bb2b2f873567ed02bb50`. The integration preserves
per-row loop labels in native, alias and compact routing. Its HTTP benchmark
also closes admitted client streams when the engine fails.

INT4 artifacts now select physical core matrices by family. LoopFormer has
four projections per block; time/delta embedders, AdaLN modulation, learned
positions, RMSNorm and the tied output embedding remain BF16. Its recurrent
clock is unchanged. HRM has four projections per layer in each of H/L modules,
with one shared matrix across repeated inner L and outer H applications.
Its initial low state, embeddings and head remain BF16. Resident states, KV,
rotary arithmetic and bidirectional-prefix/causal-continuation semantics are
unchanged. The exporter also accepts preregistered first/all/matched-row GPTQ
matrices; no full-model GPTQ run is claimed.

[Four artifact checks](native-families-int4-cpu-attempt1.log) pass for
Ouro, Nanbeige, HRM and LoopFormer. They verify protected values, physical matrix
counts, portable codes and BF16 scale rounding. The frozen integration base
passes [642 CPU tests and36 subtests](native-families-base-cpu-attempt3.log),
with29 skipped and205 CUDA tests deselected. Its [74 targeted model/cache checks](native-families-base-cpu-attempt2.log)
and [six HTTP tests](native-families-http-failure-cpu.log) pass, including an
engine error after admission.

The initial integration had missing compact loop labels and failed tests.
Its process26568 remains idle in its pre-fix HTTP wait; it has not been signalled,
terminated or counted as a passing run. The complete passing suite ran from a
separate immutable source snapshot. The final feature snapshot18cd202b passes [644 CPU tests and36 subtests](native-families-full-cpu-attempt4.log), with29 skipped and217 CUDA cases deselected. Native GPU INT4 reload/Graph/compact checks for these new families now pass in the64-case matrix below. Full checkpoints, quality gates and paired
high-concurrency/multiple-batch E2E remain required.

| Model | BF16 tok/s | Native INT4 tok/s | Paired speedup | Speed change |
|---|---:|---:|---:|---:|
| HRM-Text-1B | Not measured | Not measured | — | — |
| LoopFormer3×8 | Not measured | Not measured | — | — |


LoopFormer's differentiable adapter preserves the same eight1/8 steps, learned
positions, protected conditioning and tied readout. The
[adapter manifest](loopformer-adapter-manifest.json) specifies physical versus
resident state width and its1024-token context constraint.
[Eight pinned-author comparisons](loopformer-author-adapter-attempt1.json)
pass bitwise for FP32/BF16 × B1/B2 × T3/T7: all64 recurrent state tensors and
eight logit tensors. This uses copied tiny weights, not the full checkpoint.
The [runner](probe_loopformer_author_adapter.py) records both source hashes.
[Three focused checks](loopformer-adapter-cpu-attempt2.log) cover masked rows,
clocks, fixed depth and gradients from final logits through all eight applications
of the shared projection. Two tiny scale updates preserve all frozen parameters.
Full LoopFormer QAT remains pending; native FP8 implementation and its unrun
GPU qualification packet are described below.

The unchanged pinned author and dense adapter also agree **bitwise on all48
official-checkpoint BF16 CPU cases**:32 single inputs and16 right-padded pairs,
all at eight steps. All384 recurrent state tensors and48 logit tensors match
on valid rows; clocks are exact at every boundary. The32 independent prompts
contain1724 tokens, with3448 valid positions evaluated across both batch sizes.
The author and adapter share the same loaded parameter storage.
[Raw comparisons](loopformer-official-v1/official-loopformer-adapter-cpu-attempt1.json),
[verified summary](loopformer-official-v1/official-loopformer-adapter-cpu-summary.json),
[frozen runner](loopformer-official-v1/run.py), [manifest and inputs](loopformer-official-v1/manifest.json).
Both processes exited naturally0. The short-term read lease is released; the
owner retains the existing weights for later tests. This dense comparison does
not qualify native cached arithmetic, quantized quality or serving performance.


Native quality evaluation now accepts an explicit prefix/continuation boundary.
HRM admits the complete bidirectional prompt atomically, then scores each next
token before adding it to the causal continuation cache. The generic full-window
NLL rejects HRM to prevent target leakage. LoopFormer always uses its complete
eight-step recurrence. [Eleven CPU checks](native-continuation-cpu-attempt2.log)
match independent HRM/LoopFormer oracles for prefixes1/5, native/compact KV and
two request lifetimes; causal Ouro/Nanbeige/LoopFormer controls agree with full-window
targets. Full-checkpoint quality remains unrun.

The LoopFormer adapter snapshot also passes the
[647-test CPU suite](native-families-full-cpu-attempt5.log), with36 subtests,
29 skipped and217 GPU cases deselected.

The final continuation-quality snapshot `a513200` passes [658 CPU tests and36 subtests](native-families-full-cpu-attempt6.log), with29 skipped and217 GPU cases deselected. Frozen GPU64-case qualification still uses source18cd202b; no result is inferred from the CPU suite.

## Complete four-family tiny GPU qualification

[All64 cases](native-families-v1/native-families-v1.json) pass on RTX5090 with the frozen source18cd202b: 16Ouro+32Nanbeige FP8 regressions and16INT4 cases across Ouro/Nanbeige/HRM/LoopFormer, Graph off/on and native/compact KV. INT4 checks match complete generated token/exit-depth sequences against decoded packed-weight references across two request lifetimes, verify identical native reloads, distinct stable physical packed pointers, unchanged protected parameter sets and empty KV after drain. This is tiny deployment qualification, not full-checkpoint accuracy or speed.

| Family | FP8 cases | INT4 cases | Result | Full model speedup |
|---|---:|---:|---|---|
| Ouro | 16 | 4 | Pass | Not measured here |
| Nanbeige | 32 | 4 | Pass | Not measured |
| HRM-Text | — | 4 | Pass | Not measured |
| LoopFormer | — | 4 | Pass | Not measured |

The controller and child exited naturally0; all67 archived file sizes/hashes passed independent offbox verification. All64 temporary tensor payloads were removed after their cases passed; manifests and results remain. [Summary](native-families-v1/summary.json), [completion/cleanup](native-families-v1/complete.json). The later continuation scoring/differentiable adapter changes remain covered by658 CPU tests; this frozen GPU run does not add full-model quality evidence.

## LoopFormer FP8 deployment preparation

The FP8 path converts only the four shared core projections per block and
preserves BF16 time/delta embeddings, AdaLN conditioning, RMSNorm, residual gates,
GELU, learned positions and the per-row clock. Four explicit producers support
DYN/SH1/ST2/per-loop scales through the prepared row loop IDs. This path does
not fuse RMSNorm with AdaLN; no normalization-fusion benefit is claimed.
Encoded export/reload keeps one packed matrix per projection and the original
embedding-based tied readout.

Frozen source `9cdc7977f5f848d88cb4ca12def6af72342b8418` passes
[658 CPU tests and36 subtests](loopformer-fp8-v1/full-cpu-loopformer-fp8-attempt1.log),
with29 skipped and233 GPU tests deselected. This establishes CPU regression and
collection only. [Sixteen RTX5090 GPU cases](loopformer-fp8-v1/gpu-packet-manifest.json)
remain prepared and unrun on that device: four policies × Graph off/on ×
native/compact KV, with asynchronous scheduling, fixed R8 generated sequences,
request-ID reuse, stable packed pointers, encoded reload and KV drain. Six
native likelihood targets are also compared against the decoded FP8 reference
at unchanged0.02 absolute/relative budgets. [Probe](loopformer-fp8-v1/probe.py),
[finite controller](loopformer-fp8-v1/run.py). Full-model quality and E2E remain
pending; the existing INT4 GPU result does not qualify this FP8 implementation.

## HRM differentiable recurrence and continuation supervision

The dense HRM adapter unfolds every H/L application: three applications of the
shared L module, then one H application, for each of two outer cycles. Explicit
outer-cycle IDs select scale stages; repeated L calls accumulate into the same
physical projection and stage. Embeddings, the initial low state, untied head,
normalization, gated attention and rotary arithmetic keep their original roles.
This enables gradients through the full recurrence and does not reproduce the
author's truncated-backpropagation training schedule.

Each row supplies its prefix length. Prefix attention is bidirectional and the
continuation is causal; right padding is excluded from attention keys. CE and
KL start at the first continuation target, excluding prompt and padding targets.
The update budget still counts all valid input tokens, while supervised targets
are reported separately. Teacher and student use the same prefix boundaries.

Eight new CPU checks cover ragged FP32 states/logits against an independent
functional oracle, eager/SDPA and causal/prefix modes, future-token isolation,
bidirectional-prefix behavior, target masking and gradients through all six L
projection applications. Two tiny Q0 updates change scales while preserving
frozen weights and packed storage; each records seven input tokens and four
supervised targets. These are preparation checks, not full-checkpoint quality.

Frozen source `edc3eb0f21fa2635f28557964474466b1133d6ea` passes
[666 CPU tests and 36 subtests](hrm-adapter-v1/full-cpu-hrm-adapter-attempt1.log),
with 29 skipped and 233 GPU tests deselected; the process exited naturally with
code 0. [Focused checks](hrm-adapter-v1/hrm-adapter-cpu-attempt1.log) and
[adapter/evidence manifest](hrm-adapter-v1/manifest.json) are retained. Full HRM
adapter comparison, Q0 training, quantized quality and E2E remain unrun.

## HRM FP8 deployment preparation

The native FP8 implementation replaces four projections per H/L layer. Each
physical packed matrix is shared across its recurrent uses; activation scales
select the explicit outer H-cycle ID. The packed GQKV matrix includes the
attention-gate projection; sigmoid gating itself remains in BF16 arithmetic.
Embeddings, the initial low state and untied head retain BF16 parameters.
RMSNorm, per-module final norms, FP32 rotary arithmetic, residual additions,
the complete H/L recurrence and distinct KV planes remain intact. This version
does not fuse the nonaffine RMSNorm with activation quantization.

Frozen source `49734509500252cb0548d11fc78c51816666f425` passes
[666 CPU tests and 36 subtests](hrm-fp8-v1/full-cpu-hrm-fp8-attempt1.log),
with 29 skipped and 249 GPU cases deselected, natural exit 0. These results
establish CPU regression and test collection only; the later H20 qualification is reported below.
[Sixteen prepared GPU cases](hrm-fp8-v1/gpu-packet-manifest.json) cover
DYN/SH1/ST2/LOOP, Graph off/on and native/compact KV with asynchronous scheduling.
They compare encoded reloads, protected tensors, fixed R2/L3 generated sequences,
continuation NLL at prefix lengths 1/3, repeated request IDs, packed pointers
and complete KV drain. ST2 and per-loop have the same two-stage partition at R2.
[Probe](hrm-fp8-v1/probe.py), [finite controller](hrm-fp8-v1/run.py).
Full-weight quantized quality, memory and E2E speed remain unmeasured.


## H20 native FP8 qualification

On H20 (SM90), all 16 HRM and all 16 LoopFormer native FP8 cases pass at the
existing helper criteria. Each family covers DYN/SH1/ST2/LOOP, Graph off/on,
native/compact KV, generated sequences across repeated request lifetimes,
encoded reload, protected tensors, stable packed pointers and KV drain. The
continuation-NLL comparisons also pass their unchanged budgets.

| Family | Native FP8 cases | Encoded reload | Full-model E2E speedup |
|---|---:|---|---|
| HRM-Text | 16/16 pass | Exact | Not measured |
| LoopFormer | 16/16 pass | Exact | Not measured |

This packet uses family source `49734509500252cb0548d11fc78c51816666f425`.
The [packet manifest](h20-native-v1/manifest.json),
[HRM raw results](h20-native-v1/h20-native-families-v1-hrm.json),
[LoopFormer raw results](h20-native-v1/h20-native-families-v1-loopformer.json)
and [summary](h20-native-v1/h20-native-families-v1-summary.json) are retained.
Both family children naturally exited 0. The four-child packet itself exited 1
because the separate Huginn and Parcae branches each failed their first case:
non-padded boundary prelude states were paired with padded core metadata. Those
failures and the remaining unrun cases are retained; this is no all-family pass.
All 46 archived file hashes match the offbox copy. Temporary test tensors were
removed while preserving their archived evidence and export manifests.
Full-checkpoint G0, quantized quality, memory savings and E2E remain separate.


## HRM official weights: CPU recurrence qualification

All 112 registered BF16 CPU comparisons passed: 32 inputs at B1 with first-token,
half-length and full-length prefixes, plus 16 ragged B2 pairs with half-length
prefixes. All 224 outer H/L state tensors and 112 final logit tensors are bitwise
exact; argmax is exact. Coverage is 1,728 unique input tokens and 6,912 evaluated
tokens, using the complete R2/L3 recurrence.

The five author source files remain unchanged at revision
`aaa948ea674fd84b7bc455c9cfb455ecfefdf914`. The runner explicitly replaces the
unavailable FlashAttention dependency with CPU SDPA. This qualifies the author's
recurrence and this declared attention substitution. It does not qualify the
original CUDA attention, native cached execution, quantized quality or serving.
The adapter source is `edc3eb0f21fa2635f28557964474466b1133d6ea`;
its HRM adapter and model files are unchanged in family source `8045c32`.

[Raw results](hrm-official-cpu-v1/official-hrm-adapter-cpu-sdpa-attempt1.json),
[verified summary](hrm-official-cpu-v1/official-hrm-adapter-cpu-sdpa-summary.json),
[natural receipt](hrm-official-cpu-v1/official-hrm-adapter-cpu-sdpa-receipt.json),
[runner](hrm-official-cpu-v1/run.py) and
[pinned input/source manifest](hrm-official-cpu-v1/manifest.json) retain the evidence.
The controller and child naturally exited0 and were independently absent.
The short reader hold was released; the shared model remains under its owner's
future-work hold and was not copied, changed or deleted.

| Model | BF16 tok/s | Quantized tok/s | Paired speedup | Speed change |
|---|---:|---:|---:|---:|
| HRM-Text-1B | Not measured | Not measured | — | — |

Full native G0, quantized quality and the high-concurrency/multiple-batch E2E
matrix remain pending. No memory saving or throughput is inferred from CPU parity.
