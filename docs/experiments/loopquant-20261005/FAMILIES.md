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
Full LoopFormer QAT and native FP8 deployment remain pending.

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
collection only. [Sixteen GPU cases](loopformer-fp8-v1/gpu-packet-manifest.json)
are prepared but **not uploaded or run**: four policies × Graph off/on ×
native/compact KV, with asynchronous scheduling, fixed R8 generated sequences,
request-ID reuse, stable packed pointers, encoded reload and KV drain. Six
native likelihood targets are also compared against the decoded FP8 reference
at unchanged0.02 absolute/relative budgets. [Probe](loopformer-fp8-v1/probe.py),
[finite controller](loopformer-fp8-v1/run.py). Full-model quality and E2E remain
pending; the existing INT4 GPU result does not qualify this FP8 implementation.
