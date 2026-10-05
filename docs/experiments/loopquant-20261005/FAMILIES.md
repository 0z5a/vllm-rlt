# Shared INT4 deployment for HRM and LoopFormer

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
separate immutable source snapshot. Native GPU INT4 reload/Graph/compact checks
for these new families are queued. Full checkpoints, quality gates and paired
high-concurrency/multiple-batch E2E remain required.

| Model | BF16 tok/s | Native INT4 tok/s | Paired speedup | Speed change |
|---|---:|---:|---:|---:|
| HRM-Text-1B | Not measured | Not measured | — | — |
| LoopFormer3×8 | Not measured | Not measured | — | — |
