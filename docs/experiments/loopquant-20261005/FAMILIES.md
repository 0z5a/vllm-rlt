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
separate immutable source snapshot. The final feature snapshot18cd202b passes [644 CPU tests and36 subtests](native-families-full-cpu-attempt4.log), with29 skipped and217 CUDA cases deselected. Native GPU INT4 reload/Graph/compact checks
for these new families are queued. Full checkpoints, quality gates and paired
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
