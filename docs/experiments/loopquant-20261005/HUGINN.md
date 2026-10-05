# Huginn quantization adapter qualification

This increment uses the existing native Huginn runtime from upstream
[PR 82](https://github.com/ThinkFlowLab/vllm-rlt/pull/82), fixed at
`3b893657f8541a1f5a6125e9703113b8d59e87ff`, and the shared quantization harness
from [PR 89](https://github.com/ThinkFlowLab/vllm-rlt/pull/89), fixed at
`ed2019509b0de6d32927ae51a6b92afd81fac27c`. The integration base is
`bfc7cbcb27ef0e1d80b1a4538466fa292a868add`; the incremental review excludes both
dependencies' changes.

The adapter preserves fixed prelude injection, adjacent-pair rotary embeddings,
Q/K bias, the sandwich residual/norm order, both readout norms, and the tied
embedding/head. Recurrence is differentiable through every requested step.
Q0 teacher and student receive the same initial random state; changing a scale
does not also change the teacher's stochastic input. Core projections can use
the common quantizer, while protected prelude/coda, injection, and tied head
remain at their original precision unless explicitly selected for an ablation.

[Eight original-class comparisons](huginn-official-tiny-attempt1.json) pass
bitwise with tiny random weights: FP32/BF16, B1/B2, R1/R4. The original model code
is pinned at `bb6621b65e90b6a4b9b29ef88dc83866d450470c`; an explicit shared
initial state isolates arithmetic from random-number consumption.

[59 CPU checks](huginn-quant-cpu-attempt2.log) pass, with four GPU cases
deselected. Checks include gradients through all four recurrent states,
identical Q0 teacher/student initialization, tied storage, and existing native
Huginn tests. These results do not establish complete-checkpoint quantization
quality or engine performance.

| Model | BF16 tok/s | Strong quantized baseline tok/s | Candidate tok/s | Paired speedup | Speed change |
|---|---:|---:|---:|---:|---:|
| Huginn-0125 | Not measured | Not measured | Not measured | — | — |

The [full official checkpoint comparison](official-huginn-cpu-attempt1.json)
completed all 96 CPU BF16 SDPA cases: 32 fixed inputs × R8/16/32. All 1,792
recurrent states and 96 final logit tensors are bitwise equal to the unmodified
pinned official class. Both paths receive the same explicit seeded initial state.
The native module shares the loaded official parameter storage and retains the
tied embedding/head. [Frozen runner](run_official_huginn_cpu.py),
[manifest](official-huginn-cpu-manifest.json) and
[verified summary](official-huginn-cpu-summary.json) preserve the scope and hashes.

This completes the full-weight dense-adapter CPU comparison. Native cached GPU
semantics, all-loop calibration, quantized export, locked quality and the full
multiple-batch/high-concurrency E2E matrix remain to be executed.


The native likelihood helper now scores a complete causal window through the
actual prelude, recurrent, and coda KV layers. It fixes the initial-state seed
while restoring the caller's CPU and selected-device random state, then frees
the request's complete cache allocation after scoring. Nine new CPU cases cover
two lengths, two depths, two seeds, and rejection of a cache lacking boundary
layers. Repeated scores agree exactly; dense-adapter likelihood agrees within
the existing FP32 numerical envelope.

The [frozen CPU regression](full-cpu-huginn-quality-attempt1.log) passed
484 tests, with 29 skipped and 116 GPU cases deselected. The
[manifest](huginn-quality-cpu-manifest.json) identifies source, snapshot, and raw
evidence. GPU random-state restoration, full-weight native likelihood, and
quantized Huginn performance have not yet been evaluated.


## Native INT4 preparation

The new Huginn artifact contains one packed matrix for each of the four physical
projections in each recurrent block. Prelude/coda, injection, all norms and Q/K
bias remain BF16. The persistent RoPE buffer remains FP32; the embedding is
serialized once and the head alias is restored on load. The reused Torch
native INT4 backend is the implementation already qualified on Ouro/Nanbeige.
Its Huginn integration has not yet run on GPU.

[Five new portable-export CPU checks](huginn-int4-v1/huginn-int4-cpu-attempt1.log)
cover RTN, three supplied-matrix GPTQ artifact labels, protected tensors, and
rejection of rounded RoPE. The supplied-matrix checks validate serialization;
they do not run GPTQ calibration. Successful temporary test payloads are removed.
The [full frozen suite](huginn-int4-v1/full-cpu-huginn-int4-attempt1.log) completed
489 passed,29 skipped,118 GPU cases deselected. Source and archive hashes are in
the [CPU manifest](huginn-int4-v1/cpu-source-manifest.json).

The [future GPU packet](huginn-int4-v1/manifest.json) has two tiny Graph-off/on
cases: exact encoded reload, tied-head storage, FP32 RoPE, seeded native
likelihood with restored RNG, repeated engine requests and zero KV after drain.
It is local only, not admitted or uploaded. Full-weight Huginn quantization,
locked quality and all performance speedups remain unmeasured.


## Native FP8 preparation

Huginn now has DYN/SH1/ST2/LOOP core producers, optional fused RMSNorm,
Torch/CUTLASS rowwise GEMM, and an encoded native export. Core Q/K bias,
adjacent-pair RoPE, sandwich norms and boundary injection remain explicit.
Export keeps one packed core matrix and one protected embedding; native load
restores its head alias and FP32 RoPE buffer.

The producer and GEMM components are reused from shared harness source22f84cd.
The explicit-loop metadata changes are ported from shared commitbfada6a:
prepared rows, padded workspaces, prefill expansion, async routing and Graph
replay carry actual loop IDs. CPU reordering/padding/ID-reuse coverage passes.
Huginn itself still requires synchronous scheduling; the two async/sync metadata
GPU controls use Ouro and do not claim Huginn async support.

The [frozen full CPU suite](huginn-fp8-v1/full-cpu-huginn-fp8-attempt1.log)
passed490 tests,29 skipped,136 GPU cases deselected. This confirms CPU regression
and collection, not execution of the CUDA producers or Huginn FP8 GEMMs.
[Source manifest](huginn-fp8-v1/cpu-source-manifest.json).

The [local future GPU packet](huginn-fp8-v1/manifest.json) has18 cases:16 Huginn
policy × Graph × fused-norm combinations and2 loop-metadata controls. Each
Huginn case compares complete token/exit-depth sequences against decoded FP8
arithmetic, repeats request IDs, checks exact encoded reload and tied storage,
and requires zero KV after drain. Full-weight calibration, native GPU G0,
locked quality and formal performance remain unrun. The packet is not admitted,
uploaded or executed.


## Boundary prefill repair after H20 execution

The first H20 FP8 case stopped before completing its request-lifecycle checks:
prelude returned six live states, but the core metadata described eight padded
rows. The error was retained at the original numerical budgets. The executor
now prepares core metadata with the actual prelude row count and records that
submitted size. Prelude and coda keep their existing unpadded boundary KV.
No rotary slicing or fabricated KV rows are introduced.

The added regression compares eager and static-buffer paths bitwise for
FP32/BF16, 3/6/7 live rows, single/multiple requests and two allocation lifetimes.
GPU repair validation is prepared separately; these CPU tests do not turn the
failed H20 attempt into a pass.

[New checks](boundary-prefill-v2/source-huginn-prefill-rows-cpu-attempt1.log): 6 passed. [Model/engine/KV regression](boundary-prefill-v2/source-huginn-prefill-rows-regression-attempt1.log): 79 passed, 3 GPU cases deselected; both test processes naturally exited 0.
