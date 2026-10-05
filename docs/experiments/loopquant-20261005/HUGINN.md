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
