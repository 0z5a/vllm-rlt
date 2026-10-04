# Quantization experiment results — 2026-10-05

This is an in-progress numerical qualification. No model has completed G0–G3.
There are no measured E2E speedups yet; missing values are not zero speedup.

| Model | Baseline tok/s | Candidate tok/s | Paired speedup | Speed change | Lower 95% bound | Status |
|---|---:|---:|---:|---:|---:|---|
| Ouro-1.4B | — | — | — | — | — | G0 incomplete |
| Ouro-2.6B | — | — | — | — | — | Official CPU comparison running |
| Huginn-3.5B | — | — | — | — | — | Tiny adapter checks in separate draft |
| Nanbeige4.2-3B | — | — | — | — | — | Tiny adapter checks only |

## Native FP8 correctness

RTX 5090/SM120, Torch2.12.1+cu130, CUDA13.0. Each 64×32 projection uses one 2048-byte packed matrix. Native scalar-scale GEMM includes unfused encoding and M-padding; this is not the strong deployment baseline.

| Input rows | Maximum absolute difference vs same quantized values |
|---:|---:|
| 1 | 0 |
| 17 | 0 |
| 32 | 0.001953125 |
| 64 | 0.000000119209 |
| 128 | 0 |

Stable scale-buffer updates followed by CUDA Graph replay passed. [Raw receipt](native-fp8-probe-attempt1.json).

## Full-model numerical failures retained

Official Ouro revision `574fa66cb8bf5abdc979642d01cf2b79b16bfab1`, runtime base `ecb1f8b505b7e831815b40aec3b4598619cca23a`. Weight bytes independently SHA256 verified. Fixed criterion: `atol=rtol=3e-4`.

| Attempt | Comparison | Result |
|---|---|---|
| [1](ouro-g0-attempt1.log) | FP32 vs initial FP32 functional oracle | Fail, smoke-7 R4; abs 0.000334829 |
| [2](ouro-g0-attempt2.log) | FP32 vs official operator-order FP32 oracle | Fail, smoke-7 R4; abs 0.000328362 |
| [Diagnostic](ouro-fp64-diagnostic-attempt1.json) | Only smoke-7 vs FP64 oracle | Pass unchanged threshold; not full G0 |
| [3](ouro-g0-attempt3.log) | FP32 vs FP64 oracle over the smoke set | Adapter failed at smoke-14 R3; abs 0.000332700 |

Attempt3 stopped after the first failed assertion; later inputs were not checked.
The original logs remain retained.

The next [complete audit](ouro-g0-attempt4b.json) collected all 32 inputs at R1–R4
in both FP32 and BF16. Its process naturally exited 0, but the fixed FP32
criterion failed in 36 of 128 combinations across the five checked tensors.
See the [aggregated counts](ouro-g0-attempt4b-summary.json). The first launch
had a [missing prompt path](ouro-g0-attempt4.log) and exited before GPU work;
the corrected launch has a separate [raw log](ouro-g0-attempt4b.log).

| BF16 comparison against official eager arithmetic | Maximum absolute error | Logit argmax disagreements / token positions |
|---|---:|---:|
| Dense adapter state | 0 (bitwise equal at all 128 points) | — |
| Dense adapter logits | 0 (bitwise equal at all 128 points) | 0 / 6,896 |
| Native torch attention state | 6.03125 | — |
| Native torch attention logits | 8.8984375 | 25 / 6,896 |

The native Torch backend computes attention scores and value accumulation in
FP32; the official eager path rounds scores and probabilities to BF16. These
are baseline differences requiring incremental-cache and deployment-backend
qualification, not a passed overall G0 or a quantization quality result. The
FP32 comparison had zero logit argmax disagreements but retained nonzero
fixed-tolerance failures (maximum logit error 0.00420481).

## CPU checks

[Final full regression](full-cpu-final.log): **454 passed, 29 skipped, 113 GPU
cases deselected**; [JUnit](full-cpu-final.xml). Earlier full collection stalled
inside an existing macOS dependency load; both processes subsequently completed
naturally. The final run includes serving, GSM8K collection, and the latest
quantization changes. Skipped optional/dependency/GPU paths are not claimed as
passes. The earlier [core-only run](core-cpu-attempt1.log) is retained but does
not need to be added to the final count.

Checks include exact optimizer/RNG/data-position resumption, unchanged frozen
weights, and [BF16 eager oracle equality](bf16-adapter-cpu.log).
[GPTQ checks](gptq-cpu-attempt1.log) verify agreement with RTN for a diagonal
Hessian, consistent blocked/sequential compensation, and masked row/energy
accounting across loops. This signed static-group variant follows the
[GPTQ algorithm](https://arxiv.org/abs/2210.17323); no model-level quality or
native W4A16 performance has been measured.

Ruff lint and formatting pass. The [pre-commit attempt](pre-commit-attempt1.log)
reports that its wrapper is absent; no package was installed or upgraded.

Formal results require exported-model noninferiority and independent paired
E2E trials across C1/8/32/64, multiple batch limits, long/decode/mixed/open
workloads. Failed cohorts and captures during timing cannot establish a speedup.

After that run, FakeQuant's reference was corrected to preserve decoded values
in FP32 until the GEMM output conversion. This avoids an extra pre-matmul BF16
rounding absent from the native FP8 operation. The targeted
[38 CPU checks](loopquant-cpu-attempt3.log) pass; full GPU export parity remains
pending. Recurrent state and residual outputs remain BF16.

An explicit `attention_backend` selects eager or SDPA arithmetic for the adapter;
the Q0 teacher inherits that selection and export records it. The latest
[44 CPU checks](loopquant-cpu-trace-attempt1.log) cover both choices and native
trace metadata. A real tiny-engine cohort confirms that tracing preserves the
generated tokens and accounts for prefill/decode rows at every loop. A separate
reordered-depth/padding check excludes poisoned padding rows from statistics.

## Incremental cache and GEMM row shape

The [complete cache audit](ouro-cache-g0-attempt1.json) naturally exited 0. Both
Torch and Triton preserve every previous K/V prefix bitwise, keep depth tables
disjoint, and free every allocation. Each backend also has zero incremental
versus full-prefill logit argmax disagreements across 512 comparisons. The
registered `atol=rtol=0.02` element criterion nevertheless fails; process success
does not turn those numerical failures into passes.

| Native backend | Incremental/full maximum logit error | Failed logit comparisons | Logit argmax disagreements |
|---|---:|---:|---:|
| Torch | 6.046875 | 511 / 512 | 0 / 512 |
| Triton | 6.8984375 | 511 / 512 | 0 / 512 |

Two calibration smoke inputs with large differences were selected before the
[GEMM diagnostic](gemm-diagnostic-preregister.json). Neither variant changes the
criterion or the checkpoint. Per-token Linear is a diagnostic control, not a
performance baseline.

| Control | Incremental/full maximum logit error | Failed state / logit comparisons |
|---|---:|---:|
| [Disable BF16 reduced-precision reduction](ouro-gemm-fp32-batched-attempt1.json) | 6.34375 | 32 / 32; 32 / 32 |
| [Also fix Linear to single-token GEMM](ouro-gemm-fp32-per-token-attempt1.json) | 0 | 0 / 32; 0 / 32 |

The latter yields bitwise state and logit equality for these two inputs. It
isolates the Linear row-shape path as the cause of their incremental differences;
it is not a complete 32-input result. Both variants still fail the separate
prefill/SDPA comparison. PyTorch documents that batch shapes and BF16 reduction
choices can change floating-point results ([numerical accuracy](https://docs.pytorch.org/docs/2.12/notes/numerical_accuracy.html)).
The original failures remain visible and no threshold has been relaxed.

The unmodified official HF class is running a separate full-weight CPU
comparison using the existing Transformers 4.54.1 environment and independently
verified weights. Its full result remains pending.

## Nanbeige adapter qualification

The [original-class comparison](nanbeige-official-tiny-attempt1.json) uses
unmodified source from Nanbeige4.2-3B revision
`b82e54bd609793562a75cbf9337970a93369eab5` with tiny random weights. All eight
FP32/BF16 × R1/R2 × loop-final-norm settings have bitwise equal valid-token
logits. This verifies the adapter's residual and normalization placement on
these inputs; the official full checkpoint remains untested.

[47 targeted CPU tests](loopquant-nanbeige-cpu-attempt1.log) pass, with one GPU
case deselected. They include native Nanbeige fixed-loop output/mask checks and
two scale updates through both recurrent steps, using the same architecture
and attention for the BF16 teacher. The first shared Q0 implementation and
Ouro checks remain covered after extracting common projection bookkeeping.

Q0 now reuses one FP32 decoded reference matrix per physical projection during training, so backward does not retain a separate decoded copy per loop. It is a nonpersistent training buffer and is excluded from checkpoints and deployment exports. [48 CPU checks](loopquant-reference-cache-cpu-attempt1.log) pass, including four-loop saved-storage accounting and exact cache reconstruction after loading the packed state. This is a training-memory change, not a native serving-memory claim.


## Unmodified official Ouro-1.4B class

The pinned Hugging Face model loaded through existing Transformers 4.54.1 and
our dense adapter agree **bitwise at all 128 state/logit comparisons**: 32
inputs × R1–R4, BF16 eager attention. The full CPU run naturally exited 0 in
3221.96 s. [Raw receipt](official-ouro-cpu-attempt1.json). This confirms the
adapter against the original class, independently of the earlier functional
oracle. It does not qualify native attention, model quality after quantization,
or serving throughput. Its completed local duplicate weight was removed;
remote weights needed by later experiments remain retained.

## Complete per-token GEMM control

The [full 32-input control](ouro-gemm-full-per-token-attempt1.json) extends the
earlier two-input result. Disabling BF16 reduced-precision GEMM reduction and
using one token per Linear call removes most incremental/full-prefill drift,
but **does not remove all of it**: 53/512 state and 66/512 logit comparisons fail
the unchanged `atol=rtol=0.02`; maximum logit difference is 3.734375. All 512
logit argmax checks agree. Prior KV prefixes remain unchanged and allocations
are released. Prefill against SDPA still fails at all 128 points. Therefore the
two-input bitwise result must not be generalized to the full smoke set.

## Native rowwise FP8

The installed Torch 2.12.1+cu130 `torch._scaled_mm` accepts rowwise A scales
`[M,1]` and columnwise B scales `[1,N]` on RTX 5090/SM120. All 20 combinations
of M=1/17/32/64/128 and (K,N)=(64,32), (2048,2048), (2048,5632),
(5632,2048) pass the registered 0.02 absolute/relative encoding comparison;
maximum absolute difference is 0.015625. Two FakeQuant/native scale checks
also pass. [Raw receipt](rowwise-fp8-probe-attempt1.json). Rowwise graph replay,
producer fusion, model-level quality and performance remain untested.

## Four-window native closed-loop calibration pilot

The pilot completed four 512-token calibration windows, C4, scheduler token
budget 128, R4, and 32 generated tokens per request with native Triton
attention. It produced 69,142 trace calls and 2,896 statistics groups, with zero
failed requests and zero KV allocations after drain. Every loop/projection
received 2,048 prefill and 124 decode rows; the latter counts the 31 generated
tokens actually fed back per request. [Receipt and artifact hashes](ouro-calibration-trace-pilot-summary.json).

| Layer-0 q-projection input | Loop 0 | Loop 1 | Loop 2 | Loop 3 |
|---|---:|---:|---:|---:|
| Prefill amax | 2.125 | 6.625 | 6.6875 | 6.625 |
| Decode amax | 1.71875 | 3.34375 | 3.015625 | 2.921875 |
| Prefill RMS | 0.132093 | 0.142895 | 0.147032 | 0.149037 |
| Decode RMS | 0.132585 | 0.144288 | 0.147717 | 0.149526 |

These are pilot statistics, not the required 128-window calibration or scale
selection on dev data. Raw compressed statistics (65,519,837 bytes) and trace
(1,240,886 bytes) are retained locally with independent SHA checks. The
instrumented 247.98 s process interval is not serving performance.

The next collector accumulates statistics on the activation device and reuses
prepared row indices, removing per-update scalar/boolean-index synchronization.
Reservoir sampling remains uniform and bounded at 4,096 observations; its
seed and device are recorded because CPU/CUDA RNG streams differ. Nonfinite
CUDA data causes rejection when statistics or policies are materialized.
[49 CPU checks](loopquant-device-stats-cpu-attempt1.log) pass; two GPU checks
were deselected. GPU parity and calibration-cost comparison remain pending.

Huginn's separately based adapter is in
[draft 0z5a/vllm-rlt#10](https://github.com/0z5a/vllm-rlt/pull/10), on a frozen
integration of native Huginn PR82 and this quantization harness. Its eight
original-class tiny comparisons are bitwise exact and 59 targeted CPU checks
pass. It has not run its full checkpoint or E2E matrix.

A native fused-producer implementation is now staged for GPU qualification:
explicit caller-owned scratch, E4M3 row casting/padding, optional RMSNorm,
DYN or absolute-loop static selection, and the installed rowwise CUTLASS GEMM.
It has no custom GEMM or installed dependency changes. Its finite GPU probe
covers 120 configurations and three graph replays; these checks are **not run**
until the next resource handoff. It is not yet integrated into model serving.

## Mixed-depth metadata and fixed-depth workload checks

Prepared batches now carry an explicit device `loop_ids` vector through ordinary
staging, padded workspaces, asynchronous routing, packed-prefill expansion and
CUDA Graph inputs. Padded rows use -1; graph inputs are refreshed on every replay.
The workspace memory estimate includes the new 8-byte-per-row vector.
[Full CPU regression](full-cpu-loop-metadata-attempt1.log) completed with
**466 passed, 29 skipped, 114 GPU cases deselected**. Added GPU graph checks
make model output depend on the loop vector to detect stale replay metadata;
those two new cases have not run.

A fixed-depth cohort now rejects mismatched prefill/decode depths before any
request is admitted. The engine's prefill uses the model configuration, so
setting only request `min_loops=max_loops` is insufficient for R1–R3 curves.
The registered model recurrence must also match R. Previous R4 receipts are
unaffected. [49 targeted CPU checks](loopquant-fixed-depth-cpu-attempt1.log)
pass with 29 GPU cases deselected, including a real R2 cohort and rejection of
R4-prefill/R2-decode mislabeled as fixed R2.
