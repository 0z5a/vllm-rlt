# Quantization experiment results — 2026-10-05

This is an in-progress numerical qualification. No model has completed G0–G3.
There are no measured E2E speedups yet; missing values are not zero speedup.

| Model | Baseline tok/s | Candidate tok/s | Paired speedup | Speed change | Lower 95% bound | Status |
|---|---:|---:|---:|---:|---:|---|
| Ouro-1.4B | — | — | — | — | — | G0 incomplete |
| Ouro-2.6B | — | — | — | — | — | Not run |
| Huginn-3.5B | — | — | — | — | — | Not run |
| Nanbeige4.2-3B | — | — | — | — | — | Not run |

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
