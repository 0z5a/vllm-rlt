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
The independent failure logs are preserved. Eager-attention adapter qualification,
BF16 baseline ranges and incremental KV correctness remain before PTQ searches.

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
