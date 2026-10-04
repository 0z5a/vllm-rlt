# Shared-weight loop quantization

The experiment holds one packed weight matrix per physical projection while
changing activation-scale sharing across recurrent steps. Recurrent state,
residuals, KV, embedding, head, and exit gate stay at their baseline precision.

This draft implements the first numerical and measurement building blocks:

- Signed INT4 encoding and reconstruction, including partial groups; static-group
  GPTQ error compensation with explicit per-loop Hessian row/energy accounting.
- E4M3 FP8 encoding, a deterministic differentiable scale-only reference, and
  absolute loop-stage selection for reordered rows.
- An Ouro adapter with explicit loop boundaries, differentiable full recurrence,
  masked activation statistics, and an independent functional reference.
- Native FP8 scalar-scale GEMM through `torch._scaled_mm`. Padding and conversion
  remain in the operation. This initial unfused implementation is a correctness
  backend; it is not the optimized dynamic deployment baseline.
- Q0 token-normalized updates with per-module gradient/update audits and exact
  optimizer/RNG/data-position resumption; one-packed-weight SH1 export/reload.
- Native-engine cohorts measure admission through final drain, fixed output work,
  TTFT/TPOT, loop depths, graph counters, and post-drain KV usage.
- Native activation traces use actual prepared request/position/loop metadata,
  retain physical weight storage identities, and separate prefill/decode norm
  and projection statistics. Padding is excluded. Tracing is diagnostic only.
- Fixed norm equalization across all consumers, a PyTorch-only reader for
  verified floating-point safetensors, document-split checks, masked next-token
  NLL, paired document PPL intervals, and paired throughput tables.

The fake-quant path reconstructs weights for numerical diagnosis. It cannot
supply low-precision serving speed or deployment-memory claims. The dense
training adapter does not use the inference cache's no-gradient writes.

## Validation recorded on 2026-10-05

The final full CPU regression run passed 454 tests, skipped 29, and deselected
113 GPU cases. Earlier dependency-loading stalls resolved naturally without
terminating either process. Ruff lint and formatting checks pass; the
`pre-commit` wrapper itself is absent and was not installed.

CPU checks cover encoding boundaries, tail groups, stable shared storage,
scale gradients, padding, loop ordering, document isolation, token alignment,
and statistical rejection of failed or unequal-work trials. The checkpoint
reader was compared exactly with safetensors for FP32, FP16, and BF16. Tiny Ouro
R1/2/3/4 outputs match the independent oracle, and the scale gradient propagates
through all four recurrent steps.

On one RTX 5090 (SM120), PyTorch 2.12.1+cu130 and CUDA 13.0, the native FP8
64×32 projection passed at 1/17/32/64/128 input rows. The maximum absolute
difference from the same quantized values was 0.001953125. CUDA Graph replay
passed after changing the stable activation-scale buffer; the packed weight
pointer stayed unchanged. These checks measured correctness, not throughput.

Official Ouro-1.4B revision
`574fa66cb8bf5abdc979642d01cf2b79b16bfab1` has been downloaded and independently
SHA256-verified. Two full-model FP32 comparisons stopped at prompt 8, loop 4:
one of 1,277,952 logits exceeded the fixed `atol=rtol=3e-4` criterion. Matching
the official rsqrt/operator order did not remove that discrepancy. A separate
FP64 reference check on the failing input passed at the unchanged tolerance:
native maximum error 0.000120017, adapter maximum error 0.000032414. The original
failures are retained. A subsequent 32-input FP64 comparison failed at prompt
15, loop 3 in the SDPA adapter: one of 2,310,144 logits differed by 0.000332700.
The complete follow-up audit found the BF16 adapter state/logits bitwise equal
to the official eager oracle at all 32 inputs and four depths. Native Torch
attention differs (25 logit argmax differences across 6,896 positions), and
the fixed FP32 criterion still fails. Both native backends preserve every old
KV prefix and match full-prefill argmax on all 512 incremental checks. Numerical
element tolerances still fail. A two-input diagnostic fixes every Linear to a
single-row GEMM and restores bitwise incremental/full equality on all 32 checks;
SDPA differences remain. The complete G0 remains open. Tolerances are unchanged.
See the [raw evidence and comparison table](experiments/loopquant-20261005/RESULTS.md).

| Model | Baseline tok/s | Candidate tok/s | Paired speedup | Status |
| --- | ---: | ---: | ---: | --- |
| Ouro-1.4B | — | — | — | Full semantics qualification in progress; no performance measurement |
| Ouro-2.6B | — | — | — | Not run |
| Huginn-3.5B | — | — | — | Not run |
| Nanbeige4.2-3B | — | — | — | Not run |

No official-model quantization quality, high-concurrency performance, or speedup
is claimed. FP8 fusion, per-row mixed-depth execution, official-model GPTQ/QAT,
GPU export qualification, and the remaining model adapters are subsequent work.

## Reproduction interfaces

The implemented CLI commands are `preflight` and `report`:

```bash
python -m loopquant.cli preflight --config experiment.json --output preflight.json
python -m loopquant.cli report --pairs trial-pairs.json --output results.md
python -m pytest tests/test_loopquant.py -m 'not gpu'
```

`experiment.json` contains full `model_revision` and `runtime_revision` commits,
an explicit `format`, and a `data_manifest` path. The manifest lists JSONL
window files under `files`. Windows record `document_id`, `source_revision`,
`split`, `start`, `token_ids`, and `content_sha256`. Calibration and QAT may
share registered training documents; dev and locked test cannot share them.

`trial-pairs.json` maps case names to `TrialPair` records. The estimator is the
median of paired throughput ratios. At least five independent pairs are needed
for the distribution-free one-sided ≥95% lower bound. Failed requests, captures
inside timing, unequal output work, and nonfinite throughput reject a pair.
These input checks do not establish independent runs or quality by themselves.

The intended serving matrix includes C1/8/32/64, short/long/decode-heavy inputs,
mixed depths, and open arrival rates. Final speed claims require exported-model
quality and matched native runtime features before measuring those workloads.
