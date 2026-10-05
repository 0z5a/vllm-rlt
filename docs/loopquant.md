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
- A fixed-loop Nanbeige adapter with its two-norm residual order and optional
  final-readout norm; both adapters share projection statistics and Q0 updates.
- Native FP8 scalar-scale GEMM through `torch._scaled_mm`. Padding and conversion
  remain in the operation. This initial unfused implementation is a correctness
  backend; it is not the optimized dynamic deployment baseline.
- Q0 token-normalized updates, producer-shared scales, deterministic training
  pools, optimizer/RNG/data-position resumption, and native encoded export.
- Fused Ouro/Nanbeige FP8 producers for static, staged, and dynamic row scales, with native
  mixed-depth serving, explicit GPU loop IDs, and one packed weight per projection.
- Native INT4 W4A16 packing for PyTorch's installed tensor-core backend; 20
  native shape checks pass on SM120. No optional package is installed.
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

The latest completed full CPU regression passed 492 tests, skipped 29, and
deselected 196 GPU cases, including the Q0 deployment and data checks. Earlier dependency-loading stalls resolved naturally without
terminating either process. Ruff lint and formatting checks pass; the
`pre-commit` wrapper itself is absent and was not installed.

CPU checks cover encoding boundaries, tail groups, stable shared storage,
scale gradients, padding, loop ordering, document isolation, token alignment,
and statistical rejection of failed or unequal-work trials. The checkpoint
reader was compared exactly with safetensors for FP32, FP16, and BF16. Tiny Ouro
R1/2/3/4 outputs match the independent oracle, and the scale gradient propagates
through all four recurrent steps.

Subsequent targeted checks pass 47 CPU tests. The Nanbeige adapter matches the
unmodified pinned official class bitwise in eight tiny-weight cases: FP32/BF16,
one/two loops, and both loop-final-norm settings. These do not replace complete
checkpoint qualification. Native fixed-loop/masked outputs and two Q0 updates
also pass; the Q0 teacher preserves the student's architecture and attention.

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
| Ouro-2.6B | — | — | — | Official CPU exact; GPU/E2E pending |
| Huginn-3.5B | — | — | — | Full official CPU comparison running |
| Nanbeige4.2-3B | — | — | — | Full verified checkpoint; official CPU queued |

The fused FP8 primitives pass 120 cases and three CUDA Graph cases on SM120;
12 tiny native serving combinations and 16 native export/reload combinations
also pass. These qualify the tested operations, not complete model quality or
high-concurrency speed. Official-model GPTQ/QAT, full-checkpoint export quality
and final serving results remain pending.

## Reproduction interfaces

The implemented CLI commands are `preflight`, `train` (Q0), and `report`:

```bash
python -m loopquant.cli preflight --config experiment.json --output preflight.json
python -m loopquant.cli report --pairs trial-pairs.json --output results.md
python -m loopquant.cli train --config q0-run.json --level Q0 --output run-q0-20
python -m pytest tests/test_loopquant.py -m 'not gpu'
```

`experiment.json` contains full `model_revision` and `runtime_revision` commits,
an explicit `format`, and a `data_manifest` path. The manifest lists JSONL
window files under `files`. Windows record `document_id`, `source_revision`,
`split`, `start`, `token_ids`, and `content_sha256`. Calibration and QAT may
share registered training documents; dev and locked test cannot share them.
The [Q0 schedule](experiments/loopquant-20261005/Q0_SCHEDULE.md) records exact
sampling budgets, configuration fields, continuation, and deployment semantics.

`trial-pairs.json` maps case names to `TrialPair` records. The estimator is the
median of paired throughput ratios. At least five independent pairs are needed
for the distribution-free one-sided ≥95% lower bound. Failed requests, captures
inside timing, unequal output work, and nonfinite throughput reject a pair.
These input checks do not establish independent runs or quality by themselves.

The intended serving matrix includes C1/8/32/64, short/long/decode-heavy inputs,
mixed depths, and open arrival rates. Final speed claims require exported-model
quality and matched native runtime features before measuring those workloads.
