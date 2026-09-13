# BF16 inference and numerical validation policy

Effective 2026-09-12 for new work. This policy supersedes the FP32-first
requirements in the original roadmap and milestone issues #2–#9. Dated results,
frozen JSON contracts, raw evidence, and decisions under those contracts remain
historical records. This policy does not turn any previous failure into a pass.

## Inference precision

Use original Ouro guidance first: an explicit prescription in the paper, then
the pinned checkpoint configuration and released inference code. When those
sources leave accumulation unspecified, record the chosen backend behavior
and validate it. The [source audit](paper-notes.md#original-ouro-paper-and-precision-guidance)
finds BF16 in the release configuration and explicit FP32 operations in the
model code; the paper itself does not specify a per-kernel inference recipe.
Project precision choices must be labeled separately from author guidance.

BF16 is the primary GPU inference, kernel-validation, task-quality, and
performance target for Ouro. Use BF16 weights, ordinary activations, and KV
storage. Full-model FP32 is an optional diagnostic/reference configuration;
passing it is not a prerequisite for starting BF16 experiments.

Specify storage/input/output dtype separately from intermediate and accumulation
precision. BF16 inference can use FP32 inside selected operations:

| Operation | Precision requirement |
| --- | --- |
| Projections, MLP and LM head | BF16 operands and outputs; record the backend's accumulation/reduction mode. FP32 accumulation is compatible with BF16 inference. |
| Attention | BF16 Q/K/V and output. The official eager path uses native-dtype QK/PV products and FP32 softmax cast back to query dtype; fused-kernel FP32 reductions/statistics/accumulation are explicit backend choices to validate. |
| RMSNorm | Retain FP32 variance/reduction arithmetic, returning to the activation dtype at the defined boundary. |
| RoPE | Retain FP32 frequencies, phase and trigonometric computation; cast the positional factors to the activation dtype as specified by the model. |
| Gate and sampling probabilities | Use the pinned release as the reference. Its exit-distribution code has no explicit FP32 promotion; the current runner's FP32 sigmoid/host cumulative update is a project difference to validate. Record the sampling implementation separately. |
| Cache metadata and KV copies | Integer addresses/lengths; copy BF16 KV without changing its values. |

Promote only the operations/intermediates justified by their numerical behavior.
Do not upcast the whole model to satisfy a reference comparison, or force all
intermediates to BF16 merely because the model dtype is BF16. An accumulation
change is its own declared experimental variable. The current attention backend
uses FP32 internal arithmetic on BF16 inputs; this document does not claim it
already implements a BF16 Tensor Core attention kernel.

This distinction follows [PyTorch's mixed-precision description](https://docs.pytorch.org/docs/2.14/amp.html)
and its discussion of [numerical accuracy and reduction precision](https://docs.pytorch.org/docs/2.14/notes/numerical_accuracy.html).
Library defaults are version-dependent; record the actual flags and backend.

## Validation requirements

Keep three questions separate:

1. **Exact state and data movement.** Cache ownership, address mapping, causal
   initialization, LAST-EXITED copies, inactive-row isolation, and request RNG
   ownership remain exact invariants. A metadata-only change with identical
   floating-point work must preserve selected tensors, populated KV, tokens,
   exit decisions, and RNG progression exactly in matched BF16 A/B tests.
2. **Floating-point fidelity.** Compare BF16 kernels/model execution with an
   independent reference on identical inputs and histories. Record operand,
   accumulation, output, rounding, and backend choices for both sides. Use
   predeclared, independently justified error bounds appropriate to the
   operation, shape, and recurrence depth. Bitwise agreement across different
   arithmetic implementations or with full-model FP32 is not a general gate.
   A pinned official implementation remains a reference check; its eager
   rounding sequence does not automatically define every optimized kernel.
3. **Decisions and quality.** Record actual token/exit disagreements, top-two
   margins, and distance to gate thresholds. For changed arithmetic, use a
   predeclared decision/quality policy instead of requiring universal exact
   token agreement. Near ties are reported and assessed under that policy,
   never exempted after the run. Investigate substantial errors and decisions
   away from boundaries. Match hidden/KV state for local diagnosis; once live
   histories diverge, use separately controlled histories for tensor matching.
   Task quality uses a frozen paired protocol with a loss budget and uncertainty.

Finiteness, valid histories, memory safety, and cleanup remain required. An
unexplained large numerical discrepancy is not dismissed as normal rounding.
Retain Q1's original FP32/BF16 bounds, failures, and reports under their original
policy identity. A successor BF16 policy needs its own rationale, version, and
frozen acceptance limits before execution; missing evidence remains unresolved.

## Performance and milestone requirements

New M1–M4 baselines and optimization comparisons use BF16 on both sides with
the same accumulation policy, hardware, workload, cache capacity, and timing
boundaries. Add FP32 only for a stated diagnostic question. Compare BF16 A/B
directly; the historical FP32 M2 gain cannot establish a BF16 speedup. Recompute
memory budgets from the selected dtype: the 1,024-page Ouro pool occupies
3 GiB in BF16 and 6 GiB in FP32. Freeze page capacity and report bytes.

BF16 profiling, optimization A/B, and Q2 quality data collection may proceed
while the original Q1 diagnosis is open, after the new run's functional,
finiteness, and ownership prerequisites pass. Report performance observations,
optimization regression checks, reference fidelity, and task quality separately.
Diagnostic measurements with unresolved reference fidelity cannot establish
validated serving correctness; an unrelated old gate must not force all work
back to FP32. Promotion requires the new comparison's numerical, behavioral,
quality (when claimed), and performance gates.

Q2 uses fixed-depth BF16 as the primary quality baseline and compares adaptive
BF16 with it; full-model FP32 is an optional sensitivity control. First establish
native/official fixed-depth accuracy with the original paper's GSM8K 3-shot CoT,
strict-match, lm-eval-harness protocol. Freeze unspecified task/version/template
and generation settings explicitly, as detailed in the
[accuracy source notes](paper-notes.md#accuracy-evidence-and-the-original-evaluation-protocol).
The existing 4/64 FP32 screen is neither a BF16 accuracy pass nor a reproduction
of the paper. External speed comparisons use matched BF16 execution and an
explicit workload-equivalence
contract: fixed prompt/output lengths, depth policy, cache behavior, and timed
work. Report actual token differences under the numerical/decision policy;
they do not automatically invalidate a matched-work timing comparison. A claim
of identical generation still requires matching actual histories. Forced-token
replay is a separately labeled workload, not generation quality.
M5 retains its independent gate-availability, calibration, and lifecycle gates.

## Implementation follow-up

This is a requirements change. README GPU examples select BF16 explicitly;
the current API/CLI default is still FP32. The historical M1 schema rejects
non-FP32 contracts and M2's frozen workflow uses Q1's original FP32 gates.
Changing only a JSON dtype or command line is not an executable BF16 benchmark.
The next implementation must:

- Add BF16 as the standard GPU workflow and a versioned BF16 benchmark/validation
  contract, updating schema/runner checks while retaining historical replay.
- Record and validate operand/storage/accumulation precision independently.
- Freeze justified BF16 kernel/model and decision gates, including exact
  metadata-only A/B checks and independent LAST-EXITED invariants.
- Run a bounded BF16 M2 A/B with matched controls, numerical checks, profiles,
  and separately reported quality/reference limitations.

Those implementation and execution requirements remain outstanding. No kernel,
runtime default, frozen experiment, tolerance, or measured result changes in
this documentation revision.
