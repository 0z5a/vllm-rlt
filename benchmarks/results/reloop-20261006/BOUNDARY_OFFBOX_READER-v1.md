# Boundary tensor reader validation

The Huginn/Parcae offline state/logit auditor is prepared. Independent source
review checked the complete five-file archive and four payload hashes. It
reads CPU Torch ZIP storage with NumPy and restricted stdlib reconstruction;
Torch, CUDA and models were not imported or executed.

| Actual reader check | Result |
|---|---|
| Existing authentic CPU ZIP and original metadata whole SHA | Pass |
| Two original int64 input-ID hashes and shapes | Pass |
| 36 existing float arrays decoded | Pass; old scientific comparisons were not recomputed |
| Known BF16 bits, signed zero and strided view | Pass |
| Exact and relative-budget pass/fail controls | Pass |
| New Huginn/Parcae tensor exports audited | 0; data unavailable |

The prepared auditor checks all original request/chunk/depth records, tensor
hashes, shapes, IDs and paired states/final logits. Its documented combined
0.02/0.02 bound uses float64 arithmetic on decoded float32 operands; maximum
absolute error uses float32 subtraction. Any disagreement with original
producer records remains an audit failure.

This validates reader compatibility and its comparison helper. It supplies no
new model numerical qualification, independent KV tensor replay, high-C
serving, quality or performance result. New model execution and complete
source/raw provenance, natural waits and actual handoff remain required.
Whole-task pass contribution is zero.

[Immutable source and fixture hashes](BOUNDARY_OFFBOX_READER-v1.json).
