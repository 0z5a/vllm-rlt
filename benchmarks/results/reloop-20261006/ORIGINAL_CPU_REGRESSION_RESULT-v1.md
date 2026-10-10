# Original tiny CPU regression result

The original frozen CPU regression completed with 118 unique cases passed,
zero failures/errors/skips and both original natural returns zero. It tested
source `2597e7b992a5ec490543dddee2332a0072f6bbc2` with Python 3.12.14,
Torch 2.13.0 and pytest 9.1.1. CUDA stayed uninitialized; this scope used zero
H100s.

| CPU module | Passed cases |
|---|---:|
| Speculative decoding | 50 |
| Engine | 32 |
| KV cache | 19 |
| Prefix growth | 7 |
| Async pipeline | 10 |
| Total unique CPU cases | 118 |

All 26 GPU cases were deselected. Independent review verified the complete
XML case IDs and required lifecycle cases, all 150 evidence payloads before
and after review, 88 frozen execution sources, and all 132 source files against
the original archive and actual Git blobs. The original independent audit
also returned naturally with zero. Original tests were not rerun; immutable
pending markers are retained alongside the separate actual terminal receipts.

This suite establishes correctness for its frozen tiny CPU source. The
separate canonical e8 17-case contract, full-model H100 high-C/multiple-BS,
HTTP/Graph/async and original quality/cost gates remain open. This suite
contains no measured generation speed or GPU-memory comparison. Whole-task
pass contribution remains zero.

[Exact scope and evidence hashes](ORIGINAL_CPU_REGRESSION_RESULT-v1.json).
Results stay on the local branch.
