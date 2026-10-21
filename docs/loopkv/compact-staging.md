# Compact metadata staging

Compact traversal descriptors now use two pinned host transfers: int64 addresses
and offsets, then int32 widths, depths and context lengths. All are read-only
views of those buffers during the synchronous traversal. Record division,
remainder, cast and context-length addition are computed from existing host
values before submission, removing four pointwise GPU operations. The record
publication kernel still runs on the same stream before layer writes.

This is an incremental change to compact storage with the fixed-reservation
admission correction. It does not change payload layout, canonical version
selection, credit policy or reserved memory.

| Check | Result |
|---|---|
| CPU compact/runtime/cache/engine suite | 102 passed, 2 CUDA skips, 36 subtests |
| CUDA attention / payload | 46 bitwise / 64 exact checks passed |
| CUDA skipped maps and unused payload poison | Passed |
| CUDA stale request descriptor | Rejected after free/reuse |
| Tiny CUDA engine B1/4/16/32/64/128, C=2B | Exact tokens and exits |
| Official Ouro-1.4B P128/D128 B limit64/C128, fixed6GiB arena | All128 requests /16,384 output tokens and exits exactly match native |

[Raw CUDA gate](evidence/compact-staging-gpu-v2.json) runs at frozen source
`a5e74191529715e6657dfdd5f335592cd2c82b85` on RTX5090, Torch2.12.1+cu130,
Triton3.7.1. Independent timed staging comparisons remain pending.
The official capture uses descendant source
`cf70584d60913e1664439fb1b4acf603e2dff105` with credit reclamation disabled.
Native and staging have identical full request objects and actual batch/exit
histograms, maximum32 residents, and all2048 block credits return after drain.
The repeated prompt fixture and per-step capture are a mechanism diagnostic,
not a quality score or speed comparison. [Official manifests and receipts](evidence/compact-staging-official-v1.json).

The corrected compact quality result in [the quality report](quality.md)
uses the original five-transfer descriptor and must not be counted as validation
of this descriptor optimization.

| P128/D256 B32/C64, Ouro-1.4B | Native tokens/s | Compact tokens/s | Speedup | Scope |
|---|---:|---:|---:|---|
| Original compact prototype | 699.31 | 682.72 | 0.974× | Earlier 3-pair pilot, before admission correction |
| Two-transfer descriptors | — | — | — | Pending; compare separately to corrected compact and Native-S |
