# Full-prefix tiles in resident alias attention

The resident alias path previously set each row's first-alias position to zero,
so every tile used canonical-depth resolution. It now snapshots the host's known
first-alias position with the routing descriptor. Complete tiles before that
position use the direct reader; the boundary tile and later tiles still resolve
canonical depths. This uses the hybrid reader already qualified by the parent.

Alias descriptors contain four integers instead of three. Native descriptors
remain three integers. The metadata kernel copies the snapshot into the bank's
owned device buffer, and Graph replay copies it into stable entry storage. The
bank's last-reader event protects the snapshot; the existing request event still
orders attention after finalized exit-map publication. No new device readback or
allocation is introduced per recurrent layer.

| Validation | Result |
|---|---|
| Affected CPU async/alias/accounting suite | 29 passed, 12 CUDA skips |
| CUDA B1–128/C2B, mapped-host/H2D, five execution modes | All 60 tiny-model arms match native synchronous output/exit sequences |
| Full-prefix and mixed-tail coverage | P33/P34 crosses the 32-token attention tile; cancellation and reuse over two rounds |
| Cross-stream allocation reset and bank retirement | Both descriptor modes pass |
| Official fixed-work comparison | 28 pairs on two RTX5090 cards; every warmup/measured request exact |

The CUDA process naturally exited0 at source
`a82d4021acdc47726ec931621b7ddd9b47382ff8` on RTX5090/Torch2.12.1+cu130.
[All cases, Graph fallback counters and independently verified raw hash](evidence/resident-prefix-cuda-v1.json).

| GPU | B / C | Native Graph tokens/s | Alias + prefix Graph tokens/s | Paired speedup [95% CI] | Throughput change |
|---|---:|---:|---:|---:|---:|
| 5090 #0 | 16 / 32 | 387.05 | 380.45 | 0.9825× [0.9790, 0.9837] | -1.75% |
| 5090 #0 | 32 / 64 | 452.56 | 432.99 | 0.9568× [0.9440, 0.9641] | -4.32% |
| 5090 #1 | 16 / 32 | 385.93 | 379.58 | 0.9797× [0.9787, 0.9892] | -2.03% |
| 5090 #1 | 32 / 64 | 453.33 | 435.25 | 0.9575× [0.9363, 0.9625] | -4.25% |

[Complete method, source, independent audit and limits](resident-replay-performance.md).
This is the whole stack versus native Graph, not an incremental parent-alias comparison.
