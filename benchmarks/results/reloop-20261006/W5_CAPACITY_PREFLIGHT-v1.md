# FP32 R4 capacity preflight for the original W5 scope

This CPU calculation uses the archived official Ouro config and the current FP32/Triton cost protocol. Native E2E for these exact cases remains NOT_RUN. It preserves the original short, mixed, long, and larger-memory targets.

Each computed token/depth uses 393,216 KV bytes. A 16-token group across all four P4 history planes uses 24 MiB. The common 64 GiB pool holds 2,730 complete groups after rounding. Reduced D3/D2 does not halve the four-plane reservation.

| Original case | B / C | Target A in calculation | Input tokens | R4 KV GiB | Static status | Actual A / S | Native E2E |
|---|---:|---:|---:|---:|---|---|---|
| L1 | 1 / 1 | 1 | 512 | 0.938 | Fits this pool | NOT_OBSERVED | NOT_RUN |
| T32 | 32 / 32 | 32 | 512 | 30.000 | Fits this pool | NOT_OBSERVED | NOT_RUN |
| Q64 | 32 / 64 | 32 | 512/2048 mixed | 66.000 | Target first32 exceeds pool; queue unmeasured | NOT_OBSERVED | NOT_RUN |
| T64_short | 64 / 64 | 64 | 256 | 36.000 | Fits this pool | NOT_OBSERVED | NOT_RUN |
| C8_long | 8 / 8 | 8 | 2048 | 25.500 | Fits this pool | NOT_OBSERVED | NOT_RUN |
| C4_longer | 4 / 4 | 4 | 8192 | 48.750 | Fits this pool | NOT_OBSERVED | NOT_RUN |
| A32_large | 32 / 32 | 32 | 2048 | 102.000 | PLANNED_CAPACITY_EXCEEDED | NOT_OBSERVED | NOT_RUN |

Q64 uses an example alternating 512/2048 order only. Its first contiguous 32-resident group would need 66 GiB; its queued64 workload can still be measured with different actual residency. This calculation predicts neither the peak A nor an actual admission rejection. The final mixed trace and arrival rule must be frozen before execution.

All positions are within the archived model limit of65,536. Fits this pool is a KV-only bound; weights, activation, reference/scratch, graph memory and actual allocator admission still need their own checks. A32/P2048/output128 needs102 GiB of FP32 R4 KV, so it is preflight-rejected under this64 GiB allocation. This is not an observed OOM.

The original default is closed-loop4C. The separately frozen current candidate-cost experiment uses initial totalC and cannot provide refill evidence. The currentP512/B16C32 first baseline and its plannedB32C64 follow-up do not replace the exact mixed and long-context cases above. Per-hardware baselines, native request metrics, all raw trials, and two fresh starts remain required for the comparisons claimed.

Source: e8aa1ee57734bc5c1118c143ebdb68d0b53210a1; config SHA-256 ce9cc13da41591b8b4deca053d7dfee06424c0228628ee862ea86d725bc163f3.
Raw calculation: [w5-fp32-capacity-preflight-v1.json](w5-fp32-capacity-preflight-v1.json), SHA-256 5e52482e1257dc7ded100d3146ffacdb0bfbcd25cfc7fd9b11f8bc8e8fd4867f.
