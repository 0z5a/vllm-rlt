# Ouro1.4B native engine pilot

All eight finite arms completed on one RTX5090 with the second GPU reserved idle.
These are measured smoke E2E results, not G3 or quality-qualified speedups:
each arm has only64 requests and one trial, with5.29–12.32seconds measured.
No 60-second steady window, confidence bound or locked quality result exists yet.

Fixed workload: C32, prompt256/output32, R4, same frozen token trace,32 warmup
requests, BF16 state/KV and20GiB KV allocation. Batch token caps are128 and512;
these are token budgets, distinct from request concurrency. Async/multistream,
static buckets and CUDA Graph are enabled for every policy. The order is
BF16/DYN/SH1/ST2 at128, then ST2/SH1/DYN/BF16 at512.

| Batch token cap | Policy | Output tok/s | Vs BF16 | Speed change vs BF16 | Vs DYN | Speed change vs DYN | TPOT p95 (ms) | SLO coverage |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 128 | BF16 | 180.608 | 1.0000× | +0.00% | 1.0795× | +7.95% | 104.72 | 85.94% |
| 128 | DYN | 167.313 | 0.9264× | -7.36% | 1.0000× | +0.00% | 165.58 | 28.12% |
| 128 | SH1 | 166.168 | 0.9200× | -8.00% | 0.9932× | -0.68% | 166.70 | 28.12% |
| 128 | ST2 | 167.968 | 0.9300× | -7.00% | 1.0039× | +0.39% | 165.59 | 28.12% |
| 512 | BF16 | 387.146 | 1.0000× | +0.00% | 1.0651× | +6.51% | 77.47 | 100.00% |
| 512 | DYN | 363.494 | 0.9389× | -6.11% | 1.0000× | +0.00% | 82.31 | 100.00% |
| 512 | SH1 | 361.615 | 0.9341× | -6.59% | 0.9948× | -0.52% | 82.94 | 100.00% |
| 512 | ST2 | 359.988 | 0.9299× | -7.01% | 0.9904× | -0.96% | 83.55 | 100.00% |

Throughput includes admission, queueing, prefill, decode and drain. Every arm
completed2,048 output tokens, all at depth4, with failed requests0, KV after
drain0, and no measured Graph captures or fallbacks. Graph replays range
from362 to549. The preregistered per-request SLO is TTFT≤10,000ms and
TPOT≤100ms. B128 misses it for14.06% of BF16 requests and71.88% of FP8
requests; every B512 request meets it. These fractions are reported rather
than treating raw throughput as goodput.

SH1 and ST2 do not show a material advantage over the dynamic fused baseline
in this pilot. All FP8 arms are slower than BF16 here. Packed weights reduce
peak allocated memory from22.81/22.88GiB (BF16 B128/B512) to21.71/21.80GiB
(FP8); the same20GiB cache reservation dominates these totals. No inference
about quality follows from completion or memory reduction.

DYN uses per-row dynamic input scales. SH1 and ST2 use fixed scalar/stage
scales from the128-window calibration, shared across prefill/decode via their
maximum; ST2 has the fixed boundary2. This compares deployable policies with
different activation granularity, so changes cannot all be attributed to
removing dynamic reduction. No dev/test-guided clipping or boundary search
was performed.

The BF16 B128 arm averages14.46 core rows, versus17.29 for its FP8 arms; all
B512 arms average21.92. This is an observed scheduler interaction, not a
claim of equivalent kernel traces. Longer paired trials and cost profiling
remain necessary before a performance decision.

Runtime source `290be852825c7cd7a1592ebb9853dfe9f774ca01`, Ouro revision
`574fa66cb8bf5abdc979642d01cf2b79b16bfab1`, Torch2.12.1+cu130/CUDA13.0.
The [summary](ouro-pilot-v1-summary.json), [41 raw files and hashes](pilot-v1/ouro-split-pilot-file-manifest.json),
[child receipt](pilot-v1/ouro-split-pilot-receipt.json) and
[complete handoff](ouro-split-pilot-complete.json) retain provenance.
All children, the controller and SSH exited naturally0; both GPUs and the
original lock files were independently checked before handoff.
