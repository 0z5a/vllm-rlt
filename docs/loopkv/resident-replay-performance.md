# Resident alias fixed-work results

The prefix-aware alias stack is slower than native resident Graph execution at
both registered points on both RTX 5090 cards. Removing promotion copies does
not produce a net speedup for these complete fixed schedules.

| GPU | B / C | Native Graph tokens/s | Alias + prefix Graph tokens/s | Paired speedup [95% CI] | Throughput change |
|---|---:|---:|---:|---:|---:|
| 5090 #0 | 16 / 32 | 387.05 | 380.45 | 0.9825× [0.9790, 0.9837] | -1.75% |
| 5090 #0 | 32 / 64 | 452.56 | 432.99 | 0.9568× [0.9440, 0.9641] | -4.32% |
| 5090 #1 | 16 / 32 | 385.93 | 379.58 | 0.9797× [0.9787, 0.9892] | -2.03% |
| 5090 #1 | 32 / 64 | 453.33 | 435.25 | 0.9575× [0.9363, 0.9625] | -4.25% |

Each row contains seven balanced pairs at P128/output limit128. Tokens/s columns
are arm medians; speedup is the median of paired native/candidate elapsed ratios,
so it need not equal the ratio of the displayed medians. The95% intervals resample
seven paired ratios10,000 times with seed1705, separately for each point/card.
They describe variation within one persistent weight session per card, not
independent startup sessions; no multiple-comparison adjustment is applied.

The frozen source is `16d9aeb4c28c15094be93b32dfc8014f1f169c66`, using official
Ouro-1.4B revision `574fa66cb8bf5abdc979642d01cf2b79b16bfab1`, BF16/Triton,
Torch2.12.1+cu130, delayed gate0.5 with min2/max4 and greedy seed17. One complete
warmup precedes each measured arm on that engine. Measurement includes HTTP-free
request submission, admission, real model/gate kernels, replay scheduling, equal
CPU work counters and complete drain. Loading, trace parsing, detailed comparisons
and file writes are outside the timer.

Both GPU locks and the original heavy-IO lock were held for the entire sequential
two-card run; peer jobs explicitly yielded their windows. Controller and both
children naturally returned0. An independent local audit verified all input
hashes and compared every warmup/measured request object with the external captured
reference. All56 measurements/2,688 requests/344,064 output tokens match exactly;
the same number of warmup requests also pass. Work and Graph coverage match within
every pair, and no measured arm adds a capture.

| B / C | Recurrent rows per arm | Actual B mean | Recurrent calls | Measured Graph replays | Eager fallbacks |
|---|---:|---:|---:|---:|---:|
| 16 / 32 | 13,167 | 8.6568 | 1,521 | 1,521 | 0 |
| 32 / 64 | 26,317 | 22.5510 | 1,167 | 383 | 784 |

GPU UUIDs are `GPU-209bed64-f11a-c1f3-fe87-8a69538d8d79` (#0) and
`GPU-84c4d7e9-b1c7-5d12-e7c2-6ec2a136044d` (#1). The B32 point retains the
frozen16-entry Graph cache and its observed fallbacks in both arms. Per-arm clocks,
power, temperature, allocator peaks, raw hashes, exact orders and every ratio are
in the [result evidence](evidence/resident-replay-performance-v2.json).

This compares the complete alias stack against native Graph execution. It does
not isolate the prefix optimization against its parent alias reader. It also does
not establish live-serving throughput, SLO/P99 behavior or natural-schedule quality
equivalence. The [separate delayed quality differences](delayed-quality.md) remain
part of the result; forcing the same logical schedule does not erase them.

The independently matched raw archive SHA256 is
`b6135c0cc45d26405a69d5a346cdea4ce4d56e5fd7616ea0368628836d0c6863`.
