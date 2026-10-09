# Re:Loop original T32 whole-case admission forecast

The next original native case retains B32/C32, P512/F128, FP32 P4/R4,
closed-loop 4C refill and the common 48 GiB KV pool. This is an empirical
admission forecast; T32 has not run and has no measured speed result.

| Required scope | Original denominator |
|---|---:|
| Fresh processes, reversed three-arm order | 6 |
| Warmup / measured trials per fresh process | 5 / 5 |
| Requests / output tokens per trial | 128 / 16384 |
| All retained trials / requests / tokens | 60 / 7680 / 983040 |
| Measured trials / requests / tokens | 30 / 3840 / 491520 |

The complete prior C32 archive supplies all sixty original trial traces and
six full process intervals. Its largest full interval was 256.630454 seconds,
including model initialization, ten trials and serialization. The original
32-request trial becomes 128 requests; the estimate assumes no additional
throughput gain from increasing B16 to B32.

| Forecast component | Seconds |
|---|---:|
| One fresh process: ceil(256.630454 × 4 × 1.5 / 60) × 60 | 1560 |
| All six fresh processes | 9360 |
| Natural drain reserve | 600 |
| Complete off-device archive reserve | 900 |
| Minimum complete window | 10860 |

The original controller continues to use the greater of the frozen estimate
and observed child elapsed time ×1.5 before admitting each remaining process.
It preserves incomplete dispositions when the whole remaining scope does not
fit. A current numerical/source/model/resource binding and a new complete
window remain required; the earlier L1 grant is unchanged.

All seven W5 cases and 820 canonical inputs remain byte-identical. The estimate
does not prove resident A, decode S, T32 runtime, performance or completion.
Whole-task pass contribution remains zero.

[Exact input hashes and forecast arithmetic](T32_WHOLE_CASE_ADMISSION_FORECAST-v1.json).
