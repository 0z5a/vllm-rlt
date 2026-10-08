# Re:Loop H100 C32 native fixed-work result

One H100, FP32/eager Triton, P4, B16, initial-total C32, P512/F128 and the same48GiB KV pool in every arm. Actual resident A and active decode S were16; observed initial outstanding C was32. This is a native instrumented fixed-work test, separate from steady refill, C128, HTTP, engine-async and CUDA Graph tests.

| Policy | Mean TPS | Improvement vs D4 | Two fresh improvements | Mean TTFT (ms) | Mean TPOT (ms) | Mean native latency (ms) |
|---|---:|---:|---:|---:|---:|---:|
| D4 true off | 179.28 | +0.00% | Reference | 9133.83 | 63.18 | 17157.87 |
| D3 true off | 212.99 | +18.80% | 18.26%–19.34% | 8191.65 | 49.25 | 14446.54 |
| D3 guided ref1 w0.3 | 207.10 | +15.52% | 15.39%–15.64% | 8371.48 | 51.02 | 14851.28 |

Each arm has two fresh processes in reversed ABC/CBA order, each with five warmups and five measurements. All six processes/60 trials/1920 requests/245760 output tokens were retained; measured scope is30 trials/960 requests/122880 tokens. TPS is the arithmetic mean of the two fresh process means. Improvement is `(candidate TPS / D4 TPS − 1) × 100%`; the two-fresh range is not a confidence interval. Both guided fresh ratios exceeded the original frozen1.15 cost-reference target.

| Policy | Peak allocated delta vs D4 | Peak reserved delta vs D4 | Peak reference state |
|---|---:|---:|---:|
| D4 true off | 0 | 0 | 0 |
| D3 true off | 0 | 0 | 0 |
| D3 guided | +128KiB | +2MiB | 128KiB |

Guidance does not save KV or allocated memory in this common-pool test. All arms retain the same48GiB pool. Mean per-trial native ITL P99 was303.68ms for D4,241.91ms for D3 off and246.32ms for D3 guided; these are means of native trial P99 values, not pooled HTTP P99.

The original immutable auditor passed on the execution side and again independently off device with identical parsed results. Independent review checked all412 archived source/raw/JIT/control files and recomputed2100 request-metric scalar values from original timestamps. All scientific processes joined naturally with0. Actual handoff for the known experiment actors and original resource ownership was subsequently verified; this does not claim global reader retirement.

All11 model input byte hashes remained unchanged. The first collector failed before tar creation on the type of a reused weight alias; corrected additive link-type metadata and both attempt logs were retained. Scientific inputs/results were not changed or rerun, and no all-regular physical-hardlink claim is made.

Original W4 remains incomplete with132 missing matched-off code outputs and uncertain primary quality comparisons. This test establishes this fixed-work performance result, not quality non-inferiority, VRAM saving, external SOTA, full W5, C128, HTTP/Graph or all-model completion. Whole-task pass contribution remains zero.

[Scientific summary and immutable hashes](H100_C32_FP32_RESULT-v1.json). CUDA13.0 and necessary dependency versions are recorded there; other cluster metadata stays private. Results remain on the local branch.
