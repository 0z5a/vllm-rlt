# Re:Loop current H100 original numerical checks

The complete original FP32/eager protocol ran on one H100 with unchanged source, model and numerical criteria. Independent review verified every archived input, source, raw result and executable byte, then recomputed the emitted scalar pairs. This establishes numerical correctness within the original protocol; the actual handoff for the known experiment actors was subsequently verified. The complete C32 performance test has now actually started with a fresh verified ready; full results and audit are pending. Numerical success is not a complete high-concurrency E2E pass.

| Check | Required | Observed |
|---|---:|---:|
| Original numerical cases | 12 | 12 pass |
| Readouts | 108 | 108 |
| Forced continuation inputs | 96 | 96 |
| Ownership/lifecycle checks | 21 | 21 |
| Historical key/value comparisons | 82944 each | 82944 each |
| Independently recomputed selected-logprob pairs | 108 | 108 |
| Exact retained readout states | 96 | 96 |
| Absolute/relative tolerance | 1e-4 / 1e-4 | Unchanged |
| Archived payload files independently checked | 242 | 242 |

The immutable original test executed tensor assertions; this independent review checked source/control provenance and emitted scalar evidence, without claiming a second tensor replay. CUDA 13.0 and the necessary dependency versions are in the [scientific summary](H100_CURRENT_NUMERICAL_RESULT-v1.json). Model/source revisions remain pinned.

| H100 speed comparison | Result |
|---|---|
| D3 off vs D4, complete fixed work | Actual complete C32 started; results pending |
| D3 guided vs D4, complete fixed work | Actual complete C32 started; results pending |
| Original W5 and separate C128 refill cases | NOT_RUN |

There is no new speed, memory saving, HTTP/Graph serving, quality non-inferiority or SOTA claim. The original confirmation still has 132 missing code outputs and uncertain quality comparisons. Whole-task pass contribution remains zero. Results are held on the local branch.

The C32 test preserves B16, initial-total C32, P512/F128, six fresh processes, 60 trials, 1920 requests and245760 output tokens. This is distinct from configured-C128/refill and HTTP tests. Its first-start ready and immutable source bindings were checked independently; no partial trial is presented as a complete speed comparison.
