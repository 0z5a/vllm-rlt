# Re:Loop H100 original L1 native result

The original W5 L1 case is complete on one H100: FP32/eager Triton, P4/R4,
B1/C1, observed A=S=1, P512/F128 and a common 48 GiB KV pool. Each trial
completes the original 4C requests with closed-loop refill.

| Policy | Mean tok/s | Improvement vs D4 | Two fresh improvements | Mean TTFT (ms) | Mean TPOT (ms) | Mean native latency (ms) |
|---|---:|---:|---:|---:|---:|---:|
| D4 true off | 24.48 | Reference | Reference | 252.99 | 39.18 | 5228.29 |
| D3 true off | 32.88 | +34.31% | +33.64% / +34.98% | 250.06 | 28.68 | 3892.53 |
| D3 guided ref1 w0.3 | 32.65 | +33.34% | +31.90% / +34.80% | 252.33 | 28.89 | 3920.86 |

The six fresh processes run in ABC/CBA order, each with five warmups and five
measurements. All 60 trials, 240 requests and 30,720 output tokens are retained.
The measured denominator is 30 trials, 120 requests and 15,360 tokens. Mean
TPS averages the two fresh process means; improvement is
`(candidate TPS / D4 TPS - 1) * 100%`. The two fresh improvements are not a
confidence interval. Guidance is 0.72% slower than the matched D3 off control.

| Policy | Peak allocated delta vs D4 | Peak reserved delta vs D4 | Peak reference state | KV saving |
|---|---:|---:|---:|---:|
| D4 true off | 0 | 0 | 0 | 0 |
| D3 true off | 0 | 0 | 0 | 0 |
| D3 guided | +8 KiB | 0 | 8 KiB | 0 |

Reducing decode depth retains all four prefill history planes. The configured
KV pool is identical across arms; this case establishes no memory saving or
FP8 result.

Independent review verified all 398 original source/raw/JIT/control files,
35 actual closure files, the unchanged twelve bound inputs and all 133
source Git blobs. The original six scientific children and their controller
returned naturally with zero. The six original scientific auditors passed
on the execution side and independently off device. Actual resource handoff
for the known experiment actors and visible owner readers was verified;
global reader retirement is not claimed. All original model input hashes
were retained, with the existing readonly weight body omitted from the archive.

The original transport returned 255. Its late archival and original deadline
breach remain recorded; later recovery did not rewrite the grant or rerun the
scientific case. Compiler argv was not traced; all thirteen original JIT ELF
files remain archived.

L1 is a low concurrency native baseline. The separate
[initial-total C32 result](H100_C32_FP32_RESULT-v1.md) does not qualify the
remaining original W5 cases. C128, multiple batch sizes, full-model HTTP,
Graph/async and all-model testing remain open. Original W4 quality remains
incomplete and uncertain; no quality noninferiority or external SOTA claim
follows from this cost result. Whole-task pass contribution remains zero.

[Scientific summary and immutable hashes](H100_L1_NATIVE_RESULT-v1.json).
This additive result follows the earlier preserved
[pending closure record](H100_L1_CLOSURE_PENDING-v1.md). Results stay on the
local branch.
