# M2: reuse KV metadata across a recurrent traversal

[M2 #5](https://github.com/hsliuustc0106/vllm-lt/issues/5) selects the repeated
metadata work identified in [M1's profile](benchmarks/m1-20260910.md). Its W1
capture contains 64 recurrent traversals, 1,536 physical-layer calls and 6,144
KV-metadata HtoD copies/stream synchronizations, plus 64 position transfers.
These counts nominate the change; they do not establish an end-to-end benefit.

`OuroForCausalLM.recurrent()` now asks the cache manager for one private
`_PreparedKVBatch`. It reuses the descriptor's positions for RoPE and passes the
same descriptor through all physical layers. Each prefill depth is a separate
traversal with its own descriptor. Attention arithmetic, gates, sampling,
scheduling, full-lifetime reservation and LAST-EXITED propagation are unchanged.

The cache manager owns row/address construction. A descriptor contains immutable
host rows and allocation identities, and private device tensors for positions,
write addresses, block tables and context lengths. It lives for one synchronous
traversal; it is not stored on requests, the model or a cross-step cache.
Prepared reads/writes reject a different manager or a released/reallocated
request. Every physical layer still validates its tensor inputs and initialized
causal prefix, and publishes written positions only after its own KV writes.

Existing public cache methods remain adapters. Standalone calls construct the
whole descriptor, including metadata unused by a lone write/read. Read-only
preparation permits duplicate query rows; writes reject duplicate addresses.
The performance candidate is the model's shared traversal path. Batched sample
readback and fixed-depth gate-transfer removal are unselected, out of scope.

## Frozen comparison workflow

The checked-in `benchmarks/fixtures/ouro-m2-contract.json` fixes the candidate,
input references, exact run budget and numeric gates. A CPU probe resolves the
actual A/B source commits, common harness bytes, local model/tokenizer contents,
one physical GPU ID and CPU/NUMA policy before execution. Null template controls
cannot produce an executable plan. Both execution checkouts must be clean;
their only file differences are the cache manager and native Ouro model.

The baseline retains the original production metadata path and includes the
same benchmark/validation harness as the candidate. The harness reuses M1's
inference executor, timing boundaries and offline metric reconstruction.
It verifies source/import paths and controls in each worker. It never switches
implementations inside a running Python process.

Use the prepared Q1 environment for both sides and retain **FP32**. The
[Q1 result](q1-20260911.md) passes every required FP32 comparison; BF16 remains
unqualified and Q1's causal-diagnosis criterion remains open.

| Worker order | Excluded work | Measured work |
| --- | --- | --- |
| N-A | Seven feasibility cells, five oracle and eleven native numerical cases | None |
| N-B | Seven feasibility cells, eleven native numerical cases | None |
| A1, B1, B2, A2 | Seven warmup cells in each fresh process | Seven cells per process |
| P-A, P-B | Two warmups and W1/W4 profiles per process | None |

There are exactly **105 executions**: 27 numerical, 14 feasibility, 32 warmup,
28 measured and four profiles, with eight sequential model loads. Every worker
loads one model once and retains it across its assigned rows. A fresh engine is
created for each row; allocator/compiler caches persist within the worker.
Workers do not overlap. Process-local CUDA cleanup must complete before the
next worker starts. No shared caches are dropped.

The numerical subset uses unchanged Q1 fixtures `Q1-L16-F0`, `Q1-L256-F0`,
`Q1-L64-F2` and `Q1-L128-F3`, plus actual live gates for `Q1-L16-F0`. It compares
four Torch serial and four Triton serial cases, packed refill/no-refill, and
one live case on each side. B compares against both the independent oracle and
retained A tensors in the same execution. All 51 comparison trajectories
require the original FP32 final-logit bounds and actual prediction/exit rules;
the 17 B/A comparisons additionally require exact selected tensor/KV values.
Teacher forcing supplies eight inputs but records nine genuine predictions.

Numerical and feasibility prerequisites gate timing. The seven timed cells are
W1/refill, W2/refill, W3/refill, W4/refill, W4/no-refill, W5/refill and
W5/no-refill. Each has matched pairs A1/B1 and B2/A2. Historical M1 timings do
not substitute for these fresh A observations.

| Acceptance gate | Predeclared rule |
| --- | --- |
| Primary W1 throughput | B/A at least 1.10 in both pairs |
| Six control cells | B/A at least 0.95 in every pair |
| W1 observed variation | Minimum B strictly exceeds maximum A |
| Peak allocated and reserved GPU memory | Each B-minus-A increase at most 67,108,864 bytes, in every pair |
| Engine setup time | B-minus-A increase at most 100,000,000 ns, in every pair |
| Numerical behavior/work/cleanup | All required reference and exact A/B gates; matching work; zero retained task requests/pages |

The report retains both observations and ranges, TTFT/TPOT, synchronized time,
memory and setup measurements. Two observations do not establish statistical
confidence. A failing pair cannot be averaged away. A performance miss or
overlapping primary ranges ends promotion at the declared run limit.

Profiles are excluded from timing and follow its four periods. They observe
the actual public/prepared methods once. Candidate `kv_prepare` includes all
five metadata tensors, including positions; baseline positions sit in the
enclosing recurrent scope. Compare preparation, write, attention and position
copies together when checking transfer reduction. Scope times overlap and
include diagnostic overhead, so they cannot be added as production savings.

The total deadline is two hours, with a 600-second case limit. Tensor matching
uses at most 64 MiB, retained groups at most 2 GiB each, cumulative tensor
writes at most 8 GiB, and paired diagnostic dumps at most 256 MiB. Profile
traces are capped at 2 GiB each and 4 GiB combined; all artifacts must fit
12 GiB. Invalid source/controls, execution, numerical, nonfinite, ownership,
cleanup or budget results stop subsequent work. No replacement trials run
automatically. The offline report preserves missing/invalid evidence.

## Commands

First verify scheduler status and choose an available exact physical ID. Run
the CPU probe under the same explicit CPU/NUMA policy as the reservation's
controller. The CPU/node values below were verified on the recorded host;
resolve suitable values before running on another host.

```bash
numactl --physcpubind=56-63 --membind=1 -- env OMP_NUM_THREADS=1 \
  python -m vllm_lt.benchmarks.ab probe \
  --baseline-root /path/to/clean/A --candidate-root /path/to/clean/B \
  --contract benchmarks/fixtures/ouro-m2-contract.json \
  --model-path /path/to/prepared/ouro-1.4b \
  --gpu-id <available-physical-id> --output artifacts/m2-plan

gpu run --gpu-ids <same-physical-id> --timeout 2h \
  --note 'vllm-lt M2 frozen metadata A/B' -- \
  numactl --physcpubind=56-63 --membind=1 -- env OMP_NUM_THREADS=1 \
  python -m vllm_lt.benchmarks.ab run \
  --plan artifacts/m2-plan/plan.json --output artifacts/m2-run

python -m vllm_lt.benchmarks.ab report --run-dir artifacts/m2-run
```

Use new output directories and commit the resolved source/control contract
before device execution. The report's performance result is separate from
manual profile attribution and the criterion-by-criterion adoption decision.
Numerical or structural completion alone cannot close M2.
