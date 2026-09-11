# Q1: incremental Ouro numerical validation

This suite addresses [Q1 issue #4](https://github.com/hsliuustc0106/vllm-lt/issues/4).
The [first complete run](q1-20260911.md) passes FP32 and fails BF16; causal
diagnosis remains incomplete and Q1 stays open.
It compares real checkpoint computations and gate decisions. It makes no latency,
throughput, task-quality, or BF16 deployment claim. The original BF16 failure and
its tolerances remain in [the preview validation record](validation.md).

## Frozen comparison contract

`benchmarks/fixtures/ouro-q1.json` contains 16 synthetic text fixtures: four each
at 16, 64, 128, and 256 prompt tokens. Token IDs, tokenizer provenance, eight
continuation inputs, groups, and loop depths are committed. Eight continuation
inputs produce **nine actual predictions**. Native validation records each
prediction before substituting the next input; supplied tokens are not counted
as matching predictions.

Eight fixtures use four loops throughout. Eight force adjacent shallow-to-deeper
transitions, with different depth patterns in the same packed batch. Four fixed
fixtures additionally run with the actual gate at threshold 0.7/minimum depth two.
The first prediction always follows a full four-loop prefill.

The resolved plan contains exactly 168 implementation executions, each performed
once. This numerical experiment explicitly replaces the default two measured
performance repetitions:

| Family | Executions | Comparison |
| --- | ---: | --- |
| Disjoint feasibility | 10 | One fixture per dtype; oracle and four native configurations |
| Main | 112 | Independent oracle, Torch serial, Triton serial, Triton refill/no-refill |
| Live gate | 28 | Four fixtures per dtype, independently generated histories |
| Official fixed depth | 8 | Reviewed official eager model versus incremental oracle |
| Original reproduction | 10 | Original three prompts, all four prefill depths, original dense and packed implementations |

There are 210 comparison trajectories. The main native comparisons cover 1,152
genuine prediction points. The case table fixes backend, schedule, dtype, history,
capacity, policy hash, and reference for every execution. Main native cache
capacity is 160 physical pages of 16 tokens; prefill chunks have at most 64 tokens.

| Question | Changed variable | Required gate |
| --- | --- | --- |
| Same-dtype fidelity | Independent dense equations versus native paged execution; declared serial/chunked/packed configuration | Finite final logits, unchanged allclose bounds, exact actual top-1, correct histories/work/cleanup |
| Live behavior | Same input prompt with actual gates and independently sampled greedy tokens | Exact token and exit on a matched prefix; retain first divergence |
| Original failure | Original dense versus Torch/Triton packed prefill | Unchanged allclose at all four depths; exact top-1 required at depth four |
| BF16 sensitivity | Oracle dtype only, with identical supplied inputs and forced depths | Diagnostic only |
| Official equations | Pinned official full-sequence eager forward versus independent incremental fixed-depth execution | Same-dtype final-logit and top-1 gates |

FP32 bounds are `atol=0.001, rtol=0.0001`; BF16 bounds are
`atol=0.25, rtol=0.02`. Hidden, attention, populated KV, and gate errors retain
scales and numerical differences but have no invented acceptance bounds.
Near-tie margins are evidence, not exceptions to exact top-1 agreement.

Once a live decision changes the trajectory, later histories are explicitly
incomparable. A selected-depth logit without a corresponding reference depth is
also incomparable. This does not erase the first behavioral failure. Fixed
histories provide the separate controlled numerical comparison.

## Reference independence review

The incremental `SerialOuroOracle` owns dense
`[depth, physical_layer, position, kv_head, head_dim]` K/V and a CPU initialized
mask. It shares only immutable configuration and parameter tensors. Source review
checks the following boundaries:

- Its imports and calls contain no native `recurrent`, norm, RoPE, attention,
  cache manager, scheduler, or engine helpers.
- Each physical layer writes its own K/V before reading the initialized causal
  prefix. Token position and RoPE phase stay fixed across loops.
- Shallow exit copies that token's last computed **per-layer K/V** to skipped
  depths, preserving neighbors and shallower history.
- Native-dtype eager attention follows the pinned official equations. RMSNorm,
  RoPE phase, sigmoid, and cumulative hazard arithmetic are explicit. The gate
  uses the official fused linear bias; the original dense reference is retained
  unchanged for the original failure question.
- CPU tests check different physical-layer states, adjacent exits two then four,
  uninitialized history, cumulative hazards before minimum depth, and threshold
  one. Independent native integration tests cover chunked packed execution,
  fragmented pages, cancellation/reuse, request isolation, and failures.

The validation-only native adapter observes per-instance modules and cache
boundaries, restoring hooks and wrappers on every exit. It reads physical page
tables independently of the oracle. Forced-depth/input control is confined to a
validation subclass; live mode preserves production routing and sampling.

The official adapter uses byte-identical `modeling_ouro.py` and
`configuration_ouro.py` from
`ByteDance/Ouro-1.4B@574fa66cb8bf5abdc979642d01cf2b79b16bfab1`.
Their SHA-256 values and license attribution are retained in
`vllm_lt/validation/reference_code/README.md`. It requires Transformers 4.55.0
with optional `kernels` absent, shares already loaded weights, and performs
fixed-depth eager inference without cache or downloads. It does not provide an
adaptive LAST-EXITED oracle.

## Evidence and resource limits

The plan hashes the exact source tree, checkpoint/tokenizer bytes, official
source, inputs, comparison policy, and dependency manifest. It records one
physical GPU ID before execution. Both dtype passes use that same reservation,
one CPU thread, fixed seeds, disabled TF32/reduced-precision reductions, and
inherited/recorded CPU and NUMA affinity. Checkpoint loading is recorded separately.
Both feasibility sets precede qualification, so dtype transitions may reload the
immutable checkpoint; each load is recorded. No shared caches are dropped.

Observers match tensors by request, history, position, depth, physical layer, and
operation rather than callback arrival order. They cover the final prompt token
and all eight continuation positions at every executed layer/loop, plus all
populated final KV positions in four-position chunks. Keys and values remain
separate components. Final selected logits are computed once per prediction;
extra per-loop LM heads exist only in the original reproduction.

Typed reference spools preserve BF16 bytes. Matching uses at most 64 MiB of
transient tensor buffers. Limits are 2 GiB per reference group, 8 GiB cumulative
tensor writes, and 12 GiB total artifacts, including capped JSON records. The
real-model reference tensor upper bound is 7,236,407,304 bytes; the FP32 native
pool is 960 MiB and the largest independent oracle cache is 396 MiB. Runtime
feasibility checks activation peaks rather than inventing an activation estimate.

Persisted paired tensor dumps cover two preselected fixtures and at most the
first two finite failing fixtures, selected in deterministic execution/observed
boundary order. The preselected fixtures reserve their quota for BF16 native
same-dtype comparisons, preserving raw layers for the BF16 diagnosis. Neither
FP32 controls nor oracle dtype-sensitivity comparisons consume that quota.
Additional failure-selected fixtures capture whichever dtype failed. Capture
begins when selected and stops at the byte cap; it never reruns a case to recover
an earlier tensor. Limits are 64 MiB per fixture and 256 MiB total. Reference
spools are distinct from these native diagnostic dumps.

Each comparison JSONL has a hash, complete record count, operation summaries,
behavior traces, and explicit required versus diagnostic gates. The offline
report reconstructs decisions from those records and reconciles all cases;
selected raw dumps support inspection. Because all native tensors are not
retained, it does not claim to recompute every elementwise error offline.

## Commands

Prepare a compatible environment and local checkpoint before the experiment.
Reuse an existing suitable environment where available. Verify scheduler status
and choose an available exact physical ID, then freeze the plan without device
discovery:

The `validation` package extra pins the official adapter's Transformers/Hub/
tokenizer dependencies: `python -m pip install -e '.[validation,dev]'`. Its
environment must have optional `kernels` absent; probe rejects incompatible
environments before checkpoint loading or CUDA discovery.

```bash
python -m vllm_lt.validation probe \
  --suite benchmarks/fixtures/ouro-q1.json \
  --contract benchmarks/fixtures/ouro-q1-contract.json \
  --model-path /path/to/pinned/ouro-1.4b \
  --gpu-id <available-physical-id> --output artifacts/q1-plan
```

Run the unchanged source/plan in a new artifact directory. The driver verifies
the scheduler-assigned device against the frozen ID before touching CUDA:

```bash
gpu run --gpu-ids <same-physical-id> --wait 10m --timeout 2h \
  --note 'vllm-lt Q1 frozen numerical validation' -- \
  env OMP_NUM_THREADS=1 python -m vllm_lt.validation run \
  --plan artifacts/q1-plan/plan.json --output artifacts/q1-run

python -m vllm_lt.validation report --run-dir artifacts/q1-run
```

The two-hour outer limit wins over ten minutes per execution. Numerical
counterexamples remain recorded while the remaining planned cases run.
Invalid history, nonfinite values, device failure, cleanup failure, or exhausted
budgets stop execution and preserve incomplete evidence. A source fix or new
hypothesis requires a separately recorded plan; the driver never adds trials.

Execution completion is separate from numerical qualification. Q1 closes only
after all six issue acceptance criteria and a reviewed qualified or conclusive
negative diagnosis. Incomplete coverage or unlocalized divergence remains
inconclusive; BF16 remains unqualified until its declared gates pass.

## Review follow-up: independence and diagnostic scope

The pinned official comparison remains mandatory in numerical follow-ups: the
serial oracle and dense in-repository reference could share an interpretation
error. Official BF16 disagreement remains a separate unresolved result. The
FP32 `0.001/0.0001` limits were deliberately tightened before Q1 execution; the
original BF16 `0.25/0.02` limits and discrete failures remain unchanged.

A successor BF16 diagnostic must separate identical-input, same-state local
arithmetic replay from equal-token-history trajectories with independently
accumulated hidden/KV states. Snapshot position, depth, hidden input, populated
KV and probability state before comparing serial BF16, batched BF16 and FP32
using the same checkpoint. Check max/RMS errors per boundary, gate depth and
historical/new-computed/new-fallback KV bank, plus token margins and threshold
distance. Preserve fallback and prior-prefix invariants. Do not import another
project's error multiplier or use near ties to exempt a discrete mismatch.
This is a diagnostic requirement, not an executed experiment or a revised
qualification policy; AC-Q1-05 and BF16 deployment remain open.
