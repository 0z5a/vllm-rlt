# Simultaneous group prefix audit

This audit measures the existing engine at
`7f367ee02abb5b8c700a705062fbe98842338d21` before adding W08 group admission or
fork logic. **Runtime source is unchanged.** Ouro and Nanbeige already reuse
complete full-depth prompt pages, but a cold group admitted entirely at once
cannot hit pages that have not been produced yet.

## Observed admission behavior

Each burst contains eight identical prompts with distinct request-local sampling
seeds, four output tokens and fixed full depth. Prompt length P is 15, 16 or 17;
the physical page size is eight tokens. Actual resident concurrency C is 1, 4
or 8 and reaches the requested limit in every measured arm. Both model families
have the same observed prefill counts:

| P | C | Cache disabled: prefill tokens | Cold cache: prefill tokens | Warm cache: prefill tokens |
| --- | ---: | ---: | ---: | ---: |
| 15 | 1 | 120 | 64 | 56 |
| 15 | 4 | 120 | 88 | 56 |
| 15 | 8 | 120 | 120 | 56 |
| 16 | 1 | 128 | 72 | 64 |
| 16 | 4 | 128 | 96 | 64 |
| 16 | 8 | 128 | 128 | 64 |
| 17 | 1 | 136 | 24 | 8 |
| 17 | 4 | 136 | 72 | 8 |
| 17 | 8 | 136 | 136 | 8 |

The cache keeps KV without the final hidden state needed for first-token
readout, so it leaves at least the final prompt token for recomputation. Reuse
is limited to `8 * floor((P - 1) / 8)` tokens per hit. A page-aligned P16 prompt
therefore reuses only eight tokens; P17 can reuse sixteen. Partial-page copying
and first-token hidden/logit reuse are not implemented by this path.

For cold P17/C8, all eight first-prefill positions are zero and no physical page
is shared between active requests. Cold C4 first-prefill positions are
`[0, 0, 0, 0, 16, 16, 16, 16]`: the second wave reuses the first wave's pages.
Cold C1 positions are `[0, 16, 16, 16, 16, 16, 16, 16]`; there is reuse across
time even though only one request is active. Warm C8 positions are all sixteen.
The warm Ouro case has eight physical prefix pages shared by live requests
(two pages at each of four depths); Nanbeige has four at its two depths.

## Numerical and lifecycle checks

Three independent sequential Python processes (seeds 31, 32 and 33) each execute
108 observations: two model families, three prompt lengths, three concurrency
limits and the symmetric order disabled/cold/warm/warm/cold/disabled. Every arm
constructs a fresh engine with the same per-seed model weights. Warm arms first
complete one explicitly timed primer; cold arms begin without cached pages.

All **324 observations, 2,592 measured requests and 10,368 measured output
tokens** pass. These counts exclude separate primer and reference generations.
Tokens, exit depths, stop reasons and policy versions match an independent cold,
cache-disabled execution. Maximum selected-score absolute error is
`9.5367431640625e-7`, within the predeclared `atol=1e-5, rtol=3e-5` bound.
Scores are therefore numerically equivalent, not all bitwise equal. Each arm
verifies actual prefill plus reused-token accounting and complete request drain;
after cache reset all 256 pages have zero references.

The separate regression suites pass **51 distinct CPU checks**: 46 existing
engine/prefix/cache checks, two new physical-publication checks and three
existing Huginn capability guards. Four GPU tests are deselected. Publication
checks warm the cache, start an actual weight transaction, verify immediate
prefix invalidation and blocked generation during a partial update, then finish
the transaction. The first new-policy request is cold, the second reuses pages,
and both match a separately constructed engine with the replacement weights.

Huginn's complete recurrent prefix cache remains explicitly unsupported because
its actual latent identity differs between completions. The guard is tested;
there is no Huginn throughput row. Deterministic-prelude-only reuse remains
future work, as does group-aware leader/follower admission for the cold C8 gap.

## Timing comparison

The following is a diagnostic ablation of the **existing** implementation, not
a new group-fork speedup. All measurements use tiny random FP32 CPU models,
width 16, vocabulary 17, two physical layers, Ouro R4 or Nanbeige R2, and one
OpenMP thread on a shared host. Timing includes admission, generation, final
score export and common measurement instrumentation, but excludes engine/model
construction and network transport. It is neither a GPU nor a full-RL result.

Each ratio averages the two symmetric observations within a process, then uses
the geometric mean across the three process ratios. Approximate 95% intervals
use Student-t on those three log ratios, not individual decode steps. The two
warm columns distinguish a pre-existing cache from the complete extra-primer
cost. Warm/cache-disabled initial states differ; the primer column also does
extra generation work. Neither is a matched fresh-group method comparison.
All cases, wide intervals and regressions are retained.

| Family | P | C | Cold prefill tokens | Warm prefill tokens | Disabled/cold ratio [95% CI] | Disabled/warm ratio [95% CI] | Disabled/(primer+warm) [95% CI] |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| ouro | 15 | 1 | 64 | 56 | 1.359 [0.642, 2.876] | 1.644 [0.927, 2.917] | 1.369 [0.759, 2.467] |
| ouro | 15 | 4 | 88 | 56 | 1.106 [0.598, 2.045] | 1.449 [0.887, 2.368] | 1.080 [0.717, 1.628] |
| ouro | 15 | 8 | 120 | 56 | 0.990 [0.757, 1.294] | 1.579 [0.975, 2.557] | 1.135 [0.717, 1.796] |
| ouro | 16 | 1 | 72 | 64 | 1.468 [1.185, 1.819] | 1.584 [1.012, 2.479] | 1.302 [0.901, 1.881] |
| ouro | 16 | 4 | 96 | 64 | 1.208 [0.857, 1.703] | 1.349 [0.490, 3.718] | 1.019 [0.397, 2.614] |
| ouro | 16 | 8 | 128 | 64 | 1.081 [0.450, 2.597] | 1.427 [0.401, 5.071] | 1.048 [0.278, 3.948] |
| ouro | 17 | 1 | 24 | 8 | 1.992 [1.552, 2.555] | 4.824 [0.505, 46.105] | 2.046 [1.238, 3.384] |
| ouro | 17 | 4 | 72 | 8 | 1.601 [1.228, 2.087] | 3.099 [2.619, 3.667] | 1.938 [1.734, 2.165] |
| ouro | 17 | 8 | 136 | 8 | 1.247 [0.681, 2.282] | 4.113 [2.689, 6.292] | 2.247 [1.552, 3.253] |
| nanbeige | 15 | 1 | 64 | 56 | 1.066 [0.423, 2.690] | 1.224 [0.639, 2.346] | 1.049 [0.554, 1.989] |
| nanbeige | 15 | 4 | 88 | 56 | 1.220 [0.778, 1.913] | 1.338 [1.157, 1.547] | 0.945 [0.682, 1.308] |
| nanbeige | 15 | 8 | 120 | 56 | 0.900 [0.533, 1.522] | 1.267 [0.391, 4.107] | 0.915 [0.284, 2.951] |
| nanbeige | 16 | 1 | 72 | 64 | 1.359 [1.318, 1.401] | 1.484 [0.601, 3.661] | 1.267 [0.520, 3.089] |
| nanbeige | 16 | 4 | 96 | 64 | 1.326 [0.486, 3.612] | 1.702 [0.759, 3.816] | 1.265 [0.596, 2.684] |
| nanbeige | 16 | 8 | 128 | 64 | 1.069 [0.956, 1.196] | 1.916 [1.300, 2.823] | 1.390 [1.054, 1.833] |
| nanbeige | 17 | 1 | 24 | 8 | 2.098 [1.669, 2.639] | 2.199 [1.669, 2.896] | 1.759 [1.411, 2.194] |
| nanbeige | 17 | 4 | 72 | 8 | 1.670 [0.927, 3.010] | 3.639 [1.501, 8.819] | 2.276 [0.933, 5.551] |
| nanbeige | 17 | 8 | 136 | 8 | 1.123 [0.722, 1.746] | 3.464 [1.916, 6.264] | 1.901 [0.961, 3.760] |

In particular, cold C8 has zero reuse in every case and all its timing intervals
include one. P17 warm C8 yields 4.113x for Ouro and 3.464x for Nanbeige when the
cache already exists; including the additional primer yields 2.247x and 1.901x,
respectively, with the latter interval including one. These numbers show why
initial cache state and primer cost must be reported. They are not evidence of
a newly implemented optimization or a full-system speedup. Shorter prompts
retain negative or inconclusive primer-inclusive results.

The pool allocation is identical across arms; active-page telemetry does not
measure peak GPU memory or training-memory savings. Official checkpoints,
asynchronous CUDA/Graph execution, longer high-load stability and reward
convergence are outside this CPU audit.

## Reproduction

```sh
for seed in 31 32 33; do
  PYTHONPATH=. OMP_NUM_THREADS=1 python benchmarks/audit_group_prefix.py \
    --seed "$seed" --output "results/observations-s$seed.jsonl"
done
python benchmarks/summarize_group_prefix.py results --output results/table.md
pytest -q -m 'not gpu' tests/test_engine.py tests/test_prefix_growth.py \
  tests/test_kv_cache.py tests/test_prefix_publication.py \
  tests/test_huginn.py::test_unsupported_execution_rejected_before_cache_allocation
```

Use an existing compatible environment. The retained run used Python 3.12.14,
Torch 2.13.0 and pytest 9.1.1, without package changes. Raw per-request outputs,
scores, latencies, hit counts, page counts, test receipts and source hashes are
under [`benchmarks/results/group-prefix-audit-cpu/`](../benchmarks/results/group-prefix-audit-cpu/).
