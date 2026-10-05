# Cold group prefill with independent continuations

Eight equal prompts arriving at a cold engine previously performed eight
prefills when admitted together. The opt-in `SchedulerConfig(group_prefill_fork=True)`
now admits one leader, shares its completed full-depth KV pages and copies its
first-response hidden state into each follower. Every follower retains its own
sampling generator, parameters, output state and policy version. The default is
disabled. This implementation supports Ouro and Nanbeige with `last_exited` KV;
Huginn's latent-conditioned recurrent prefix is explicitly rejected.

The comparison parent is the untouched commit
`fd993ec5512904b68e16f4d541682c076c487b5d`, the existing-prefix audit in draft
[#20](https://github.com/0z5a/vllm-rlt/pull/20). The parent and candidate both use
ordinary prefix caching, initially empty. This measures a new cold-group
optimization, following the separate [existing-cache audit](group_prefix_audit.md).

## Ownership and admission

Groups are keyed by policy version and exact prompt tokens. The snapshot owns
page references independently of the leader's lifetime. Full pages remain
shared; a partial tail receives private pages on its first continuation write.
Only valid prefix positions are copied. Overwriting a shared prefix is rejected.
First-response hidden states are copied; RNG state is never shared. Groups with
one request create no snapshot. Aborts, request-ID reuse and complete policy
publication release or invalidate the corresponding state.

Admission reserves tail-copy and remaining growth capacity. Forks follow the
existing bounded-bypass rule. A pending snapshot can be reclaimed when it blocks
normal admission, and a longer follower can fall back to recomputation when no
active request can release enough space. Tests reproduce and repair these
capacity/fairness cases. A separate untouched-parent reproduction also fails
when the last preempted request resumes directly into PRELUDE; the scheduler now
selects that stage after admission and the same sequence drains successfully.

For CUDA, caller-stream events order successive steps' snapshot and copy reads;
async execution streams wait on the caller and per-request completion events.
Hidden tensors record their consuming streams. This adds no device-wide
synchronization. The four new BF16 CUDA/Graph cases alternate caller streams and
reuse request IDs, but **have not run**. Event ordering is implemented, not GPU
qualified. Full-model CUDA/Graph and memory-pressure qualification remain gates.

## Validation and measurement

The final nine-file regression passes **183 unique CPU tests**, with **39 GPU
tests deselected**, including the four new cases. CPU coverage includes sampled
tokens and scores, page boundaries, distinct seeds, aborts, policy publication,
preemption, incremental allocation, scheduler modes, speculation, capacity
fallback and fairness. Earlier runs overlap and are not added to this total.
Initial failures and the independent parent/candidate resume reproduction are
retained under `benchmarks/results/group-prefill-fork/regression`.

Timing uses tiny random FP32 CPU models: width 16, vocabulary 17, two physical
layers, Ouro R4 or Nanbeige R2, block size 8 and 1,024 physical pages. Every
request emits eight tokens at fixed full depth with temperature 0.8 and a
distinct seed. P is prompt length, G group size, and C the concurrency limit;
observed peak active count equals `min(C,G)` in every arm. This includes queued
C1/C4 cases and simultaneous C8, not high-load production serving.

Seeds 31/32/33 each run four **fresh sequential processes** in parent/fork/fork/parent
order. Each process measures all 40 workloads using a fresh cold engine per
workload. A separate cache-disabled, serial oracle runs before each timed case.
Timing includes admission, generation, final score export and common telemetry;
it excludes model/engine construction, oracle work and final cache reset.
One OpenMP thread is used on a shared host. Process startup and network are not
timed; no Ray, learner, reward, optimizer or policy-publication step is timed.

All **480 observations, 3,504 measured requests and 28,032 output tokens** pass.
Tokens, depths, termination and policy versions match both independent oracles
and opposite comparison arms. Maximum absolute selected-score error is
`9.5367431640625e-7`, within the fixed `atol=1e-5, rtol=3e-5` tolerance; this is not
bitwise score equality. Prefill plus ordinary-cache hits plus group reuse equals
G times P. All allocations drain and every page reference clears after reset.
Candidate G8 cases execute exactly P prefill tokens and seven forks, with no
capacity fallback in the timed matrix. Separate pressure tests exercise fallback.

The earlier 480-observation screening run predates the final admission and
stream fixes. It is retained as `screening-v1` with its exact source patch and
module snapshot, and is **excluded from all final estimates and totals**.

## Speed comparison

Within each seed, the ratio divides the mean of the two parent times by the mean
of the two candidate times. Reported ratios are geometric means across the three
seed cohorts. Approximate 95% intervals use Student-t with two degrees of freedom
on the three log ratios. Millisecond columns are arithmetic means of six
observations each, so dividing them need not reproduce the geometric ratio.
All 40 workloads, one-request controls, regressions and wide intervals are shown.

`Live pages` is the observed maximum used physical pages, including ordinary
cache owners and pending group snapshots. Both arms allocate the same 1,024-page
tensor pool. These counts are **not allocated/reserved GPU peaks**: for example,
Ouro P128/C8 uses 544 versus 96 pages, while P7/C1 increases from 8 to 12 because
the pending snapshot retains a tail page at each depth. No peak-memory saving is
claimed. Short queued cases can already reuse most full pages in the parent.

| Family | P | G | C | Parent ms | Fork ms | Speed ratio [95% CI] | Prefill A/B | Live pages A/B |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| nanbeige | 7 | 1 | 1 | 6.65 | 9.17 | 0.784 [0.261, 2.354] | 7/7 | 4/4 |
| nanbeige | 7 | 8 | 1 | 53.19 | 47.87 | 1.111 [0.948, 1.301] | 56/7 | 4/6 |
| nanbeige | 7 | 8 | 4 | 26.32 | 25.75 | 1.026 [0.725, 1.454] | 56/7 | 16/18 |
| nanbeige | 7 | 8 | 8 | 21.22 | 17.86 | 1.188 [1.041, 1.355] | 56/7 | 32/32 |
| nanbeige | 8 | 8 | 1 | 53.61 | 46.08 | 1.162 [0.947, 1.427] | 64/8 | 4/4 |
| nanbeige | 8 | 8 | 4 | 27.42 | 22.64 | 1.210 [0.993, 1.473] | 64/8 | 16/10 |
| nanbeige | 8 | 8 | 8 | 22.87 | 17.55 | 1.300 [0.961, 1.759] | 64/8 | 32/18 |
| nanbeige | 9 | 8 | 1 | 49.08 | 46.40 | 1.055 [0.797, 1.398] | 16/9 | 4/6 |
| nanbeige | 9 | 8 | 4 | 26.02 | 22.82 | 1.138 [0.807, 1.604] | 40/9 | 16/12 |
| nanbeige | 9 | 8 | 8 | 24.97 | 17.34 | 1.438 [1.133, 1.824] | 72/9 | 32/18 |
| nanbeige | 17 | 8 | 1 | 49.60 | 47.24 | 1.050 [0.932, 1.183] | 24/17 | 6/8 |
| nanbeige | 17 | 8 | 4 | 28.82 | 24.12 | 1.196 [1.070, 1.337] | 72/17 | 24/14 |
| nanbeige | 17 | 8 | 8 | 34.74 | 19.02 | 1.828 [1.644, 2.032] | 136/17 | 48/20 |
| nanbeige | 128 | 8 | 1 | 87.60 | 68.83 | 1.272 [1.175, 1.376] | 184/128 | 34/34 |
| nanbeige | 128 | 8 | 4 | 118.98 | 50.98 | 2.333 [2.056, 2.648] | 544/128 | 136/40 |
| nanbeige | 128 | 8 | 8 | 193.37 | 42.89 | 4.510 [4.259, 4.776] | 1024/128 | 272/48 |
| nanbeige | 129 | 1 | 1 | 34.11 | 56.02 | 0.702 [0.142, 3.481] | 129/129 | 34/34 |
| nanbeige | 129 | 8 | 1 | 78.44 | 69.56 | 1.127 [1.007, 1.261] | 136/129 | 34/36 |
| nanbeige | 129 | 8 | 4 | 113.29 | 52.09 | 2.171 [1.910, 2.468] | 520/129 | 136/42 |
| nanbeige | 129 | 8 | 8 | 192.60 | 98.44 | 2.570 [0.259, 25.497] | 1032/129 | 272/48 |
| ouro | 7 | 1 | 1 | 13.72 | 14.80 | 0.931 [0.603, 1.439] | 7/7 | 8/8 |
| ouro | 7 | 8 | 1 | 107.01 | 99.20 | 1.088 [0.842, 1.405] | 56/7 | 8/12 |
| ouro | 7 | 8 | 4 | 53.90 | 48.28 | 1.137 [0.630, 2.052] | 56/7 | 32/36 |
| ouro | 7 | 8 | 8 | 42.82 | 37.74 | 1.158 [0.602, 2.229] | 56/7 | 64/64 |
| ouro | 8 | 8 | 1 | 116.07 | 108.28 | 1.098 [0.505, 2.384] | 64/8 | 8/8 |
| ouro | 8 | 8 | 4 | 55.58 | 49.10 | 1.145 [0.636, 2.064] | 64/8 | 32/20 |
| ouro | 8 | 8 | 8 | 48.98 | 37.12 | 1.318 [0.911, 1.906] | 64/8 | 64/36 |
| ouro | 9 | 8 | 1 | 110.82 | 107.57 | 1.027 [0.637, 1.656] | 16/9 | 8/12 |
| ouro | 9 | 8 | 4 | 55.02 | 48.72 | 1.126 [0.795, 1.595] | 40/9 | 32/24 |
| ouro | 9 | 8 | 8 | 55.28 | 35.38 | 1.554 [1.160, 2.081] | 72/9 | 64/36 |
| ouro | 17 | 8 | 1 | 109.80 | 102.48 | 1.070 [0.734, 1.561] | 24/17 | 12/16 |
| ouro | 17 | 8 | 4 | 63.50 | 50.07 | 1.262 [0.900, 1.771] | 72/17 | 48/28 |
| ouro | 17 | 8 | 8 | 69.61 | 38.42 | 1.811 [1.450, 2.263] | 136/17 | 96/40 |
| ouro | 128 | 8 | 1 | 174.07 | 168.76 | 1.038 [0.796, 1.354] | 184/128 | 68/68 |
| ouro | 128 | 8 | 4 | 227.46 | 105.08 | 2.171 [1.551, 3.040] | 544/128 | 272/80 |
| ouro | 128 | 8 | 8 | 357.94 | 98.69 | 3.631 [2.423, 5.440] | 1024/128 | 544/96 |
| ouro | 129 | 1 | 1 | 69.96 | 78.19 | 0.895 [0.654, 1.225] | 129/129 | 68/68 |
| ouro | 129 | 8 | 1 | 156.47 | 159.56 | 0.987 [0.708, 1.375] | 136/129 | 68/72 |
| ouro | 129 | 8 | 4 | 227.03 | 114.78 | 1.996 [1.286, 3.098] | 520/129 | 272/84 |
| ouro | 129 | 8 | 8 | 366.14 | 101.19 | 3.622 [2.875, 4.564] | 1032/129 | 544/96 |

## Scope and reproduction

The P128/G8/C8 CPU ratios are 3.631 [2.423, 5.440] for Ouro and
4.510 [4.259, 4.776] for Nanbeige. P129 Nanbeige has a much wider interval
[0.259, 25.497], and one-request controls are slower on their central estimates.
These results support workload-dependent CPU reduction of repeated prefill;
they establish neither official-model GPU speed nor full-RL acceleration.

| Required endpoint | Result |
| --- | --- |
| Tiny CPU cold-group generation | Measured in the complete table |
| CPU numerical/lifecycle regression | 183 pass; original failures retained |
| CUDA, Graph and official-model group fork | NOT_RUN; four new GPU cases pending |
| Allocated/reserved GPU peak reduction | NOT_MEASURED |
| High-load network throughput and long-run failure rate | NOT_MEASURED |
| Full online RL speed, held-out reward convergence, time-to-quality | NOT_MEASURED |
| Huginn deterministic-prelude sharing | NOT_IMPLEMENTED |

Use the repository's existing environment; no environment changes are required
by this patch. Run `pytest tests/test_group_prefill.py -m 'not gpu'` for the
focused CPU suite. `benchmarks/benchmark_group_fork.py` is portable across the
parent and candidate through `PYTHONPATH`: run each seed with labels
`parent-a1`, `fork-b1`, `fork-b2`, `parent-a2`, adding `--fork` only to the two
candidate invocations. Keep the parent source immutable. The retained
`paired-controller.py.txt` records the exact invocation sequence used here.
Regenerate `summary.json` and `table.md` with
`python benchmarks/summarize_group_fork.py benchmarks/results/group-prefill-fork/clean-cpu --parent-manifest benchmarks/results/group-prefill-fork/parent-source.json`.

The validation manifest binds runtime, benchmark, test and raw-evidence hashes.
The parent archive contains 54 runtime files reconstructed from the exact Git
revision; the candidate has 55. No parent production file is patched for timing.
The implementation stays opt-in pending target-GPU and full online-RL evidence.
