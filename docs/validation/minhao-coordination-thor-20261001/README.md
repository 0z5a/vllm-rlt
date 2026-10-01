# Phase 1 / VIME coordination checks — 2026-10-01

These checks exercise the semantics in
[MinhaoLi0318's PR 1 proposal](https://github.com/ThinkFlowLab/vllm-rlt/issues/70#issuecomment-5927860064)
on the existing Thor runtime. They test the contributor implementation, not an
unpublished Phase 1 branch. The
[coordination reply](https://github.com/ThinkFlowLab/vllm-rlt/issues/70#issuecomment-5930413915)
records the interface differences and division of upstream work.

The replay requirement exposed a real omission: an entropy-seeded request
returned `sampling_params.seed=None`. The
[unfixed regression](raw/minhao-seed-before.log) fails on that missing effective
seed. `Request` now resolves an independent 63-bit seed when it is created and
retains it through output, preemption, abort and existing PD parameter transport.
Default `seed=0` is unchanged. An abort before any sampling also reports the seed;
the caller's immutable `SamplingParams(seed=None)` is not modified.

| Thor CPU check | Result |
|---|---|
| Entropy seed reporting and exact token/score/depth replay | Ordinary, async and speculative paths pass |
| Abort before sampling | Effective seed retained; zero selected scores |
| Preemption | Token IDs, selected scores, depths and RNG continuity match uninterrupted execution |
| Existing probability/publication/cache/stop/abort checks | All pass |
| Full focused suite including in-process PD publication | **7 tests pass, 1.736 s** |

See [CPU log](raw/minhao-cpu.log). Independent VIME FP32 mixed-depth replay,
recomputed gradients and generic physical-parameter conversion are checked
separately. No runtime packages are installed or updated.

The user authorized concurrent short CUDA validation while an existing
Qwen-Image training reservation remained active. **All six CUDA contract tests
pass in 135.474 s**, including entropy-seed replay with graphs, stop/abort,
preemption, speculative target scores and publication recovery; see
[CUDA log](raw/minhao-cuda-parallel.log). Startup is included in the suite time. The originally queued CUDA suite also
completed naturally after the reservation was released: **6/6 in 3.989 s**,
[log](raw/minhao-cuda.log).
Independent VIME FP32 mixed-depth replay/recomputed gradients and generic
physical-parameter conversion pass **two CPU tests in 0.302 s**. No existing
training process or reservation was interrupted.

Earlier full-model training/save/resume and real-PD evidence is retained in the
[Thor](../training-contract-thor-20261001/README.md) and
[A100](../training-contract-a100-20261001/README.md) reports.

## Selected-score cost

`benchmarks/logprob_overhead.py` warms each mode and rotates the measurement
order over five repeats. Each unprofiled window generates two prompts with 32
response tokens each, T=0.9, fixed per-request seeds, no filtering and CUDA
graphs enabled. Sync and speculative paths use Ouro exit; async uses delayed
Ouro exit. Token sequences must match across score modes; selected scores must
align with every committed token. Initialization and capture are excluded.

Separately, all three modes receive an ops-only and a full-detail CPU/CUDA
profiler window (`record_shapes`, stacks and memory enabled for full detail).
The raw JSON retains the 20 operators with largest self CPU time per window,
with counts, self device time and memory. These are operator summaries, not
exported timeline traces. Profiled runs are excluded from throughput medians.
The [tiny smoke JSON](raw/minhao-tiny-overhead.json) checks all three paths and
18 profiler windows with one repeat; its timings are not full-model results.

### Full Ouro-1.4B shared-GPU observations

The full run overlaps another Qwen-Image training phase. GPU utilization
before this run was **98%**; [snapshot](raw/minhao-gpu-before.csv).
These measurements include resource contention and are not isolated throughput
results. Each mode has five unprofiled repeats. All three paths complete;
token IDs match across modes and selected scores align with committed tokens.

| Path | None median (s) | Raw median (s) | Processed median (s) | Raw time overhead | Processed time overhead |
|---|---:|---:|---:|---:|---:|
| sync | 1.8017 | 1.8280 | 1.8281 | +1.46% | +1.47% |
| async | 1.6974 | 1.7849 | 1.7894 | +5.15% | +5.42% |
| speculative | 1.4366 | 1.4462 | 1.4486 | +0.67% | +0.83% |

| Path / score mode | Tokens/s | Speed vs None | Median extra PyTorch allocation (KiB) |
|---|---:|---:|---:|
| sync / none | 35.52 | 1.000× | 933.0 |
| sync / raw | 35.01 | 0.986× | 965.0 |
| sync / processed | 35.01 | 0.986× | 933.0 |
| async / none | 37.70 | 1.000× | 965.0 |
| async / raw | 35.86 | 0.951× | 963.5 |
| async / processed | 35.77 | 0.949× | 965.0 |
| speculative / none | 44.55 | 1.000× | 2733.5 |
| speculative / raw | 44.25 | 0.993× | 2793.5 |
| speculative / processed | 44.18 | 0.992× | 2761.5 |

[Raw timings / all 18 profiler operator summaries](raw/minhao-logprob-overhead.json),
[benchmark log](raw/minhao-logprob-overhead.log). Extra allocation is
per-window PyTorch peak above its starting allocation; it excludes resident
model/cache/graph storage and other processes.

Between completed paths, the process temporarily waited in NVIDIA UVM external
allocation mapping (`os_acquire_rwlock_read/write`); an independent GPU status
query also waited. The [captured stack](raw/minhao-driver-stack-wait.log) and
[kernel log](raw/minhao-driver-wait.log) record concurrent `NV_ERR_NO_MEMORY`
allocation errors. Both waits resolved naturally; all paths completed without
process termination, driver reset or environment changes. Initialization waits
are outside the warmed measurement windows. The harness now drops the previous
engine and clears its allocator cache between paths to avoid retaining that
engine while constructing the next one. The recorded full run predates this
inter-path cleanup; its timed body is unchanged.
The updated cleanup harness then passes a separate tiny-model smoke across
all three paths and 18 profiler windows ([JSON](raw/minhao-cleanup-smoke.json),
[log](raw/minhao-cleanup-smoke.log)); its [57,039,421-byte model is removed](raw/minhao-cleanup-smoke-removed.json).

All five full-model depth/effective-seed replay cases also complete, including
delayed async. The [VIME score-error table](https://github.com/0z5a/vime/blob/feat/ouro-rollout-cuda-graphs/docs/validation/minhao-coordination-thor-20261001/README.md)
records the remaining training/inference recomputation differences; full-depth
K=4 also has a nonzero residual. Exact seed replay and score-recomputation
agreement are separate checks.

The [completed full-model copy and replacement shard](raw/minhao-cleanup.json)
are removed (3,232,344,846 bytes). The 358,667,054-byte local transfer copy
was also removed. The original slower download completed naturally; its
[2,873,677,792-byte model copy was removed](raw/minhao-slow-cleanup.json) by the
existing queue, without repeating GPU tests. All four task-owned model folders
and both replacement transfer copies are absent after validation.

## Interface coordination

| Proposed Phase 1 interface | Current reference draft | VIME consumer plan |
|---|---|---|
| `RequestOutput.logprobs` | `RequestOutput.log_probs` | Use the upstream spelling once its PR 1 branch is available |
| Engine-level `raw_logprobs` / `processed_logprobs` | Request-level `raw` / `processed` | Configure processed scores for the trainer; avoid a second compatibility API |
| Effective seed reported in resolved parameters | Fixed by this follow-up | Keep seeded reproduction and opt-in entropy groups |
| Full-depth first token and executed decode depths | Already present | Full four-loop prefill and differentiable `last_exited` trace replay |
| PR 2 versioned full publication | Validated reference implementation | Reuse one agreed upstream contract |

The old `RLEngine.set_loop_budget` mutated `total_ut_steps`; the current VIME
adapter supersedes it and does not need reduced-depth prefill. No new transport
layer is needed for the pilot: the consumer is in-process `LLM`, and existing PD
worker IPC remains inside the engine. MinhaoLi0318 and IDEA-V retain the proposed
Phase 1 upstream roles; this draft supplies reusable implementation and checks.
