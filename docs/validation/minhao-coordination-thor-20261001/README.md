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

At the time of the initial coordination reply, new CUDA checks are queued behind
an existing Qwen-Image training reservation; they are not counted as passed here.
Full-model score-mismatch and selected-score-overhead measurements will be added
after that reservation permits execution. Earlier full-model/real-PD evidence
is retained in the [Thor](../training-contract-thor-20261001/README.md) and
[A100](../training-contract-a100-20261001/README.md) reports.

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
