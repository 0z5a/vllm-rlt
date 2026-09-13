---
name: vllm-lt-review-pr
description: "Review PRs and local changes for hsliuustc0106/vllm-lt: Ouro engine and KV correctness, serving behavior, and BF16 accuracy/speed evidence. Use for maintainer reviews, repeat reviews, and author self-reviews of this project; not for vLLM-Omni or unrelated repositories."
---

# Review vllm-lt changes

Review the requested change against its intended base. Return actionable,
evidence-backed defects; zero findings is valid. Distinguish implementation
correctness, reference fidelity, accuracy, speed and memory qualification.

## Freeze the review

- Verify the repository/remotes and applicable AGENTS.md. The project is
  `hsliuustc0106/vllm-lt`; accept its verified forks. Do not inherit vLLM-Omni's
  architecture, reviewers, test requirements or milestone gates.
- For a PR, record base/head SHAs, merge base, state, description, full diff,
  relevant review threads and checks. Read code from the frozen head, not a
  moving local branch. A stacked PR's prerequisite stack is its immediate
  comparison base; distinguish a cumulative comparison against main.
- For local changes, record HEAD, intended base, status and the relevant staged,
  unstaged and untracked diff. Preserve user changes. A review does not authorize
  source fixes, commits, benchmark campaigns or posting a GitHub review.
- Before delivery, check whether the PR head changed. Label the reviewed SHA;
  recheck affected findings if continuing against the newer head. Do not present
  old CI or experiment evidence as validation of an untested newer revision.
- When a prerequisite is reported merged, fetch the verified upstream main
  before selecting a new experiment base. If PR metadata disagrees with Git,
  verify the merge's parents/ancestry or integrated diff rather than relying on
  a cached API state or commit subject alone. Report the discrepancy and the
  verified Git SHA; do not keep treating an integrated prerequisite as open.
  Preserve the requested snapshot for an ongoing review.

## Explain core changes and Plan B

For changes to core behavior in scheduling, engine/request ownership, KV,
model execution or kernels, explicitly assess the design choice before the
implementation details. Keep the explanation proportional to the change:

- **Why change:** name the concrete failure, measured bottleneck or required
  capability. Cite the reproducer, profile or requirement; distinguish evidence
  from a hypothesis. Explain why the existing design cannot adequately handle it
  and why this core module, rather than a smaller change at its caller, must change.
- **Chosen approach:** connect the proposed mechanism to that problem and state
  the invariants, complexity and compatibility costs it introduces.
- **Plan B:** compare at least one credible alternative, such as a narrower fix,
  reuse of an existing mechanism, or retaining the current implementation while
  gathering evidence. Explain its correctness, performance and maintenance
  tradeoffs, why it was not selected, and what evidence would favor it instead.
  If no viable alternative is apparent, explain the constraints; do not invent one.
  Discuss an existing fallback or rollback where relevant, without requiring a
  second implementation merely to demonstrate an alternative.

Use the PR's rationale when supported, and label reviewer-proposed alternatives
as such. Missing rationale or unresolved tradeoffs are design questions, not
automatically proven bugs. Surface them prominently when they affect whether
the core change is justified.

## Follow the changed behavior

Read the changed code with its callers, ownership and failure paths. Use the
reviewed checkout's `docs/design.md`, `docs/serving.md`, `docs/accuracy.md` and
`docs/benchmarks.md` where relevant. Treat archived reports as dated evidence.
If `docs/ab-tests.md` and `benchmarks/ab.py` exist in the reviewed snapshot,
inspect their actual contract and validators; do not assume a local proposed
workflow has merged. Resolve stale documentation against code and explicit
current requirements rather than treating every historical statement as a gate.

| Changed area | Review focus |
| --- | --- |
| `request.py`, `core/scheduler.py`, `engine/llm_engine.py` | Stage/progress ownership, one in-flight decode token per request, full-depth chunked prefill, exactly one first output from final prefill, refill/cohort routing, bounded progress, admission and cancellation. |
| `core/kv_cache_manager.py`, attention kernels | Per-request/depth block tables and causal lengths; populated versus merely reserved KV; partial pages, mixed depths, skipped-depth propagation and reuse. |
| `models/`, `worker/model_runner.py`, sampling | Shared-core recurrence and normalization, position/RoPE unchanged across a token's loops, cumulative gate updates even before minimum exit depth, forced maximum-depth exit, coda/logit selection and request-local RNG progression. |
| `serving/`, entrypoints | Worker ownership, real readiness, queue/admission bounds, disconnects, timeout/error propagation, streaming token/usage accounting and shutdown. |
| `benchmarks/`, `vllm_lt/benchmarks/`, result documentation | Frozen controls, source/protocol provenance, complete coverage, timing/accounting, paired gates and honest handling of failed evidence. |

Specific invariants to trace when touched:

- Last-exited KV copies each layer's final computed token K/V to skipped deeper
  depths, preserving shallower state and neighboring tokens. A whole-page alias
  is not equivalent when adjacent tokens exited at different depths.
- Reservation covers executable positions, excluding the last sampled token
  when it has no forward pass. Full-depth prefill still needs all model depths
  even when decode has a smaller loop limit. Incremental allocation needs an
  explicit progress/ownership argument; release must wait for dependent work.
- Inactive/padded rows must not read uninitialized KV or mutate live KV, RNG,
  outputs or hidden state. Persistent buffers/graphs additionally need stable
  addresses, slot lifetime, synchronization, valid fallback and bounded memory.
- Streaming counts actual generated IDs, including empty-text token events;
  usage and termination markers are not additional tokens. An interrupted
  stream must not be accepted as a successful short response by the pinned
  benchmark client. Slow-client/error cleanup must preserve other requests.
- Readiness means model, tokenizer and engine are initialized. Validate the
  first real request; compilation/warmup and process-to-readiness are separate
  measurements. Refill alone does not imply asynchronous host/device overlap.

## Assess accuracy and performance evidence

The standard accuracy and speed A/B tests both use BF16 weights, ordinary
activations and KV. Check explicit configuration and actual parameter/cache
tensor dtypes, not just a CLI default or declared metadata. Full-model FP32 is
only an explicitly requested diagnostic; do not select it as a fallback baseline
because an older harness requires it. Retain justified FP32 reductions and
intermediates within BF16 inference and record accumulation flags separately.
Historical FP32 measurements or numerical failures remain scoped to their
original contracts; neither silently qualifies nor automatically blocks a new
BF16 experiment.

When reporting current main's accuracy or speed, identify the execution SHA,
dtype, workload and timing boundary for each number. A stored regression score,
a full-dataset result from another branch, and a measurement of the latest main
are distinct evidence. Merging serving support or accuracy tests does not rerun
the benchmark. If the current BF16 baseline has not been measured, say so; do
not substitute old FP32 throughput. Consult the current fixture and frozen
contract for accuracy floors and performance gates instead of hard-coding a
historical score, merge SHA or local proposed threshold into the review.

- Metadata/data-movement-only changes preserving arithmetic require exact
  selected state, populated KV, tokens/exits and RNG behavior. Arithmetic
  changes need independently justified, predeclared numerical/decision bounds;
  do not demand blanket bitwise equality with a different arithmetic path or
  invent near-tie exceptions after seeing failures.
- For accuracy, verify model/dataset/tokenizer/package pins, exact question and
  prompt IDs, four-loop/greedy or declared adaptive policy, demonstrations,
  stopping/token limits and strict scoring. Check raw-response rescoring,
  missing/duplicate questions, paired losses/gains, parse/length failures and
  the actual frozen floor/loss budget. Equal total scores can hide different
  failures. A selected regression cohort is not a held-out quality estimate.
- Fixed-depth accuracy and synthetic token/exit replay do not qualify live
  adaptive quality. Compare adaptive history against the same last-exited
  semantics; official/full-depth recomputation can have different KV history.
- Speed A/B must isolate the PR change, use matching workloads and declared
  arithmetic, exact devices, CPU/NUMA placement, package/client code, capacity
  and cache controls. Check the actual measured order against the frozen plan;
  ABBA means A1/B1/B2/A2, paired by repetition. Disclose any restarts and warmups.
- Exclude feasibility, first-request diagnostics, warmups and profiling from
  measured timings. Require complete successful requests and correct token/work
  accounting; failed, truncated or selectively resumed runs cannot enter a
  passing comparison. Keep original failures and exhausted budgets visible.
- Report both observations and variability under the declared acceptance rule.
  Separate aggregate throughput from per-request latency, HTTP/client timings
  from engine-only timings, and allocation bytes from useful KV occupancy.
  Changed output lengths can distort accuracy-run timing; those durations alone
  do not establish a controlled speedup. Missing memory/capture evidence remains
  unqualified where that claim requires it.
- Apply thresholds frozen for this experiment; do not impose unmerged proposals
  or a speedup requirement on correctness-only PRs. A reproducibly invalid speed
  claim is actionable; missing evidence is otherwise a qualification limitation,
  not proof that the implementation is wrong.

## Validate and deliver

Use targeted CPU tests or a small reproducer to resolve a concrete uncertainty.
Check test dependencies and skips; a green suite with GPU cases skipped does not
validate kernels or real serving. GPU validation requires the user's applicable
experiment scope and verified scheduler reservation; never use an unreserved
device or silently expand a benchmark's exhausted run budget. If unavailable,
finish the code/evidence review and identify the remaining validation precisely.

For each finding, give priority, a short defect title, the smallest relevant
changed-line location, a reachable trigger, observable impact and supporting
evidence. Trace the failure into unchanged callers if needed, but attribute it
to this change. Exclude speculative risks, unrelated backlog and style-only
feedback. In repeat reviews, verify fixes at the new snapshot and avoid reposting
resolved findings.

For core behavior changes, open with a short assessment of why the change is
needed and its Plan B, including any unresolved design question. Then give
findings in priority order, followed by a brief validation/scope note and the
reviewed SHA. For other changes, lead directly with findings.
If there are none, say so without implying unrun accuracy,
speed or GPU gates passed. Use verified PR diff links or absolute local file
links. Draft/post comments only within the user's explicit authorization; select
reviewers from actual ownership evidence rather than inventing owners.
