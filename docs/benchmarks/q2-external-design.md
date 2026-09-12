# Q2 cached external FP32 comparison

This is the historical v1 design. The current default is the
[BF16 single-request comparison](q2-external-bf16.md); the frozen FP32
contracts and timing disposition below remain unchanged.

This PR implements only the cached external deliverable of [Q2 #8](https://github.com/hsliuustc0106/vllm-lt/issues/8). It compares accepted PR14 compact native inference with the pinned official Ouro implementation on fixed-depth W1. No GPU result exists yet. The separate FP32 task-quality screen follows in another PR; BF16/adaptive comparisons remain blocked by Q1, so this PR cannot close Q2.

## Goal, constraints and non-goals

The goal is an auditable comparison of equivalent actual generation: the same 128 prompt tokens, 64 greedy outputs, FP32, four loops and ignored EOS. Exactly 22 production files match accepted PR14 `630a8fdc0dd47b6da68a3d30db6b851fc08af5c5`. The rejected M4 tile64 and unqualified M3 graph implementation are excluded. Native attention remains Triton tile32 with compact prepared metadata, conservative admission and last-exited paged KV.

The official baseline uses published eager attention, dynamic concatenating KV and fixed-four forward calls. It retains published gate PDF work during prefill. It is a specifically pinned implementation, not the fastest possible external implementation. The experiment does not establish adaptive, BF16, serving, long-context or task-quality claims. It introduces no inference optimization, extra model, autotuner, graph, cache-policy change or modified attention arithmetic. A valid slower native result is acceptable evidence; no speedup threshold is required.

## Adapter interface

`OfficialOuroCachedReference(config, weights)` shares caller-owned immutable FP32 parameter storage using the same verified meta/assign initializer as the unchanged Q1 full-prefix reference. It neither loads remote code nor downloads weights. The pinned published sources and Transformers 4.55.0 are verified; optional `kernels` must be absent. Q1 still records `use_cache=False`, while this adapter has separate provenance with `use_cache=True`, exact shim SHA and 96 depth/layer slots.

| Operation | Contract |
| --- | --- |
| `start(prompt, max_outputs)` | Ready state only. One full prompt call; returns unsampled `[vocab]` logits. Reject invalid IDs, bounds and unsupported full-attention settings before mutation. |
| `advance(actual_previous_token)` | Active state only. Consumes exactly one actual prior prediction at the next contiguous position. No hidden sampling, forced continuation or prefix recomputation. |
| `snapshot(inspect_cache=False, check_finite=False)` | Host-only call/counter metadata by default. Detailed shape/length/storage and optional finite scans are explicitly requested after the final delivery/synchronization boundary. |
| `complete(completion_confirmed=True)` | Host acknowledges all submitted work has completed after exactly the declared number of predictions. |
| `reset(...)`, `close(...)` | Release request/cache or wrapper references. Active/failed states require explicit completion confirmation. Caller-owned weights are never modified. |

The published cache assigns `key_cache`/`value_cache`, but installed Transformers exposes inherited getter-only properties; its inherited `get_mask_sizes` also expects a different `layers` representation. A local subclass adds writable list properties and `get_mask_sizes(cache_position, layer_idx)` returning `(old_length + input_count, 0)` for this contiguous, unpadded, full-attention case. Published `update`, attention, RoPE and normalization remain unchanged. The subclass remains an instance of the published universal cache, preventing automatic conversion to another cache implementation.

Actual cache indexing is `depth * 24 + physical_layer`: 96 distinct slots. Fixed calls use `exit_at_step=3`, `logits_to_keep=1`, `use_weighted_exit=False`, `use_cache=True` and autocast disabled. W1 performs one prefill at positions 0–127 followed by 63 one-token calls at positions 128–190. Final cache length is 191, not 192: the final predicted token is not forwarded. A failed forward poisons that request; there is no automatic retry.

## Shared driver and evidence interfaces

`q2_external_schema.make_plan(contract_path, model_path, gpu_ids=..., gpu_uuid=..., affinity=...)` produces a strict plan without device discovery or checkpoint tensor loading. It hashes the full source, imported module paths, all checkpoint/tokenizer/input files, cached official provenance, dependencies, exact UUID/physical ID, CPU/NUMA binding, environment and controls. `validate_plan` checks its internal relationships; `verify_plan` rereads actual identities before device use. The 128-token W1 fixture is byte-equivalent to M1.

`q2_external_driver.execute_case` reuses the loaded native engine and official wrapper. One native FP32 weight storage, one official view, and the native six-GiB pool stay resident throughout all eight executions. Actual parameter pointers/shapes/dtypes and common pool pointers are recorded and checked. Request state resets outside the delivery interval while the worker allocator state is preserved; shared caches are not dropped.

The common timing observer retains actual returned token IDs/depths and host timestamps. Arrival is before native submission or official prefill; a token timestamp is after greedy readback and copied output history are ready, before collector accounting. Final synchronization is recorded separately. M1's `summarize_records` independently reconstructs delivery throughput, TTFT, TPOT and synchronized wall time. Admission/submission subintervals are unavailable for both adapters; no synthetic timestamps or zero durations stand in for them. Native scheduler/API dispatch overhead and official forward/append-cache overhead remain part of their implementations. Native request release occurs in its normal final engine step; official cache inspection/reset occurs after delivery. Those lifecycle differences are recorded rather than hidden by a synthetic common scheduler.

There are no stage hooks, per-layer timers, profiler captures or page scans inside timed generation. The driver retains bounded native stage metadata after step return and official actual call/position records; this host accounting is included before the next step. Feasibility alone checks native recurrent/coda finiteness and all 64 official logit outputs. After official delivery, actual cache metadata proves 96 independent depth/layer histories, each length 191; feasibility additionally checks all final K/V tensors are finite. Numeric logits/tensors and traces are not exported by this experiment.

| Artifact | Meaning |
| --- | --- |
| `plan.json` | Immutable source/input/dependency/control identities, resource estimates and exact execution order. |
| `manifest.json` | Parent-owned process-group launch interval, completion prefix, failures and final status. |
| `worker.json` | One load/preparation interval, actual environment/arithmetic, weight/pool proof, finalized rows, equivalence gate and final cleanup. |
| `runs/ID/{started,result,completed,acknowledged}.json` | Full-case start/deadline, actual output/timing/cache/accounting, result hashes and immutable final acknowledgment. |
| `active-case.json` | Watchdog marker covering reset, execution, checks, export and acknowledgment. |
| Offline report | Recomputed raw metrics, exact feasibility/measured histories, provenance, bounded chronology, completed/missing/failed records and paired values/ranges. |

A result without matching worker/parent acknowledgment and immutable completion hashes is retained but cannot qualify as a completed measured execution. The controller watchdog owns one child process group; it terminates only that child on deadline, byte-cap or control failure. Device cleanup settles submitted work, releases task-owned request/model/workspace references and requires zero final allocated/reserved bytes. The post-run scheduler probe separately confirms the reserved device was released.

## Frozen finite budget

| Execution | Purpose | Included in A/B timing? |
| --- | --- | --- |
| Native feasibility | First real request after preparation; finite native outputs and actual sequence | No |
| Official feasibility | First official request; cached-call proof and all 64 IDs must match native | No |
| Native warmup, official warmup | Same W1 work, checked against feasibility | No |
| Native 1, official 1, official 2, native 2 | Two complete matched pairs in ABBA order | Yes |

The complete case limit is 600 seconds; the one-hour overall deadline also includes source verification, loading, export and final cleanup. There is one worker and one model load. Native executes exactly 380 engine steps; official executes exactly 64 forward calls. Each row emits 64 token events plus one finish event. The raw cap is 32 MiB total and 2 MiB per case, including partial records; no tensor dumps or profiles are allowed. The physical native pool is 6,442,450,944 bytes; equivalent official final K/V payload is 300,417,024 bytes in addition to the common resident pool. Native reserves 48 pages for W1's complete lifetime.

Any unexpected source/control, finite-value, actual-token, fixed-depth, cache/call/ownership, cleanup, deadline or artifact-budget failure stops later execution and preserves partial evidence. No automatic retry, replacement row, extra equivalence run, additional measured observation or profile is permitted. Both feasibility histories must match before warmups or measurements. Every later output history must match that frozen feasibility sequence.

## Acceptance criteria / 验收标准

| Q2 criterion | This PR's acceptance |
| --- | --- |
| AC-Q2-01 | Freeze accepted FP32 prerequisite, model/tokenizer/source/control/UUID identities, input W1 and numeric run/resource budgets before execution. Task dataset/prompt/parser and paired quality margins belong to the later quality deliverable. |
| AC-Q2-02–03 | Open, outside this external-only PR. No fabricated quality denominator, quality score, BF16/adaptive qualification or loss interval. |
| AC-Q2-04 | Tiny CPU cached/full-prefix and slot/position/lifecycle/storage tests pass; both real feasibility runs match all 64 actual IDs at four loops and prove genuine cached incremental generation. |
| AC-Q2-05 | All four ABBA measured rows complete with equivalent histories, identical reserved device and controls, valid request/worker cleanup and auditable raw timing. Negative speed results are valid; missing or noncomparable results leave the deliverable open. |
| AC-Q2-06 | Independent offline reconstruction, exact commands/manifests/CPU and device outcomes, two-pair observations/ranges, exclusions, cleanup, partial-prefix accounting and checksummed evidence. |

A comparison can be complete even when neither side has a demonstrated speed advantage. With only two observations per side, report raw values/ranges rather than infer statistical confidence; overlapping ranges support no winner. A pre-run resolved contract will record final source SHA, exact command/path hashes and selected device before GPU execution. No task-quality or whole-issue completion follows from this PR alone.
