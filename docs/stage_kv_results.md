# Huginn stage-aware KV: storage and native CPU generation

Huginn's prelude and coda execute once per token; its core executes R times. `CacheConfig(stage_aware=True)` allocates the boundary layers once and retains a separate core KV plane for every recurrent depth. The default remains rectangular. For 2+4+2 layers at R32, actual KV tensors hold 132 planes instead of 256: 48.4375% less storage at identical logical token capacity.

The unit of admission, growth and release is a page bundle containing R core pages and one boundary page. Core depth IDs remain compatible with scheduler accounting. Explicit byte budgets round down to complete bundles. Preemption copies both pools page by page without a full request-sized GPU temporary. Coda can write after recurrent finalization; shallower exits copy only recurrent planes. Layer views and page mapping also route the recurrent CUDA Graph wrapper.

This is opt-in for native Huginn last-exited KV. Existing Huginn restrictions on prefix sharing, async scheduling, UVA and speculation remain. CUDA Graph address routing is implemented and GPU cases are registered, but CUDA Graph execution and automatic CUDA memory profiling are **NOT_RUN**. This CPU report does not qualify official checkpoint numerics, GPU throughput, high-load network serving, peak memory, full RL, or reward convergence.

## Correctness

The final regression passed **280**, skipped three official Nanbeige asset checks (uncached tokenizer, remote code and weights), and deselected 66 GPU tests. Fifteen new stage-cache cases cover:

- Exact R32 2+4+2 physical allocation, complete-bundle byte budgeting, atomic failed growth, release, stale ownership and request-ID reuse.
- An independent dense Huginn oracle at R1/R3/R32, with parallel and one-token execution; per-loop hidden state and final logits agree within atol3e-6/rtol3e-5 on tiny FP32 models.
- Exact generated tokens, exit depths and selected scores for rectangular versus stage layout, incremental allocation on/off, repeated request IDs and mixed exit trace32/1/17/2.
- Both-pool CPU snapshot/restore after physical relocation and real native preemption/resumption, retaining request RNG state.
- The Graph device-cache wrapper's page mapping on CPU; this is not CUDA capture/replay.

An earlier broad run had277 passes and two P/D fixture failures: those tests bypassed the constructor and omitted the existing `_update_version` field. Adding that field to the fixture resolved both; production error handling was not weakened. Earlier new-test API mistakes (output field name and selected-score request value) were corrected before measurements. Their failed receipts remain in the task evidence directory.

## Measurement contract

Shared macOS/arm64, existing Torch2.13.0/Python3.12.14, FP32 and one CPU thread. Tiny random Huginn width16, vocabulary13, stages2+4+2. Each burst enqueues2C requests, prompt lengths2–6, four output tokens, fixed R, temperature0.8 and explicit per-request sampling/latent seeds. The scheduler uses max_num_seqs=C, token budget64, prefill chunks2 and incremental allocation. Both arms receive5C page bundles. Recorded actual peak resident requests equal C in every observation.

Time starts before burst admission and ends after all final token/depth/score objects are exported. It includes the native scheduler, prefill and decode. It excludes model/engine construction, tokenization, network, reward, reference, optimizer, publication and checkpoint work. This is **in-process generation**, not full network serving or complete RL. Completion latency is measured from the common burst start and includes queueing; TTFT/TPOT were not measured.

Every observation exactly matches saved tokens/depths/scores and leaves zero used KV blocks. There were no OOM, numerical mismatches or incomplete requests in these finite CPU measurements. Identical outputs across repeats do not create independent quality samples. KV bytes sum actual allocated key/value tensors and exclude parameters, scratch, other resident engines and allocator overhead; they are not peak memory.

Three random-weight seeds31/32/33 form paired cohorts. Reported ratios are geometric means of parent-or-rectangular time divided by stage time. Approximate95% intervals use a t interval on three cohort log ratios (df2); this small shared-host experiment cannot establish broad stable performance. CPU regressions remain visible.

## Unmodified parent comparison

The baseline is the untouched #6 commit46145be03fc053da147f32c45f2a23893313cb70. All137 extracted Git blobs were independently matched. Twelve sequential fresh processes run parent/stage/stage/parent per seed; each process verifies its import root, warms once and measures once per case. One engine is present per case. The candidate uses stage-aware KV. This comparison includes the effect of the common KV access refactor.

| R | Actual concurrency | Clean parent ms | Stage ms | Speed ratio [95% CI] | Completion p95 ms, parent → stage |
|---:|---:|---:|---:|---:|---:|
| 3 | 1 | 56.09 | 51.07 | 1.098 [0.615, 1.960] | 73.07 → 79.60 |
| 3 | 16 | 375.44 | 315.98 | 1.188 [0.896, 1.576] | 419.32 → 338.54 |
| 3 | 64 | 1318.40 | 1091.30 | 1.208 [0.993, 1.470] | 1457.11 → 1333.14 |
| 32 | 1 | 384.79 | 361.11 | 1.066 [0.828, 1.372] | 463.13 → 514.21 |
| 32 | 16 | 2514.08 | 2562.23 | 0.981 [0.804, 1.197] | 2833.43 → 2702.39 |

## Same-tree layout ablation

Three separate processes each warm both arms and run three rectangular/stage/stage/rectangular blocks per case:180 observations. Both engines share immutable model parameters but retain distinct KV/state. The rectangular arm contains the candidate's common access refactor, so this is a layout ablation rather than an unchanged-parent baseline.

| R | Actual concurrency | Rectangular ms | Stage ms | Speed ratio [95% CI] | KV bytes, rectangular → stage | Reduction |
|---:|---:|---:|---:|---:|---:|---:|
| 3 | 1 | 51.74 | 57.36 | 0.902 [0.826, 0.985] | 30,720 → 20,480 | 33.33% |
| 3 | 16 | 345.66 | 351.65 | 0.983 [0.821, 1.178] | 491,520 → 327,680 | 33.33% |
| 3 | 64 | 1017.10 | 1021.84 | 0.995 [0.822, 1.205] | 1,966,080 → 1,310,720 | 33.33% |
| 32 | 1 | 382.95 | 384.84 | 0.995 [0.851, 1.164] | 327,680 → 168,960 | 48.44% |
| 32 | 16 | 2395.97 | 2399.19 | 0.999 [0.828, 1.205] | 5,242,880 → 2,703,360 | 48.44% |

Both comparisons show no stable generation speed benefit on this CPU workload. The stage layout reduces allocated KV storage, while boundary mapping and bundle bookkeeping can add overhead. GPU capacity and timing need separate qualification.

## Reproduction and retained artifacts

`benchmarks/bench_stage_kv.py` runs the same-tree experiment. `benchmarks/bench_stage_kv_clean.py` runs one fresh parent/candidate arm with `--source`, `--arm`, `--seed`, `--order`, `--expected` and `--output`; point PYTHONPATH at the selected clean source tree. It imports the shared workload by explicit file path and asserts the actual native-package import root.

Raw observations, complete expected outputs, descriptive summaries and SHA256 manifests are committed in `benchmarks/results/stage-kv-cpu/` and `benchmarks/results/stage-kv-clean-cpu/`. The task-local validation receipt records source hashes and final test counts. The successful tests used already-resident existing packages through a loader that fails on iCloud-evicted files; no package or environment was installed, replaced or updated. This task's other numerical tests had exited before formal timing. The host was shared and not reserved.

Official Huginn checkpoint, CUDA eager/Graph, fixed GPU budget capacity, network saturation, complete RL and reward convergence remain pending. R1/R3/R32 tiny tests are not official-model coverage.
