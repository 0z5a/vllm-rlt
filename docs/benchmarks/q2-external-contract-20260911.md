# Resolved Q2 cached external experiment — pre-run contract

**No device result is claimed.** This is the external FP32 deliverable of Q2 #8; task quality, BF16 and adaptive behavior remain outside this finite experiment. The design is [q2-external-design.md](q2-external-design.md).

One implementation comparison: accepted PR14 compact native Triton tile32 versus the pinned official eager cached implementation. Rejected M4 tile64 and unqualified M3 capture are excluded. All 22 native production files match accepted PR14 `630a8fdc0dd47b6da68a3d30db6b851fc08af5c5`. No optimization, new model or quality experiment is added.

## Pre-device selection amendment — GPU2

The original GPU1 plan `89437ce5a427e1a073011250ee7b585f44c419a2565542fe6b180a5fd07b0880` (file SHA256 `7e93e37be7f5a24f441816a52afba5d5a652e4eb4132b042a2eb8023b703e985`) was never executed. The immediate launch metadata showed that GPU1 had become IN_USE, so no reservation or device workload was started. This amendment selects available GPU2, UUID `GPU-cbf66259-f4ab-0ede-1811-82037dde5924`, before any feasibility or measured result exists. It is a pre-device availability change, not a replacement trial.

Only `contract.controls.gpu_ids`, `contract.controls.gpu_uuid`, and their dependent `controls_sha256`/`plan_sha256` values changed. The source, inputs, dependencies, eight-row schedule, 3600s/600s deadlines, 32MiB raw cap, one worker/load and resident 6GiB native pool are unchanged. CPU0–7 and memory binding0 remain valid on the root-verified GPU2 NUMA0 topology. The new plan was cloned from the already verified plan and checked with the frozen schema offline; no new checkpoint rehash or probe occurred.

Preserve the original plan at `artifacts/q2-plan-20260911/plan.json`, original contract at `artifacts/q2-contract-gpu1-unexecuted.md`, and original command/command JSON unchanged. `artifacts/q2-gpu2-amendment.json` records the exact four-field before/after diff and source hashes. The original occupied-device metadata remains local-only because it contains unrelated workload details; publication retains its file hash and the minimal IN_USE reason, not those details.

## Frozen identity

| Input | Identity |
| --- | --- |
| Execution source | `c96112fa69f3a3920e86317d80f396400c3d5e10`; `/data/hsliu2/tmp/vllm-lt-q2-external-exec-v2`; 135 tracked files |
| Canonical plan | `7c7c58c6043534d479acd8e605b870fa35856a2f088b14160fff74cff170bf71` |
| Exact plan file | `7cdfee9c2dfd01bd858c05e5479a6aaf1255d4aec270992c0b47ac7a4aa58785`; 178,924 bytes; `/data/hsliu2/tmp/vllm-lt-q2-external/artifacts/q2-plan-gpu2-20260911/plan.json` |
| Controls | `ac2218c0763b7a8312e65d732a05a6cca0cf4ae2dc9418d6bf1fd39de3046e9f` |
| Cached official provenance | `b272d55f57718d0dd8fec31bf6c593fb313d19d18ddb234805e64c0456ee6836`; full record in plan |
| Selected host record | `aff248a771fdfa97870028eae136d38a593ae2b80609d2b4aca70705e61af1b7`; `/data/hsliu2/tmp/vllm-lt-q2-external/artifacts/q2-host-gpu2-before.json`; 2026-09-11T01:18:14.180182+00:00 |

Host `dedicated-developjob-8gpu2-a029z-64896bc8cf-8p2lw`, account `hsliu2`. Physical GPU 2, UUID `GPU-cbf66259-f4ab-0ede-1811-82037dde5924`; logical CUDA device 0. CPU IDs `0,1,2,3,4,5,6,7`, memory node binding `0`, intra-op/OMP threads 1, inter-op threads 1, seed 0. The scheduler alone sets visibility. Refresh all four status forms immediately before launch; the earlier metadata record is not a reservation.

Interpreter `/home/hsliu2/tmp/venvs/vllm-lt-q1/bin/python` and prepared model `/data/hsliu2/tmp/vllm-lt-models/ouro-1.4b`. Full source, input/tokenizer/checkpoint byte hashes, dependencies, arithmetic and import origins are in the plan. No download, environment rebuild or checkpoint tensor load is part of the CPU probe/report. TF32 and reduced-precision reductions remain disabled.

No CPU result count is inferred by this generator. Preserve root full-suite logs, failed/final fixture logs, exact guards and independent reviews separately. CPU evidence file `/data/hsliu2/tmp/vllm-lt-q2-external/artifacts/q2-cpu-verification-final.json`, SHA256 `10044965bd7102dce146aa79647048f92401a360e132bde046622612b1adb341`: `{"full_cpu": {"corresponding_commit": "1af8716cb238121e0a20a84088eaefdcb7a9215c", "format_only_ast_equal": true, "log_sha256": "bae30d478f8bb9bd957a0d48c0729ead5b0d6a0af0514dd00ad812db1033cc3d", "passed": 761, "seconds": 145.99, "skipped": 16, "source_files_unchanged_during_run": 135}, "gpu_runs_so_far": 0, "pre_device_report_fix": {"changed_files": ["vllm_lt/benchmarks/q2_external_report.py", "tests/test_q2_external_report.py"], "execution_commit": "c96112fa69f3a3920e86317d80f396400c3d5e10", "focused_passed": 46, "log_sha256": "4e627f5d02c4c9da0b125924b8216f82c444926b9c7c5ec7a3bc59e88c859726", "production_or_driver_changes": false}, "real_cpu_integration": "128 prompt +64 actual native/official greedy outputs match,96 slots,length191; GPU bookkeeping stubbed, no GPU qualification"}`.

## Fixed workload and residency

W1 uses 128 prompt tokens, 64 actual greedy outputs, FP32 and exactly four loops; EOS is ignored. Both feasibility sequences must match every actual token before warmup. Every later row must match that sequence and depth 4. No continuation token is forced. Official generation performs one 128-token prefill followed by 63 single-token cached calls; 96 independent depth/layer cache slots end at 191 positions because the last predicted token is not forwarded.

The native engine and official view share a single immutable FP32 weight storage. The native **6,442,450,944-byte (6 GiB) pool stays resident for all eight rows**, including official rows. Actual weight/pool pointers and ownership are checked. Native reserves 48 pages; official final K/V payload is 300,417,024 bytes in addition to that common pool. Reset request/cache state outside delivery timing, retain worker allocator state, and never drop shared caches.

## One worker, eight rows, fixed stop rules

| Run | Implementation | Phase | Timing pair |
| --- | --- | --- | --- |
| N-feas | native | feasibility | excluded |
| O-feas | official | feasibility | excluded |
| N-warm | native | warmup | excluded |
| O-warm | official | warmup | excluded |
| N1 | native | measured | pair-1 |
| O1 | official | measured | pair-1 |
| O2 | official | measured | pair-2 |
| N2 | native | measured | pair-2 |

Exactly one model load and one worker: two feasibility, two warmup and four measured rows in native 1 / official 1 / official 2 / native 2 order. Native has 380 steps per row; official has 64 calls. No retry, replacement, extra equivalence execution, profile or extra measurement is allowed.

Global budget 3600 seconds; complete-case budget 600 seconds, including request reset, execution, checks, export, acknowledgment and cleanup. Raw output cap 32 MiB total / 2 MiB per case. No tensor dump or profile is allowed. The outer scheduler timeout is a termination allowance, not extra experimental runtime. Stop on source/control, finite/equivalence, cache/ownership, deadline, byte-budget or cleanup failure, preserving any completed prefix and partial records.

Raw destination: `/tmp/hsliu2-vllm-lt-q2-external-20260911/run`. All data are new task-owned ordinary files. Cleanup only this task's child process group, model and device memory. Require final zero allocated/reserved bytes, then independently record the selected device becoming available. A scheduler record never substitutes for worker cleanup.

## Acceptance and interpretation

Feasibility must prove finite outputs and genuine incremental cached histories, with all 64 actual native/official IDs equal. Warmup/measured output histories must match feasibility. All four timing rows need frozen controls, result/completion/ACK hashes and bounded lifetimes. A missing acknowledgment cannot be inferred from result files.

Use common host token-delivery timestamps after actual selection/readback, with final synchronization recorded separately. Recompute TTFT, TPOT and tokens/s from raw events; do not invent admission/submission durations. No stage hooks, per-layer timers or page scans occur inside timed generation. Preserve preparation/loading, resets and final cache inspection separately.

There is **no speedup minimum** for this external baseline deliverable. Both paired values and observed ranges are reported; a slower native result is valid comparable evidence. Overlapping ranges support no winner, and two observations do not establish statistical confidence. Corrupt evidence stays invalid, known required failures stay failed, and missing-only evidence stays inconclusive. No result closes the separate task-quality/BF16/adaptive Q2 requirements.

## Commands and chronology

The original CPU probe and source freeze precede this document; the GPU2 plan is a metadata-only amendment validated offline. Review and DCO-commit the resolved document before the first device execution. Do not rerun the probe into its existing output. Commands below are derived reproducible syntax; any exact earlier probe command is retained separately in the commands JSON. This generator executes none of them.

```bash
cd /data/hsliu2/tmp/vllm-lt-q2-external-exec-v2
numactl --physcpubind=0,1,2,3,4,5,6,7 --membind=0 env OMP_NUM_THREADS=1 PYTHONPATH=/data/hsliu2/tmp/vllm-lt-q2-external-exec-v2 /home/hsliu2/tmp/venvs/vllm-lt-q1/bin/python /data/hsliu2/tmp/vllm-lt-q2-external/artifacts/q2_cpu_cli.py probe --contract /data/hsliu2/tmp/vllm-lt-q2-external-exec-v2/benchmarks/fixtures/ouro-q2-external-contract.json --model /data/hsliu2/tmp/vllm-lt-models/ouro-1.4b --gpu-ids 2 --gpu-uuid GPU-cbf66259-f4ab-0ede-1811-82037dde5924 --output /data/hsliu2/tmp/vllm-lt-q2-external/artifacts/q2-plan-gpu2-20260911
gpu run --gpu-ids 2 --nonblock --timeout 70m --note 'Q2 cached external c96112fa69f3a3920e86317d80f396400c3d5e10' -- numactl --physcpubind=0,1,2,3,4,5,6,7 --membind=0 env OMP_NUM_THREADS=1 PYTHONPATH=/data/hsliu2/tmp/vllm-lt-q2-external-exec-v2 /home/hsliu2/tmp/venvs/vllm-lt-q1/bin/python -m vllm_lt.benchmarks.q2_external run --plan /data/hsliu2/tmp/vllm-lt-q2-external/artifacts/q2-plan-gpu2-20260911/plan.json --output /tmp/hsliu2-vllm-lt-q2-external-20260911/run
numactl --physcpubind=0,1,2,3,4,5,6,7 --membind=0 env OMP_NUM_THREADS=1 PYTHONPATH=/data/hsliu2/tmp/vllm-lt-q2-external-exec-v2 /home/hsliu2/tmp/venvs/vllm-lt-q1/bin/python /data/hsliu2/tmp/vllm-lt-q2-external/artifacts/q2_cpu_cli.py report --run-dir /tmp/hsliu2-vllm-lt-q2-external-20260911/run
```

Generator SHA256 `764dad7d7e94bfb1c65eaf103b64c5bc3c375d6dced1f3b6bf1cb0e5d51e9315`; CPU wrapper `1b6f4ab49606c8c9a3bf282b77833535e77e9695985b8d8b6d8b183ba682d846`; host metadata helper `4dff4e52b6b1dcf5ab514c8ad99881bffc8b1ca90b3b57d257672a7ebfd85c66`. No memory-access checker execution or quality/performance outcome is implied by preparation.
