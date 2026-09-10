# M3 owned graph pools: 2026-09-11

**Milestone: incomplete/inconclusive; the optional graph path remains unqualified.** All 101 executions completed in 865.722 seconds, and both timing pairs, numerical checks, owner checks and memory caps pass. Required graph-replay evidence is missing from the frozen W4 profile window. Keep [PR #15](https://github.com/hsliuustc0106/vllm-lt/pull/15) as a draft; no third device attempt or additional profile was run.

This is the changed candidate under the [pre-run contract](m3-capture-owned-contract-20260911.md), using identical clean A/B source `0e4516a011ccea2e96900d2092dbe92573b3fd4a`. Only graph replay is enabled on B. A uses the same padded tensor body; these measurements do not establish gain over default compact execution. The earlier [stopped candidate](m3-capture-stopped-20260911.md) failed its 512-MiB memory cap and remains separate evidence; no observations were pooled.

## Frozen execution and measured results

Real `ByteDance/Ouro-1.4B` weights/tokenizer at `574fa66cb8bf5abdc979642d01cf2b79b16bfab1`, FP32, Triton 3.7.1, Torch 2.13.0+cu130, Python 3.12.13; physical GPU 0, L20X, UUID `GPU-006eb78c-23cb-f37d-eb7c-0ccb578b8f11`. CPU cores 56–63, memory node 1, one intra/inter-op thread, seed 0, and reduced-precision arithmetic flags disabled. All eight workers used the same prepared environment and model files. Raw output used the same local overlay filesystem on both sides; this changed storage control is not proof of the first attempt's I/O-error cause.

Order was N-A, N-B, A1, B1, B2, A2, P-A, P-B: 23 correctness executions, 14 excluded feasibility executions, 28 warmups, 28 measured executions, four diagnostic warmups and four profiles. Setup/compilation, model loading and diagnostic work are outside measured throughput. Pair 1 is A1/B1; pair 2 is A2/B2.

| Cell | A tokens/s, pairs 1 / 2 | B tokens/s, pairs 1 / 2 | B/A, pairs 1 / 2 | Peak reserved increase, MiB, pairs 1 / 2 |
| --- | ---: | ---: | ---: | ---: |
| W1-refill | 20.560 / 20.851 | 42.087 / 42.199 | 2.047095 / 2.023773 | 276 / 276 |
| W2-refill | 138.699 / 141.515 | 141.631 / 140.215 | 1.021141 / 0.990811 | 310 / 310 |
| W3-refill | 17.817 / 17.923 | 17.894 / 17.813 | 1.004296 / 0.993906 | 344 / 344 |
| W4-refill | 137.430 / 139.627 | 158.503 / 157.409 | 1.153330 / 1.127356 | 378 / 378 |
| W4-no_refill | 108.225 / 110.581 | 138.719 / 140.544 | 1.281763 / 1.270956 | 412 / 412 |
| W5-refill | 109.031 / 110.618 | 126.848 / 126.063 | 1.163410 / 1.139624 | 446 / 446 |
| W5-no_refill | 108.625 / 111.191 | 126.833 / 126.677 | 1.167621 / 1.139266 | 480 / 480 |

Both W1 throughput ratios exceed 1.10. Its TTFT ratios are 0.974207 and 0.952014 against the 1.05 ceiling; all six control throughput ratios exceed 0.95. W1's A range is 20.560–20.851 tokens/s and B range 42.087–42.199, strictly separated. These are two measured observations per side, not a confidence interval or a general serving claim. Every paired peak-allocated increase is 33,655,808 bytes. The maximum peak-reserved increase is 503,316,480 bytes (480 MiB), below 512 MiB by 32 MiB.

## Correctness, lifetime and remaining memory limitation

All 15 model cases and 31 comparison streams completed: 13 qualifying cases/30 qualifying streams plus two excluded feasibility cases/one feasibility comparison. The 3,879 compared boundaries include 279 final-logit records; zero required numerical, top-1, behavior or nonfinite failures were reported. Final logits retain the FP32 atol 0.001/rtol 0.0001 policy and discrete histories match. Normalized loop outputs, actual gate logits and full final populated KV prefixes are retained under the projected-observation contract; this does not claim per-layer hooks inside graph replay or a full KV snapshot after every loop.

Two lifecycle executions retain 346 typed records and 960 guard chunks. Six direct kernel executions retain six output records and 240 guard chunks. Model observations include 304 dispatches and 624 published rows. The source-bound owner audit covers 44 B executors/88 independent pool-owner lifetimes, null owners on A/Torch, stable recorded owner/executable identities and empty closed bucket inventories. Setup performed 88 recordings, 264 scratch warmup traversals and 88 verification traversals; capture recording itself is not an executed traversal. Maximum recorded B executor setup was 1.520519 seconds, within the 60-second setup limit.

**Repeated create/close cycles still accumulate reserved memory.** In each B timing worker, setup adds 38 MiB and the net reservation after each engine release grows by 34 MiB, reaching 12,156 MiB after 14 engines; A remains at 11,680 MiB. Live allocated memory after every engine release is identical across these observations. The bounded 14-case memory envelope passes, but the owner change does not prove stability for arbitrary executor churn or establish the allocator cause. All eight terminal workers record zero allocated/reserved bytes after workspace release; the scheduler subsequently reports GPU 0 available with 0 MiB used.

## Profile requirement and versioned report correction

W1's matched windows cover steps 2–97 and 16 subsequent outputs. B has 64 actual CUDA graph launches with unique correlations linking 73,664 GPU kernel events to the stable bucket-4 graph; A has zero graph launches. Total host launch API records fall from 73,712 on A to 432 on B. GPU kernel events remain 73,712/74,032, so this demonstrates launch reduction, not removal of model work. Diagnostic event durations are not end-to-end latency savings.

W4's matched windows cover steps 2–35 and 16 subsequent outputs. All seven recurrent batches have eight live rows and correctly use compact fallback, because the executor supports at most four live rows. Consequently both windows contain zero graph replays. B's 95 replay calls elsewhere in the full inference cannot replace the required in-window replay evidence. Bucket-8 correlation is not established by W1. This coverage gap prevents milestone acceptance even though all four trace files and all measurements are present.

The original frozen reporter counted CPU `user_annotation` and mirrored GPU `gpu_user_annotation` scopes together, producing two false W1 errors alongside the real W4 gap. Its original report remains byte-for-byte unchanged: JSON SHA-256 `c2d2ba2b0f9dfc1af8549a2abb7208450842d987f442707ff4c074cf8e923bfb`. Post-run auditor `c39dfb844e3964727f0c9838f25e5e33f88cc148` counts host scopes separately and classifies a valid fallback-only candidate window as missing replay coverage; malformed or uncorrelated claimed replay remains invalid. It does not weaken W4's requirement. The separate CUDA-blocked reassessment took 50.100 seconds and reports no errors, one missing W4 proof, **incomplete/inconclusive**. Reassessment SHA-256: `c059ce72c345adba4de26dfe9cd4da7c0b220d39f751588d32440c602a366ff3`. The independent profile audit preserves its original producer/result and a separately versioned correction with the same remaining gap.

## Acceptance and reproducibility

| Required gate | Outcome |
| --- | --- |
| Frozen source/controls, complete 101-execution budget, all eight worker cleanups | Pass |
| FP32 numerical/history, inactive/lifecycle guards, owner/identity checks | Pass |
| Both W1 throughput/TTFT pairs, all controls and both memory caps | Pass within the declared bounded workload |
| Actual W1 bucket-4 replay correlation | Pass |
| Required W4 replay/profile coverage | Missing; unqualified |
| Overall graph milestone | Incomplete/inconclusive; PR remains draft |

At execution source, 926 CPU tests passed with 16 skipped; Ruff lint/format passed. The earlier 906-pass/20-fixture-error run and 88-test fixture correction remain retained. The post-run reporter change passed all 71 targeted CPU tests and Ruff. No new device memory-checker execution was possible; no BF16, task-quality, injected-device-fault or unbounded-churn qualification is claimed.

The [pre-run contract](m3-capture-owned-contract-20260911.md) contains exact reproduction commands, source/model/input hashes, limits and exclusions. Plan identity is `1f14fcc26b43d707f911430456c62ab81805f54961b2f72d3562382e8fde3cc2`. Raw records are retained at `/tmp/hsliu2-vllm-lt-m3-capture-20260911/run-owned-pools`; support and the separate reassessment are under `/home/hsliu2/tmp/vllm-lt-m3-capture/artifacts`. The [evidence package](https://github.com/hsliuustc0106/vllm-lt/releases/tag/m3-capture-owned-evidence-20260911) preserves both reporter versions, original A/B sources, all traces/typed payloads, commands, CPU/probe logs, owner/manual/profile audits and cleanup. Checkpoint weights stay outside source Git and the evidence package. Byte-integrity packaging is separate from an ordinary-path rerun of the offline report.

M4 proceeds from the qualified compact synchronous implementation, with a separately frozen attention candidate. A future graph qualification must select a profile window that actually exercises the supported mixed-row buckets before reserving a device; it is outside this completed run budget.
