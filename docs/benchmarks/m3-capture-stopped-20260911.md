# M3 capture attempt 1 — stopped by I/O failure

The candidate **failed a required memory gate**, and the shared-filesystem attempt is **incomplete**. It stopped on a controller `OSError: [Errno 5] Input/output error` during worker B2 after 314.510195961 seconds. No further GPU execution was started under this contract. The original reporter returned `evidence_status=invalid`, `decision=inconclusive` because of the prefix-audit bug described below. Independent validation recovered a completed-pair memory failure, so the warranted outcome is **failed with incomplete coverage**; issue #6 cannot pass on this attempt.

The [pre-run contract](https://github.com/hsliuustc0106/vllm-lt/blob/27f9e80c998a9658c6ffbad28431e8a1d211a8cc/docs/benchmarks/m3-capture-contract-20260911.md) was committed at `5a08fa86482719bc7291d6f498956acb5f0a6d2a` before reservation. Both execution checkouts used source `0d9284a849c3ecdc2bbb5fbc4103729c98ce998d`, plan `528e2665c0bf5588a4cb5ea100c7c46188f7fc3600b8b2c7eb54e5e46c5e78a6`.

## Retained coverage

The pre-timing correctness gate completed and passed: 15 model cases and 31 comparison streams (13 cases/30 streams qualify; two cases/one stream are excluded feasibility), two lifecycle evaluations and six direct kernel evaluations. The numerical audit verified 120 eager dispatches, 120 actual replay dispatches and 64 counted backend fallbacks. These bounded correctness results do not establish full performance acceptance.

The parent acknowledged 65 executions through N-A, N-B, A1 and B1. Worker B2 acknowledged two further warmups, giving 67 worker acknowledgments. A third B2 warmup has a complete result and case cleanup record before controller termination but no worker acknowledgment; it is a result-only tail, not a completed planned worker. B2 measured runs, A2 and all four profiles are absent. Partial timing observations are preserved and are not pooled into any subsequent experiment.

The first offline reporter discards the trusted worker prefix when it encounters B2's unfinished manifest. This produces zero eligible timing records and cascading missing-worker errors even for earlier complete workers. The original report remains unchanged; a later, separately identified report correction must distinguish valid terminal workers from interrupted or corrupt evidence. The corrected, CPU-only reassessment recovered four trusted workers and 14 eligible first-pair timing rows, with no invalid-evidence errors: **failed / incomplete**. It retained the 532 MiB breach ahead of missing later observations. The reassessment is separately saved as `artifacts/capture-prefix-reassessment.json`, SHA-256 `ac3f3a8f0ebfbdf163c2ec19ce6749ce943b210d9915d5914f068af5433446a5`; the original report is unchanged. The targeted controller/report suite passed 72 CPU tests. This correction cannot manufacture missing observations or close M3.

## Failure and cleanup limits

The controller stored the exception type/message but no traceback, failing syscall or path. Its monitored operations include active-marker reads, artifact directory traversal/stat and child waiting; the exact cause is unresolved. The run directory resides on the shared FUSE filesystem. That is a reason to test local artifact storage separately, not proof that the filesystem caused this event.

The controller's child cleanup path terminates only its own process group. The scheduler command exited with status 1; post-stop status checks found exact GPU 0 available and `nvidia-smi` reported 0 MiB. B2 has no terminal worker memory/cleanup record, so this external observation does not assert that B2 completed its in-process teardown. No other workload or reservation was modified.

All original plans, manifests, partial results, typed payloads, events, console logs and reports remain under `artifacts/capture-run-20260911`; they must be published alongside later evidence. The original report SHA-256 is `a64ea1f46ec2c50f1f3fe4891499dc34ef7f95580ad5d6e0881a5543c4835b96`. The independent stopped-run audit and publication manifest supply exact byte coverage.

## Required failure and disposition

The completed A1/B1 W5-no-refill pair increased peak reserved memory by **557,842,432 bytes (532 MiB)**, exceeding the frozen **536,870,912-byte (512 MiB)** ceiling by **20 MiB**. Its peak allocated increase was 33,655,808 bytes and its throughput ratio was about 1.111601; passing those separate gates does not override reserved memory. The original exception happened later, during B2 warmup.

A local-storage repeat was considered after the I/O error, and a CPU-only storage probe plus unexecuted draft fixture were retained. That repeat was cancelled when the independent audit exposed the prior required memory failure. No second GPU run or modified acceptance limit was used. The partial measurements are not pooled with any future experiment.

The graph path remains private, opt-in and unqualified. This candidate cannot make a PR ready: its memory failure and absent second pair/profiles must remain visible. A future optimization needs a concrete change addressing the resource requirement and a separately frozen experiment; this report does not claim an allocator cause from peak snapshots alone.
