# Decode async scheduling: preparation, execution, and output delivery

> Validation update: the 244-test result below is the pipeline-stage snapshot. Later KV/FlashAttention regressions and all performance sweeps are complete; see [全部测试汇总](testing_summary_20260915.md)。

This document describes the completed decode pipeline, replacing the earlier
implementation that could refill a request only after CPU coda delivery.
The relevant paper is **Section 4.3 and Figure 3, page 5**, in
`/tmp/cdb-paper-2608.09444.pdf`. Figure 3 overlaps preparation of batch N+1,
using gate N-1, with execution of batch N. Separate boundary execution is an
additional form of overlap, not the definition of async scheduling.

## Implemented behavior

| Requirement | Code | Behavior and verification |
| --- | --- | --- |
| CPU prepare overlaps preceding GPU forward | `ModelRunner.prepare`, `PreparedExecution` | Capture depths, positions, output indices and CPU KV addresses before submitting the next forward. CUDA event tests keep a preceding forward busy and assert that preparation completes first. |
| No CPU token roundtrip before the next prelude | `Submission.device_values`, `Request.input_token_tensor`, `ModelRunner._execute` | Coda's device sample feeds embedding directly. Tests hold CPU delivery while prelude and recurrent execution advance. |
| Scheduling progresses independently of delivery | `Request.num_output_placeholders`, `num_scheduled_outputs` | Position and trace lookup include submitted outputs. The public generated-token list contains delivered outputs only. |
| Stable output metadata | `Submission.depths`, `output_indices`, `LLMEngine._deliver_coda` | Deliver using the submitted token's depth/index even when the request has entered the next token's loop. Variable-depth replay tests deliberately delay delivery. |
| Nonblocking input copies | `InputStaging.prepare/transfer` | Four pinned host banks; two packed metadata copies per recurrent submission. Every submission gets fresh device metadata. A host bank waits only for its H2D completion before reuse. |
| CPU/GPU and copy/compute overlap | `ModelRunner.submit` | AS uses one GPU stream. AM uses core, boundary and input-copy streams; a CUDA event joins copied metadata to its consumer. Tests verify H2D completion while the previous core is still busy. |
| Independent core/boundary execution | `Scheduler.schedule(prefer_recurrent=...)`, `_step_async` | AM can give independent recurrent requests one turn when the preceding core is still in flight. Otherwise it refills first: a pending coda readback alone must not cause a smaller core batch. |
| Bounded submissions | `ModelRunner.submission_events` | At most three model submissions are retained in flight: room for two core rounds and an interleaved boundary stage. CPU prepare precedes backpressure on the oldest submission. |
| Delayed stop/abort and safe reuse | `_deliver_coda`, `_finish`, `Scheduler.finish`, `ModelRunner.release` | At most one undelivered output per request. Next-token prelude/core may run, but a second coda cannot sample before the first output's stop decision. EOS/abort removes queued work and waits for that request's last GPU event before freeing KV/state. |
| Existing routing policy retained | `_delayed_signal`, `_trace_exit` | Gate r-1 controls whether a token enters r+1, after r has been submitted. Threshold 1 uses the hard depth bound without collecting irrelevant gate scores. |

Sources: [engine](../vllm_lt/engine/llm_engine.py),
[runner](../vllm_lt/worker/model_runner.py),
[scheduler](../vllm_lt/core/scheduler.py),
[request state](../vllm_lt/request.py),
[input staging](../vllm_lt/worker/buffers.py),
[pipeline regression tests](../tests/test_async_pipeline.py).

## Concrete execution example

Suppose A and B have both submitted round 2. A exits at depth 2; B continues
to depth 4. A's current coda will produce output index 5, meaning five outputs
have already been delivered.

1. The host routes A to coda and B to recurrent. GPU events ensure A's coda
   consumes its final round-2 state and finalized KV.
2. Coda submission captures `depths=(2,)` and `output_indices=(5,)`. The engine
   immediately sets A's placeholder count to 1 and queues A's next prelude.
   Its scheduled output count is now 6, while its public token list still has 5.
3. Prelude consumes the sampled GPU tensor. It does not read token 5 from the
   CPU list. A's next input position already includes that placeholder.
4. If the preceding core is still running, AM can schedule B's independent
   round 3 alongside A's boundary work. Otherwise it prepares A's prelude
   immediately and refills the next core batch with A and B together.
5. A can now have `loops_done=1` for its next token when the previous coda is
   delivered. Delivery appends **depth 2**, taken from the submission snapshot,
   not depth 1 from the mutable request. Placeholder decrement plus token-list
   append leave the scheduled output count and position unchanged.
6. If that delivered token is EOS, A's queued next work is removed. Already
   submitted next-token work is allowed to complete before its KV/state is
   reused. No second sampled output is emitted. B continues normally.

For the lookahead dependency itself: while GPU round 3 runs, CPU consumes the
round-2 signal and prepares membership/addresses for round 4. An exit signal
means round 3 is the final round; otherwise the token enters round 4. This is
the r-to-r+2 convention in Section 4.3. It does not make an additional loop
speculative relative to the chosen delayed-gate policy. Delayed EOS can still
cause unused work on the *next token*, as described above.

## S / AS / AM switches

Keep exit policy, KV layout, depth bounds, model and workload identical.
Do not pass `--static-buffers` or `--pad-to-power-of-two` for dynamic baselines.

| Variant | CLI flags in addition to `--attention-backend triton --exit-mode ouro_delayed` | GPU streams |
| --- | --- | --- |
| S | No async flag | Synchronous execution on the current stream |
| AS | `--async-scheduling --single-stream` | One stream for model and transfers; CPU preparation can overlap GPU execution |
| AM | `--async-scheduling` | Core, boundary, and input-copy streams |

No new user-facing switch is required. Static buffers remain opt-in. Async
always has pinned **host** staging/readback pools for safe DMA ownership;
this does not enable persistent static device workspaces, persistent request
state slots, padding, or CUDA graphs.

`random_lookahead` still uses random weights; there is no distilled lookahead
checkpoint. `ouro_delayed` reuses the original trained early-exit gate but is
a delayed-routing heuristic, not a trained lookahead predictor. Compare sync
and async using the same exit policy; this change does not establish model
accuracy with an untrained lookahead head.

## Validation and reproducibility

The final full CPU/CUDA suite on B300 passed **244 tests**, with **11 skipped**
for the optional `lm_eval` dependency. GPU tests ran. The 21 new pipeline tests
cover forced delayed delivery, variable depths, EOS, cancellation/ID reuse,
slow GPU backpressure, DMA reuse, device token feedback and overlap topology.
The artificial CUDA delays in topology tests prove ordering, not speedup.

```bash
cd /home/zjy/code/david/b_workspace/vllm-lt
CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /home/zjy/code/david/b_workspace/.b_rdma/bin/python -m pytest --run-gpu -q \
  > /home/zjy/code/david/tmp/ouro-async-pipeline-20260915/verification.log 2>&1
```

Raw test logs, source snapshots/hashes, benchmark commands, per-request token
IDs/depths and profiler traces are retained under:

`/home/zjy/code/david/tmp/ouro-async-pipeline-20260915/`

The accepted performance runs are `verified-c1`, `verified-c4`, and
`verified-c8`; earlier folders are intermediate experiments, including rejected
multi-stream scheduling policies. `verified-source/` and
`verified-source-sha256.json` identify the benchmarked code.

Performance measurements use the real local Ouro-1.4B checkpoint, BF16,
LAST-EXITED, 128 input tokens, 512 output tokens, greedy sampling, ignored EOS,
and exactly four loops. Each measurement completes two waves of requests
(`2 * concurrency`) after a full-length warmup, with three randomized-order
repeats of S/AS/AM. Initialization/warmup are excluded; admission and drain are
included. Each concurrency runs on a separate GPU with fixed CPU affinity.
There is no profiler in throughput runs; lightweight batch and event counters
are enabled equally where applicable.

The profiler capture in `ops-shapes/` uses `torch.profiler`, CPU/CUDA operators
and shapes only: no stacks, memory events, or custom ranges. Profiler timings
are diagnostic and must not replace the unprofiled throughput measurements.

Event observations distinguish GPU work still in flight at preparation start
from GPU work still in flight after the entire preparation; neither is an
overall GPU utilization percentage.

## Accepted performance results (2026-09-15)

Medians of three repeats, with static buffers/padding/graphs disabled:

| Concurrency | S tokens/s | AS tokens/s | AM tokens/s | AS vs S | AM vs S |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 33.01 | 32.83 | 32.69 | -0.57% | -0.97% |
| 4 | 128.03 | 126.11 | 126.59 | -1.49% | -1.12% |
| 8 | 255.12 | 255.47 | 254.17 | +0.14% | -0.37% |

All **27 trials**, **234 requests** and
**119,808 output tokens** completed without errors. Output token
IDs matched across variants/repeats for each tested prompt; every exit depth
was four. Source hashes match `verified-source-sha256.json`.

At concurrency 8, S executes 4,095 recurrent batches for 32,704 recurrent
rows. AS executes 4,096; AM executes 4,096–4,097. Mean batch size is about
7.98 in all modes. The earlier diagnostic had AS at 5,075 calls (mean 6.44)
and AM at 5,115 calls (mean 6.39) for the same 32,704 rows. CPU coda delivery
no longer prevents the next token from refilling the batch.

These measurements show restored batching and near-parity, **not a reliable
async speedup**. At concurrency 4, AS and AM medians remain roughly 1–2% below
S; the table reports that residual difference rather than calling it zero.
In the unprofiled event observations, almost every previous GPU submission
had completed by the time recurrent CPU preparation began; only one of the
measured preparations still had the preceding submission in flight after
preparation. The topology tests prove that the pipeline can overlap, while
this eager fixed-depth workload offers little such overlap in practice.
Reducing Python/eager launch overhead remains separate performance work;
CUDA graphs were neither enabled nor implemented here.

Additional real-model correctness validation used **12 cases**:
FP32/BF16 × LAST-EXITED/SHARED × S/AS/AM, each with eight requests, prompt
lengths 1–8, chunked prefill, and 16 outputs with per-token exit depths cycling
through 1–4. All token IDs and exit depths matched the synchronous reference
within each dtype/layout, and KV/state were reclaimed. See `ragged-real.py`,
`ragged-real.json`, and `ragged-real.log`. Their wall times are not throughput
comparisons because the cases were not independently warmed up.

The final test, benchmark, ragged-replay and profiler processes all exited
successfully. `ops-shapes-traces.zip` contains the three torch profiler traces.
Earlier BF16 observations in the historical runtime document are not erased:
these new checks establish equivalence for the stated workloads, not every
possible prompt or gate threshold.

To repeat the concurrency-8 throughput comparison with a fresh log directory:

```bash
cd /home/zjy/code/david/b_workspace/vllm-lt
RUN_DIR=/home/zjy/code/david/tmp/ouro-async-recheck-$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$RUN_DIR"
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  taskset -c 8-11 /home/zjy/code/david/b_workspace/.b_rdma/bin/python \
  -m benchmarks.runtime_baseline \
  --model /home/zjy/code/david/b_workspace/models/Ouro-1.4B \
  --variants S AS AM --concurrency 8 --requests 16 --repeats 3 \
  --input-length 128 --output-length 512 --output "$RUN_DIR/c8" \
  > "$RUN_DIR/c8.log" 2>&1
```

Choose an available GPU first. For concurrency 1 or 4, change concurrency and
request count together to `(1, 2)` or `(4, 8)` and use a new output directory.
The accepted run's lightweight counter wrapper is retained as
`/home/zjy/code/david/tmp/ouro-async-pipeline-20260915/count_pipeline.py`.
