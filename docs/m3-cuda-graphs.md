# Bounded Ouro CUDA graph experiment

This private FP32 path captures one Ouro recurrent traversal. It follows
[M3 #6](https://github.com/hsliuustc0106/vllm-lt/issues/6) after the inactive-row
and persistent-buffer prerequisites in PRs #13 and #14. It is opt-in. The
compact runner remains the default, and Q1 still gates BF16 separately.

The tensor body contains the 24 physical layers, shared normalization, the
actual gate logits, and copies into fixed output buffers. Prefill, prelude,
coda, sampling, CPU scheduling, cumulative gate hazards and LAST-EXITED KV
propagation remain outside capture.

## Ownership and interfaces

```python
engine._enable_recurrent_graph(use_graphs=True)  # before adding any request
# Existing add_request(), step(), and output delivery interfaces are unchanged.
engine.model_runner._close_recurrent_graph()   # after requests complete/abort
```

`use_graphs=False` selects the same private tensor path and buffer shapes with
eager execution. It is the experiment control. Both configurations load the
same frozen source revision. This comparison supports a decision about the
optional two-bucket executor; it cannot establish a gain over the compact
default by itself.

| Live rows | Physical rows | Table columns | Live slots |
| --- | --- | --- | --- |
| 1–2 | 4 | 32 | 1, 3 |
| 3–4 | 8 | 32 | 1, 3, 5, 7 |

Each bucket owns metadata, pageable CPU staging, hidden input/output, gate
output and gather indices. Tensor shapes, addresses, model weights and the KV
pool must remain stable for the executor lifetime. The graph pools are
independent so bucket order can change. Empty dispatches execute no tensor
body. More than four live rows, more than 32 required columns, and the Torch
attention backend use counted compact fallbacks. No new bucket is captured
during request execution.

`KVCacheManager._make_tensor_decode_view(storage)` resolves tensor metadata,
per-layer pool views and kernel callables before setup. Its result contains no
request allocation, lease generation or written-prefix state. The model's
`_recurrent_tensor(hidden, view)` and checked eager entry point share the same
arithmetic body.

The host transaction is:

1. Build and validate the logical batch; check every layer's preceding prefix.
2. Begin a traversal ticket, prepare the selected storage and bind its new lease.
3. Gather hidden rows and update dynamic positions, page addresses and lengths.
4. Execute the eager tensor body or replay once; gather fresh live outputs.
5. Read the actual sigmoid gate probabilities back to the host.
6. Commit every initialized current position and release the metadata lease.
7. Publish the fresh hidden rows, then run the existing routing/finalization.

Request hidden states do not alias graph buffers. A request waiting in coda
therefore survives other requests' replay. Python prefix mutations happen only
at the confirmed completion boundary. The manager prevalidates every row
before committing; a partial host mutation failure quarantines its storage.
An engine update failure after a successful commit preserves that completed
history and fails the executor. It does not claim to roll back device work.

Uncertain device completion retains graph, pool and borrowed-buffer ownership.
The executor cannot be retried. Safe close first establishes completion and
then releases graph resources; a failed close retains them for later settlement.

## Setup and limits

Setup precedes request admission and uses four task-owned pages from the same
fixed KV pool. It saves their bytes, seeds preceding history, then performs
three eager warmups, one capture recording and one explicit verification
replay per bucket on an ordered side stream. CUDA capture records commands;
it does not execute the recorded kernels. There are eight actual scratch GPU
traversals per graph-owning engine: six warmups plus two verification replays.

The setup verification requires finite physical hidden/gate outputs and
positive-zero inactive rows. Warmup/replay deltas are diagnostics; they do not
replace the independent real-model numerical contract. After confirmed
completion, setup restores every saved page byte and the exact allocator
free-list order. Runtime generations are measured relative to the post-setup
baseline. Allocation generations themselves are not rewound.

The implementation uses raw `CUDAGraph.capture_begin`/`capture_end`, with one
independent pool per bucket. It avoids helpers that add implicit warmups or
allocator-cache clearing. Capture and all outputs stay alive until safe close.

| Resource | Limit |
| --- | ---: |
| Common device tensor payload, both buckets | 256 KiB |
| Common CPU staging, both buckets | 16 KiB |
| Additional retained graph allocated memory | 256 MiB |
| Additional retained graph reserved memory | 256 MiB |
| Setup peak allocated increase | 512 MiB |
| Setup peak reserved increase | 512 MiB |
| Measured peak allocated/reserved increase, B minus A in each pair | 512 MiB |
| Complete executor setup | 60 seconds |

For the pinned model, common payload is 198,588 bytes and staging is 1,884
bytes. Saving four scratch pages costs 24 MiB of temporary host memory. The
memory baseline includes loaded weights, the existing pool and common bundles
after synchronization, before graph setup. Peak counters reset at that boundary.
Setup and retained-memory measurements remain separate from steady inference.

Configured budget violations fail the experiment. The private runner can
decline graph setup and use counted compact execution after a configured
budget limit is exceeded, provided completion, scratch restoration and graph
release succeed. The decline is recorded and cannot trigger a second setup
attempt. CUDA allocation errors, device errors, correctness failures or an
uncertain cleanup never trigger this fallback. The A/B harness requires the
configured executor and stops before inference when setup declines its budget.

The 60-second setup limit is a completion requirement checked between setup
phases. A blocked CUDA call is interrupted by the independent 600-second case
or 7,200-second global process watchdog. The implementation does not promise
that a blocked call can be interrupted at exactly 60 seconds. Such a setup
cannot qualify for performance measurement.

## Frozen validation and performance protocol

The resolved contract records exact source/model/dependency hashes, physical
GPU assignment, CPU/NUMA affinity, arithmetic controls and artifact limits.
The prepared checkpoint is real `ByteDance/Ouro-1.4B` at revision
`574fa66cb8bf5abdc979642d01cf2b79b16bfab1`. No download, setup, warmup or
profiler execution contributes to the measured throughput values.

The combined protocol has eight workers and 101 executions:

- 23 correctness executions: 15 real-model cases, two held lifecycle cases,
  and six direct kernel cases. The model cases split into 13 qualification and
  two excluded feasibility cases; 30 of their 31 comparison streams qualify
  correctness, and one feasibility stream is excluded.
- 14 excluded performance feasibility executions, one per side/cell.
- 28 warmups and 28 measured executions across the seven M1 cells, in worker
  order A1, B1, B2, A2.
- Four diagnostic warmups and four profiles for W1 and W4 refill, each profile
  bounded to the existing 16-output window.

N-A and N-B execute correctness and feasibility before measured workers start.
Each worker loads the model once. Real-model cases receive a fresh engine and
pool; held lifecycle cases use their own small runner/cache, and direct kernel
cases own independent guarded pools.
The 44 graph-owning B cases account for at most 88 capture recordings, 264
warmup device traversals and 88 verification device traversals. All work has
a 7,200-second global cap, a 600-second case cap and a 16-GiB raw-artifact cap.

The numerical projection records normalized loop outputs and actual gate
logits for the last prompt query and eight continuation queries, final logits,
exact token/depth histories, and full final populated KV across every physical
layer, all four materialized depths and every populated position. Full KV is
chunked by four positions. It does not claim per-layer diagnostic hooks or
full-KV snapshots after every loop. Those Python hooks cannot run inside replay.
Original FP32 logit tolerances are unchanged. Intermediate and full-KV values
must be finite; their numerical deltas remain diagnostic. The held lifecycle
also checks page guards and ownership transitions.

The required performance target is at least 10% higher W1 throughput in each
matched pair. Each of the six controls may regress by at most 5%, and W1 TTFT
may increase by at most 5%. W1's minimum B throughput must exceed its maximum A
throughput under the observed-range rule. Two samples are not a confidence
interval. Failed required gates take precedence over an inconclusive range.

M2's retained W1 results improved throughput by 20.19–21.06% while leaving a
per-layer launch sequence. That evidence motivates the launch experiment and
the predeclared engineering target; it does not predict a CUDA graph speedup.
Timing includes input preparation, output publication, actual gate transfer,
routing, finalization and all fallbacks. The ordinary path collects host
counters without rich pointer descriptors. Excluded profiler runs require
actual `cudaGraphLaunch` events correlated with dispatch, bucket and GPU work.
The pinned Torch build uses Kineto revision
`094d3c1d072362d0a919a77299459eee94f97931`, whose
[CUDA activity serializer](https://github.com/pytorch/kineto/blob/094d3c1d072362d0a919a77299459eee94f97931/libkineto/src/CuptiActivity.h#L430)
emits correlation and graph identities for this CUDA build. The report still
requires those fields in the actual retained trace.

Run only the frozen plan through the verified exact-ID GPU scheduler. Stop on
the first execution/control/resource failure, preserve the failed prefix and
do not expand the run budget to find a favorable outcome. A result report and
raw evidence must identify every unmet acceptance criterion before adoption.
