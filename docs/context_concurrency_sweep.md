# Context/concurrency decode sweep (2026-09-15)

## Token budget

This sweep uses `max_num_batched_tokens=2048` and `prefill_chunk_size=2048`
for every S/AS/AM case. Decode consumes one scheduled token per request per
loop, so a concurrency of 256 needs 256 tokens of this budget, independent of
context length. A 64K KV history does **not** count as 64K newly scheduled tokens.
The budget is a ceiling, not padding; static device buffers, padding and CUDA
graphs remain disabled.

A budget of 512 would already cover the maximum decode batch. The larger 2048
budget reduces prompt chunking overhead; a nominal 64K prompt needs roughly
32 chunks instead of 128. Increasing only the budget without also increasing
the per-request prefill chunk limit would not achieve that improvement.

## Matrix and memory

| Nominal context | Actual prompt tokens | Final KV positions | Concurrencies | Reserved KV at maximum concurrency |
| --- | ---: | ---: | --- | ---: |
| 1K | 1,024 | 1,151 | 1, 8, 32, 64, 128, 256 | 216 GiB |
| 4K | 4,096 | 4,223 | 1, 4, 16, 32, 64 | 198 GiB |
| 16K | 16,384 | 16,511 | 1, 2, 4, 8, 16 | 193.5 GiB |
| 64K | 65,409 | 65,536 | 1, 2, 4 | 192 GiB |

The real Ouro-1.4B checkpoint uses 24 layers, 16 KV heads, head dimension 128,
and four stored recurrence depths in LAST-EXITED. BF16 therefore requires
`2(K/V) * 24 * 16 * 128 * 2(bytes) * 4(depths) = 786432 bytes`
per position, or **0.75 MiB**. Model weights are small compared with these KV
allocations. Allocation includes full request lifetime and block rounding.

The model's positional limit is 65,536. Every trial generates 128 tokens,
so the nominal 64K group reserves room for decode instead of exceeding that
limit. Its prompt length differs explicitly from a 65,536-token input.

Each point compares S, AS, AM with three repeats and full-length warmup.
The maximum concurrency additionally runs controlled mixed depths 2/3/4
using `ouro_delayed`, per-request minimum depths and thresholds 0/1. This
exercises gate readback and heterogeneous stages without attributing quality
to an untrained lookahead predictor.

## Measurement protocol

Implementation: [benchmarks/context_sweep.py](../benchmarks/context_sweep.py).

This is a **decode-only saturation/overlap experiment**, not end-to-end serving
throughput. All requests repeat the same prompt and start from completed
prefill. Actual model prefill is computed once and copied into **independent
physical KV pages** for the other requests. There is no fake KV, shared-page
optimization, or uncomputed hidden state. Prefill and replication times are
recorded separately and excluded from decode timings.

The immutable prompt prefix is retained. Before each warmup and measurement,
request state, queues, output placeholders, hidden states and written-prefix
metadata are restored. Old decode suffix bytes are excluded by current context
length and overwritten before use. The measured runner is the same runner
used for its warmup. Tests compare this protocol with normal prefill/generation
on CPU and CUDA for fixed and mixed depths.

The cache pool reserves the largest concurrency for each context throughout
that context's sweep; smaller points use only their active requests' pages.
This keeps memory allocation out of timing. The entire pool is freed when the
context process exits.

Timings start before the first coda after prefill and end after output delivery
and GPU completion. Both sampled-output throughput (including that first coda)
and decode-token throughput (excluding one prefill-produced output per request)
are reported. Batch histograms, inter-token latency and whether GPU work remains
in flight before/after CPU preparation are logged. Event counts are overlap
observations, not a GPU utilization percentage. `nvidia-smi` telemetry is also
saved. No profiler is active in throughput measurements.

Output tokens and depths are checked across S/AS/AM and repeats at each point.
The script records any differences and fails acceptance if differences occur.
This is correctness evidence for the stated workload, not model-quality scoring.

## Large-KV bug found by the sweep

The initial 16K/64K runs failed with CUDA illegal memory access; growing
concurrency also exposed it in shorter contexts. The Triton attention kernel
loaded int32 physical block IDs and multiplied them by the block stride in
int32. These products can exceed `2**31` **elements** even when each operand
fits in int32. The earlier small KV pools did not exercise those addresses.

The fix promotes block IDs to int64 **before** the K/V address multiplication.
Block tables remain compact int32 tensors. A strided-cache regression crosses
that address boundary while touching only a few logical rows: it reproduced
the CUDA error before the fix and passes afterward. Full validation after the
fix: **249 passed, 11 optional-dependency skips**; CUDA tests ran.

Sources: [kernel](../vllm_lt/kernels/triton_attention.py),
[address regression](../tests/test_attention_64bit.py),
[prefix-protocol tests](../tests/test_context_sweep.py).

## Runs and reproducibility

Root: `/home/zjy/code/david/tmp/ouro-context-sweep-20260915/`

The failed initial runs are retained at the root and are not accepted results.
The corrected sweep is under **`round2/`**, with one context per GPU 4–7 and
fixed CPU affinity. GPU 0–3 had external allocations and were excluded.
`source-sha256.json`, source copies, command JSON, GPU telemetry, raw output
IDs, per-trial summaries and exit codes are retained together.

Example for a new output directory on an available GPU:

```bash
cd /home/zjy/code/david/b_workspace/vllm-lt
CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /home/zjy/code/david/b_workspace/.b_rdma/bin/python -m benchmarks.context_sweep \
  --model /home/zjy/code/david/b_workspace/models/Ouro-1.4B \
  --context 1024 --concurrencies 1 8 32 64 128 256 \
  --output-length 128 --max-num-batched-tokens 2048 \
  --repeats 3 --mixed-at-max --output /path/to/new/run \
  > /path/to/existing/log-directory/context-1024.log 2>&1
```

Final `round2/` results: **207/207 measurements**, all four processes exited 0, no output disagreements. See [consolidated results](testing_summary_20260915.md).

## Static buffer and padding ablation

The completed follow-up run is retained under
`/home/zjy/code/david/tmp/ouro-context-sweep-20260915/static-ablation/`.
It waited for the existing 1K sweep to finish successfully and GPU 4 to become
idle. The driver invoked CUDA prefix-protocol tests, but omitted `--run-gpu`: all
six were skipped, not passed. A later audit actually ran all six successfully
(`gpu-tests-audit.log`); the original skipped log is retained.
CPU protocol validation passed all six fixed/mixed-depth and buffer-mode cases.
The driver executes a saved source snapshot and records status, commands,
telemetry, output IDs/depths, exit code and a final summary in that directory.

This first ablation uses 1K context, concurrency 32 and 128, plus mixed 2/3/4
exit depths at concurrency 128. Each case compares all nine combinations of
S/AS/AM and the following buffer modes, with three repeats (81 measurements):

| `--buffer-modes` value | Static device buffers | Power-of-two batch padding | CUDA graphs |
| --- | --- | --- | --- |
| `dynamic` | Off | Off | Off |
| `static` | On | Off | Off |
| `static-pad` | On | On | Off |

Add `--buffer-modes dynamic static static-pad` to the sweep command to enable
this ablation. Every repetition shuffles the nine combinations. Each measured
trial has a full-length warmup; all combinations share the same real prefill
checkpoint and are compared for identical output token IDs and exit depths.
Token budget/chunk size remain 2048 and output length remains 128. Dynamic
baselines are rerun within the ablation, since its KV pool is sized for 128
requests rather than the original 1K sweep's 256 requests.

Static buffers alone do not fix batch shapes. Padding rounds each stage batch
up to the next power of two; it does not pad every batch to 2048. Fixed-depth
power-of-two concurrency provides a control, while mixed exits exercise smaller
and irregular stage batches. This is not a reproduction of the paper's full
static-shape/CUDA-graph configuration: graph capture/replay is not implemented.

Early exit can reduce total compute, but a larger async/sync speedup is not
guaranteed. Mixed exits create independent boundary/core work and gate readback,
but also smaller batches and more scheduling. Compare S/AS/AM within the same
exit policy; do not attribute the raw fixed4-to-mixed234 throughput change solely
to asynchronous scheduling. Mixed234 uses controlled thresholds/minimum depths,
not a distilled lookahead gate or a model-quality evaluation.

The static ablation completed **81/81 measurements**, exit code 0, with matching
outputs across all buffer/scheduling combinations. Static/padding did not improve
throughput in these cases; see the [final tables](testing_summary_20260915.md).
