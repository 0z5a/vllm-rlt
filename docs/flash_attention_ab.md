# Triton versus official FlashAttention: paired decode benchmark

## Protocol

Run root: `/home/zjy/code/david/tmp/ouro-flash-attn-20260915/ab/`.
The driver updates `results-summary.md` every ten seconds; a row appears only
when both backends have completed all three repeats. Raw results include each
trial's output tokens, exit depths, stage batch histogram, overlap event counts,
ITL percentiles, memory peaks, elapsed time and throughput.

| Parameter | Setting |
| --- | --- |
| A | Existing correctness-first Triton paged attention |
| B | Official FA4 4.0.0b30, chosen by `flash_attn` on B300 |
| Context / concurrency | 1K: 1, 32, 128; 4K: 1, 32, 64 |
| Scheduling | S, AS, AM |
| Exit policy | Fixed four rounds at every concurrency; mixed 2/3/4 at maximum concurrency |
| Buffer mode | Dynamic; static buffers, padding and CUDA graphs off |
| KV | LAST-EXITED, BF16, 16-token pages |
| Output length | 128, greedy, ignore EOS |
| Token budget / prefill chunk | 2048 / 2048 |
| Repeats | Three measured trials, each following a full-length warmup |
| Total | 144 measured trials, plus 144 warmups |
| Devices | GPU 4: 1K; GPU 5: 4K; A/B interleaved on the same GPU |
| CPU binding | 40–43 for 1K; 44–47 for 4K; OMP/OpenBLAS threads = 1 |

Each context loads one real Ouro-1.4B model and computes a real Triton prefill
checkpoint. Prompt KV is copied into independently allocated pages for every
request. All A/B trials restore that same checkpoint: this deliberately isolates
the **decode attention backend**, excluding prefill backend differences.
The six scheduling/backend combinations are shuffled within every repetition.
Both backends reserve the same physical KV pool for a context, sized for its
maximum concurrency. Source snapshots and SHA256 hashes are saved with the run.

No profiler is enabled during throughput measurement. GPU telemetry is logged
separately. First-use kernel compilation is excluded through full-length warmup.
The script synchronizes the old runner before switching the cache's attention
callable; it then creates and warms the new runner before its measured trial.

Output identity is checked separately within each backend and across backends.
Within-backend scheduling/repeat differences fail the run. Cross-backend output
differences are reported, since BF16 accumulation differences can change greedy
tokens even for numerically correct attention. Controlled exit depths keep the
number of recurrent rows comparable; any output difference must still be
reported with the timings, not described as exact equivalence.

This is a homogeneous-prompt, decode-only engine benchmark, not HTTP serving
throughput or an optimized prefill benchmark. FA4 prefill throughput is not
measured here. FlashAttention currently uses one query row per visible prefix
in our adapter, including chunked prefill; this is not a packed-prefill kernel
integration. The experiment also does not reproduce the paper's H100/FA3/graphs
configuration or establish GPU idle percentages.

## Reproduce

Use a new output directory and an available GPU:

```bash
cd /home/zjy/code/david/b_workspace/vllm-lt
CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  taskset -c 40-43 ../.b_rdma/bin/python -m benchmarks.context_sweep \
  --model ../models/Ouro-1.4B --context 1024 --concurrencies 1 32 128 \
  --output-length 128 --max-num-batched-tokens 2048 --block-size 16 \
  --repeats 3 --mixed-at-max --compare-attention-backends \
  --output /path/to/new/output \
  > /path/to/existing/log-directory/attention-ab.log 2>&1
```

`--compare-attention-backends` requires CUDA and uses Triton prefill followed by
paired Triton/FlashAttention decode. The normal `--attention-backend` option is
still available for single-backend sweeps. The resolved package/version is
recorded in `manifest.json` and every measured trial.

## Validation before measurement

The short real-model A/B protocol smoke completed all six backend/scheduling
combinations without output or depth differences. The six CPU prefix-protocol
regressions also passed. The underlying FA4 integration previously passed the
full suite (272 passed, 11 optional-dependency skips) and 12 real-model smoke
configurations. These checks establish the measurement setup, not its speedup.

## Completed results

All 144 measured trials completed successfully. Results below are ratios of three-run throughput medians (FA4 / Triton), with identical settings within each pair.

| Context | Concurrency | Exit policy | S | AS | AM |
| --- | ---: | --- | ---: | ---: | ---: |
| 1K | 1 | fixed4 | 0.821x (-17.9%) | 0.825x (-17.5%) | 0.828x (-17.2%) |
| 1K | 32 | fixed4 | 1.004x (+0.4%) | 0.936x (-6.4%) | 0.930x (-7.0%) |
| 1K | 128 | fixed4 | 1.417x (+41.7%) | 1.282x (+28.2%) | 1.305x (+30.5%) |
| 1K | 128 | mixed234 | 1.323x (+32.3%) | 1.191x (+19.1%) | 1.254x (+25.4%) |
| 4K | 1 | fixed4 | 1.420x (+42.0%) | 1.383x (+38.3%) | 1.377x (+37.7%) |
| 4K | 32 | fixed4 | 1.960x (+96.0%) | 1.904x (+90.4%) | 1.888x (+88.8%) |
| 4K | 64 | fixed4 | 1.737x (+73.7%) | 1.810x (+81.0%) | 1.815x (+81.5%) |
| 4K | 64 | mixed234 | 1.758x (+75.8%) | 1.810x (+81.0%) | 1.828x (+82.8%) |

Measured output tokens: 1,036,800; measured requests: 8,100. Within-backend disagreement trials: 0; cross-backend disagreement trials: 0.

FA4 is slower at 1K/C1 and does not improve the 1K/C32 asynchronous cases. It improves the larger-batch and 4K cases. These are engine-level results including CPU invocation and scheduling, not isolated kernel speedups. The CPU/GPU overlap counters do not establish an idle-time percentage or prove the cause of a regression.

For fixed4 at 4K/C64, FA4 AS/AM are approximately 10% faster than FA4 S. Multistream benefits are workload-dependent; AM should not be assumed faster than AS.

Detailed throughput and p95 inter-token latency:

| Context | C | Policy | Mode | Triton tok/s | FA4 tok/s | FA4 / Triton | Triton ITL p95 ms | FA4 ITL p95 ms |
| ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1024 | 1 | fixed4 | S | 36.07 | 29.61 | 0.821x | 28.16 | 34.32 |
| 1024 | 1 | fixed4 | AS | 35.91 | 29.64 | 0.825x | 34.52 | 42.30 |
| 1024 | 1 | fixed4 | AM | 35.67 | 29.54 | 0.828x | 35.13 | 34.94 |
| 1024 | 32 | fixed4 | S | 869.77 | 873.40 | 1.004x | 37.89 | 37.14 |
| 1024 | 32 | fixed4 | AS | 932.14 | 872.33 | 0.936x | 35.35 | 37.47 |
| 1024 | 32 | fixed4 | AM | 928.32 | 863.80 | 0.930x | 35.54 | 37.62 |
| 1024 | 128 | fixed4 | S | 1869.51 | 2648.58 | 1.417x | 71.61 | 50.53 |
| 1024 | 128 | fixed4 | AS | 2037.65 | 2611.79 | 1.282x | 66.36 | 51.30 |
| 1024 | 128 | fixed4 | AM | 2039.98 | 2662.00 | 1.305x | 65.03 | 49.46 |
| 1024 | 128 | mixed234 | S | 2073.71 | 2744.49 | 1.323x | 73.87 | 53.65 |
| 1024 | 128 | mixed234 | AS | 2280.69 | 2716.45 | 1.191x | 77.91 | 65.80 |
| 1024 | 128 | mixed234 | AM | 2141.58 | 2684.51 | 1.254x | 73.83 | 67.05 |
| 4096 | 1 | fixed4 | S | 19.07 | 27.08 | 1.420x | 53.32 | 37.44 |
| 4096 | 1 | fixed4 | AS | 19.55 | 27.03 | 1.383x | 52.03 | 46.10 |
| 4096 | 1 | fixed4 | AM | 19.57 | 26.96 | 1.377x | 51.95 | 46.68 |
| 4096 | 32 | fixed4 | S | 399.30 | 782.83 | 1.960x | 81.54 | 41.64 |
| 4096 | 32 | fixed4 | AS | 417.70 | 795.29 | 1.904x | 77.90 | 40.78 |
| 4096 | 32 | fixed4 | AM | 416.92 | 786.94 | 1.888x | 78.08 | 41.39 |
| 4096 | 64 | fixed4 | S | 623.30 | 1082.58 | 1.737x | 104.42 | 59.92 |
| 4096 | 64 | fixed4 | AS | 658.47 | 1191.55 | 1.810x | 98.96 | 54.54 |
| 4096 | 64 | fixed4 | AM | 658.50 | 1195.11 | 1.815x | 98.92 | 54.32 |
| 4096 | 64 | mixed234 | S | 691.89 | 1216.16 | 1.758x | 106.47 | 62.85 |
| 4096 | 64 | mixed234 | AS | 732.74 | 1326.15 | 1.810x | 123.19 | 66.30 |
| 4096 | 64 | mixed234 | AM | 669.12 | 1223.14 | 1.828x | 110.23 | 61.67 |

Raw results, commands, source hashes, GPU telemetry and output IDs: `/home/zjy/code/david/tmp/ouro-flash-attn-20260915/ab`.

No default runtime configuration was changed on the basis of these measurements.
