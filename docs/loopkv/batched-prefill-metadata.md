# Batched prefill metadata

The opt-in eager path prepares one metadata set for all recurrent prefill depths. With 32 prompt rows and eight depths, native/alias constructs 6,784 metadata bytes instead of 11,264, and compact constructs 7,936 instead of 13,312. Both call the staging helper twice instead of sixteen times. These are CPU construction counts; CUDA traffic, peak allocation and E2E speed have not been measured for this change.

Frozen implementation: `aa60941d6238072e05a9a571615f1e2726907670`. Results and artifact hashes: [batched-prefill-metadata.json](evidence/batched-prefill-metadata.json).

## Execution contract

`ExecutionConfig(prefill_batch_metadata=True)` defaults to false. It requires LAST_EXITED with Torch or Triton attention and rejects static buffers, CUDA Graphs and prefill UVA. Request pointers, positions, context lengths and map widths share storage within one traversal. Depth-specific write addresses and query depths retain separate views. Compact storage reserves the existing per-depth records before staging; the existing per-depth map publications remain.

The traversal preserves HRM prefix read lengths, Huginn/Parcae boundary KV, the original gate computation policy and stale-allocation checks. It adds no cross-step metadata cache or quantized scale representation. Preparing every depth together keeps more metadata live until prefill completes; fewer constructed bytes do not imply a smaller peak allocation or payload pool.

## Validation

The existing local Python 3.12 / Torch 2.13.0 environment was used without installation or upgrades. The complete CPU lane returned zero: **601 passed, 29 skipped, 119 GPU tests deselected, and 36 subtests passed**. The focused lane returned zero with 121 passed and three GPU tests deselected. Existing tests exercise cancellation, immediate request-ID reuse, delayed coda, prefix attention and recurrent state across storage modes.

Six seeded tiny checkpoints ran through the existing `experiments.loopkv.screen` entrypoint at BF16, prompt lengths 1/7/17, four generated tokens, B=1/4/16/32 and C=2B. Each family compares native/alias/compact/credits with the option off/on. All six children and their controller naturally returned zero. The independent audit compares complete request objects to the first native arm, recomputes work from raw schedules, checks recurrent depth/row histograms and validates cache drain. These small random checkpoints qualify architecture paths; they are not official-weight, unchanged-author or quality results.

| Family | Generation arms | Requests | Output tokens | Output and work comparison |
|---|---:|---:|---:|---|
| Ouro | 32 | 848 | 3,392 | Exact |
| Nanbeige | 32 | 848 | 3,392 | Exact |
| Huginn | 32 | 848 | 3,392 | Exact |
| HRM Text | 32 | 848 | 3,392 | Exact |
| LoopFormer | 32 | 848 | 3,392 | Exact |
| Parcae | 32 | 848 | 3,392 | Exact |
| Total | 192 | 5,088 | 20,352 | Exact |

Separate official BF16 LoopFormer/Parcae CPU runs at P8/D16/R8, B=1/4/16/32 and C=2B are still running. Their planned 64 arms are excluded from completed counts. That controller's source checkout remains frozen. Shared-machine CPU timings are diagnostic and are not used for speed claims.

## Metadata and speed comparison

The actual prefill entrypoint was instrumented with FP32 tiny Ouro, four requests × eight prompt tokens, R=1/4/8 and native/alias/compact storage. All 18 hidden-state and used-KV hash comparisons pass, and all cache allocations drain.

| R | Storage | Stage calls, off → on | Constructed metadata bytes, off → on | Byte reduction |
|---:|---|---:|---:|---:|
| 1 | Native / alias | 2 → 2 | 1,408 → 1,408 | 0.00% |
| 1 | Compact | 2 → 2 | 1,664 → 1,664 | 0.00% |
| 4 | Native / alias | 8 → 2 | 5,632 → 3,712 | 34.09% |
| 4 | Compact | 8 → 2 | 6,656 → 4,352 | 34.62% |
| 8 | Native / alias | 16 → 2 | 11,264 → 6,784 | 39.77% |
| 8 | Compact | 16 → 2 | 13,312 → 7,936 | 40.38% |

For this compact fixture, n=32: the old total is 52nR bytes and the candidate is n(24+28R) bytes. Two tensor stages remain for R=1. These counts exclude record-map publication and device payload operations; they are not profiler H2D or HBM counters.

The initial instrumentation assertion failed because it included CPU reference-read staging in preparation: 66 calls instead of two for the alias R1 fixture. The corrected phase counter separates those operations. CPU alias/compact reference reads still stage 64R times; those calls are not reported as savings. Numerical hashes passed before the counting fix, and a separately bound recorder reproduces the corrected report byte-for-byte.

| Speed / memory comparison | Baseline | Candidate | Improvement |
|---|---|---|---|
| CUDA engine E2E throughput | Pending | Pending | Not established |
| HTTP E2E throughput / latency | Pending | Pending | Not established |
| CUDA peak allocated / reserved bytes | Pending | Pending | Not established |

A 5,993,146-byte H20 qualification packet is prepared locally and registered behind the shared-node queue. It contains these six seeded architecture screens plus the 18-case prefill audit; it has not been uploaded or executed. Environment qualification and whole-window handback remain prerequisites. Performance follow-up requires matched official weights and scheduling, repeated balanced baseline/candidate runs, realized batch sizes, profiler traffic and peak-allocation measurements before considering default enablement.
