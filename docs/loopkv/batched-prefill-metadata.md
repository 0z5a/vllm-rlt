# Batched prefill metadata

The opt-in eager path prepares one metadata set for all recurrent prefill depths. With 32 prompt rows and eight depths, native/alias constructs 6,784 metadata bytes instead of 11,264, and compact constructs 7,936 instead of 13,312. Both call the staging helper twice instead of sixteen times. The same construction counts were observed on CPU and NVIDIA H20. These are metadata source bytes, not measured H2D/HBM traffic; Official H20 E2E screening results are now available in the linked report. The H20 diagnostic shows a small increase in peak allocated memory.

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

The same six seeded checkpoints also passed **192 CUDA arms, 5,088 requests and 20,352 output tokens** on NVIDIA H20 (SM90), using Python 3.12.3, Torch 2.12.1+cu130 and Triton 3.7.1. The isolated environment inherits the installed packages; no packages were installed or upgraded. The generation and metadata children and controller all naturally returned zero. The offbox archive contains 705 files, all independently verified by SHA256; the independent generation audit also reconciles every request and raw work counter. [H20 evidence](evidence/batched-prefill-h20.json) records the plans, checkpoint hashes, allocation diagnostics and receipts.

Official BF16 LoopFormer/Parcae CPU runs at P8/D16/R8, B=1/4/16/32 and C=2B completed all **64 arms, 1,696 requests and 27,136 output tokens**. Both children returned zero. Independent audits found every complete request object and every work histogram exact against the first native arm and the previous official CPU golden runs. All cache blocks and compact records drained. [Raw official comparison](evidence/batched-prefill-official-cpu.json) includes plan/checkpoint hashes and child receipts. This is CPU correctness evidence, not CUDA, unchanged-author, quality or E2E speed qualification.

## Metadata and speed comparison

The actual prefill entrypoint was instrumented with FP32 tiny Ouro, four requests × eight prompt tokens, R=1/4/8 and native/alias/compact storage. All 18 hidden-state and used-KV hash comparisons pass on CPU, and all 18 pass again on H20; all cache allocations drain. These are exact comparisons within each device, not a cross-device equality claim.

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
| H20 official Thinking engine E2E | [Per-case results](h20-thinking-prefill-e2e.md) | Six balanced pairs per case | -1.37% to +0.59% |
| HTTP E2E throughput / latency | Pending | Pending | Not established |
| H20 peak allocated bytes, tiny B32 | See below | See below | Slight increase |
| H20 peak reserved bytes, tiny B32 | 36 or 54 MiB by family | Unchanged | 0% |

| Tiny family, B32/C64 | Native peak allocated, off → on (bytes) | Native delta | Compact peak allocated, off → on (bytes) | Compact delta |
|---|---:|---:|---:|---:|
| ouro | 35,811,840 → 35,817,984 | +6,144 | 35,895,808 → 35,906,560 | +10,752 |
| nanbeige | 35,808,768 → 35,810,816 | +2,048 | 35,892,736 → 35,896,320 | +3,584 |
| huginn | 42,257,920 → 42,262,016 | +4,096 | 42,343,936 → 42,351,104 | +7,168 |
| hrm | 39,976,448 → 39,979,520 | +3,072 | 40,060,416 → 40,065,024 | +4,608 |
| loopformer | 36,049,408 → 36,063,744 | +14,336 | 36,139,008 → 36,164,096 | +25,088 |
| parcae | 47,341,056 → 47,349,760 | +8,704 | 47,425,024 → 47,438,848 | +13,824 |

These allocation samples come from one ordered, traced correctness run per arm, with peak statistics reset after engine construction. They are allocation diagnostics rather than repeated performance trials or whole-process VRAM measurements. At B32, alias has the same allocation delta as native, and credits has the same delta as compact. Reserved bytes remain unchanged within each pair. Keeping all depths live explains the tradeoff: constructed metadata bytes fall while peak allocated bytes rise by 2,048–25,088 bytes in these tiny fixtures.

The H20 packet has completed, its raw archive is verified offbox, its six temporary weight payloads have been removed, and the shared GPU/IO window has been returned. The prepared environment and evidence remain. The first official H20 E2E follow-up below measures paired throughput and allocated memory. Broader contexts and models, fresh-process confirmation and profiler traffic remain necessary before considering default enablement.

## Official H20 E2E follow-up

[Official Thinking results](h20-thinking-prefill-e2e.md) record all 147 complete generation arms, exact outputs/work, paired throughput and observed allocated memory. The six paired medians range from -1.37% to +0.59%; 6/6 descriptive intervals include no change. This is fixed-R4 engine E2E screening in one resident model process. The broader official-model matrix, adaptive-depth serving, fresh-process confirmation and HTTP measurements remain unfinished.
