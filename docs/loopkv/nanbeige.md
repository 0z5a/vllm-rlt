# Nanbeige4.2-3B official checkpoint

The complete official checkpoint passes 32 synchronous storage comparisons:
native, rectangular alias, compact records and compact reclaimed credits at
B1/4/16/32/64/128/256/512 with C=2B. Across all arms, **8,104 requests and
1,037,312 output tokens** complete. Prompt IDs, output IDs, exit depths, finish
reasons and complete request objects are identical at each batch point.
An independent audit also reconciles every raw step with the reported work;
all four storage modes execute the same recurrent batch/depth histograms and
prefill work. Every block credit and compact reservation returns after drain.

| B limit | Queued requests C | Peak resident requests | Mean recurrent B | Output tokens per arm | Largest reserved GiB across four arms | Four-storage parity |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 2 | 1 | 1.00 | 256 | 7.949 | Exact |
| 4 | 8 | 4 | 3.98 | 1,024 | 8.078 | Exact |
| 16 | 32 | 16 | 15.54 | 4,096 | 8.594 | Exact |
| 32 | 64 | 32 | 30.16 | 8,192 | 9.281 | Exact |
| 64 | 128 | 64 | 56.94 | 16,384 | 10.676 | Exact |
| 128 | 256 | 128 | 102.40 | 32,768 | 13.449 | Exact |
| 256 | 512 | 256 | 204.80 | 65,536 | 19.273 | Exact |
| 512 | 1024 | 512 | 409.60 | 131,072 | 30.742 | Exact |

The B512 point actually reaches 512 resident requests and mean recurrent batch
409.6; C1024 includes queued requests. Maximum allocator reserved memory is
33,009,172,480 bytes. These observations are specific to P128/D128 and a
22 GiB KV arena ceiling; they are not a general capacity limit for this model.

This is a full-depth negative control: fixed R2, threshold1, greedy seed17,
EOS ignored and four repeated tokenizer-pinned prompt fixtures. No skipped-depth
storage reduction or admission expansion is available at full depth. These are
not independent task-quality questions. Model loading is shared across the
four storage arms and the node concurrently ran another qualification, so
recorded elapsed seconds are diagnostic only.

| Official E2E comparison | Native tokens/s | Candidate tokens/s | Speedup |
|---|---:|---:|---:|
| Nanbeige4.2-3B, B1–512/C2B, four storage modes | — | — | Not measured |

The checkpoint is
`Nanbeige/Nanbeige4.2-3B@b82e54bd609793562a75cbf9337970a93369eab5`.
All ten runtime files, totaling 8,360,887,509 bytes, were independently matched
to the pinned source manifest before execution. The implementation is frozen at
`f2b6add3270a02b599f4d2da0e76445e0e8f946d`, on RTX5090 with
Torch2.12.1+cu130 and Triton attention. The controller and every worker naturally
returned0. All104 raw result files (97,224,593 bytes) were copied off the node
and verified by size and SHA256 before the audit.

[Frozen inputs, all32 summaries, raw-file hashes and completion receipt](evidence/nanbeige-official-sync.json).
Async/Graph, broader contexts, natural exits, task quality and independent-process
performance remain outstanding. The local duplicate weights were removed only
after their CPU work and readers finished; the verified node copy remains for
these required tests and another registered reader.
