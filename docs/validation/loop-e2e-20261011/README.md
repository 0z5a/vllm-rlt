# Loop E2E supplements — 2026-10-11

This update records completed, independently audited LoopCD and LoopGraph stages. ReLoop's remaining native/HTTP matrix and ScaleRLT's training/scaling matrix are still in progress. These are stage results, not a claim that every requested E2E case is complete. This branch changes documentation and result metadata only.

## LoopCD: complete FP32 numerical evidence

The unchanged `e8aa1ee57734bc5c1118c143ebdb68d0b53210a1` engine and Ouro checkpoint revision `574fa66cb8bf5abdc979642d01cf2b79b16bfab1` passed the public-entry and complete W2 supplements on H100. The independent CPU auditor streamed every archived payload and recomputed all recorded element comparisons, with FP32 values promoted to binary64, original `atol = rtol = 1e-4`, and byte equality for exact parameter pairs.

| Stage | Recorded tensors/pairs | Elements checked | Coverage |
| --- | ---: | ---: | --- |
| Public entry | 293 | 1,435,254,791 | 269 exact parameter pairs, 18 numerical comparisons, 6 full-vocabulary log-softmax comparisons |
| W2 | 167,196 | 350,773,356 | 12 cases, full state/logits/log-softmax/selected log probabilities, 82,944 key and 82,944 value comparisons, readout and lifecycle records |

Abort, refill, stale-owner rejection and request-ID reuse were included. All 227 archived payloads passed SHA-256 verification; producer and independent auditor returned zero. Detailed counts and content hashes are in [loopcd-full-tensors.json](loopcd-full-tensors.json). The unchanged original category/lifecycle auditor was checked by AST equality.

The original high-concurrency protocol declares a 64 GiB KV pool, which does not fit on the available H100 partition with model weights. Its status remains **capacity-limited, not run**. A separately named 48 GiB variant preserves the original 512-token inputs, 128-token outputs, B16/C32 and B32/C64 cases, three strategy arms, two fresh orders, and five warmups plus five measurements. That variant is pending and cannot substitute for completion of the original 64 GiB specification. Earlier exploratory quality uncertainty is unchanged.

## LoopGraph: real serving and request reuse

The original author engine and current conditional-router composition use the same official Ouro checkpoint, BF16, FlashAttention 3, four recurrence/KV planes, `last_exited`, greedy feedback, and the original live-gate threshold. The original author source is unchanged. The native source differs only by the previously recorded FA3 import compatibility adapter; launch guards were rebound to the actual runtime and resources.

| Supplement | Profiles / configured capacity | Requests | Output tokens | Result |
| --- | --- | ---: | ---: | --- |
| Original-author comparison | fixed depth 4 and live gate, capacity 2, two reuse waves | 8 | 32 | Exact generated tokens and aggregate exit-depth/core-work parity |
| Async polling and refill | both profiles, capacity 2 | 512 | 1,280 | Exact author outputs or EOS/one-token prefixes; complete drain |
| Wider bucket qualification | fixed depth: capacity 32, context cap 384; live: capacity 64, context cap 128 | 768 | 1,920 | Exact author outputs or prefixes; complete drain |

The stress runs cover nonblocking event polling, submission while a readback is in flight, queued replacement work, repeated request IDs, duplicate-ID rejection, invalid input rejection, refusal to admit directly across an active reader, refusal to close a busy engine, and zero physical KV blocks after close. The two small-capacity profiles each reused IDs 248 times; the wider profiles reused them 192 and 384 times.

All three complete archives passed independent CPU audit, including payload hashes, 345 native/author source files, model metadata, actual child/parent return codes, generated tokens, timestamps, aggregate depth/core-work accounting, and exported drain observations. See [loopgraph-serving.json](loopgraph-serving.json).

These serving runs use the two original short prompts (5 and 15 tokens), with requested output lengths 3 and 5, plus explicit one-token/EOS cases. The configured context limits are not evidence of long-context execution. Full hidden-state/KV numerical parity is covered by separate math/G3 gates; this serving export does not contain full hidden/KV tensors. These co-running, instrumented runs do not establish a throughput gain, a quality improvement, or completion of the formal W1/W2 performance matrix.

## Remaining work

- ReLoop: complete the original frozen-token native and HTTP cost matrices. The small token-ID HTTP input support is on `codex/reloop-http-tokenids-20261011` (`e4f504872975563b8f31e91f02996add158cf8c8`). Capacity-exceeded cases stay explicitly separate.
- ScaleRLT: finish all continuous/save/fresh-resume checks and independent exports. The single-node, two-GPU GRPO pilot completed seven real updates with nonzero gradients and all local trace/checkpoint checks. It is not yet a scaling result.
- The attempted two-rollout, four-GPU ScaleRLT configuration was rejected before training: the pinned native backend explicitly supports one rollout GPU/engine. The next supported comparison keeps one rollout engine and increases training data parallelism across nodes. No validator is bypassed.
- Scaling measurements must hold checkpoint, dtype, depth, samples, reward fixture, seed and publication storage fixed. Record actual role placement, full-update wall time, response tokens, training/rollout/publication time, and GPU efficiency. Retain failures and regressions; one pilot does not establish improvement. Shared-publication timing and multi-node qualification are pending.

PR #82's accepted coda-only numbers remain +0.34% / +0.85%; they are not combined with any of these experiments. PR #115 was separately updated with real CUDA cleanup/recovery coverage and 88 passing targeted tests, with no skips.
