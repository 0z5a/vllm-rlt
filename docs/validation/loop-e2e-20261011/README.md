# Loop E2E supplements — 2026-10-11

This update records completed, independently audited LoopCD and LoopGraph stages. ReLoop's native matrix and five ScaleRLT scaling pairs have completed; HTTP export and the final independent receipts are still being checked. These are stage results, not a claim that every requested E2E case is complete. This branch changes documentation and result metadata only.

## LoopCD: complete FP32 numerical evidence

The unchanged `e8aa1ee57734bc5c1118c143ebdb68d0b53210a1` engine and Ouro checkpoint revision `574fa66cb8bf5abdc979642d01cf2b79b16bfab1` passed the public-entry and complete W2 supplements on H100. The independent CPU auditor streamed every archived payload and recomputed all recorded element comparisons, with FP32 values promoted to binary64, original `atol = rtol = 1e-4`, and byte equality for exact parameter pairs.

| Stage | Recorded tensors/pairs | Elements checked | Coverage |
| --- | ---: | ---: | --- |
| Public entry | 293 | 1,435,254,791 | 269 exact parameter pairs, 18 numerical comparisons, 6 full-vocabulary log-softmax comparisons |
| W2 | 167,196 | 350,773,356 | 12 cases, full state/logits/log-softmax/selected log probabilities, 82,944 key and 82,944 value comparisons, readout and lifecycle records |

Abort, refill, stale-owner rejection and request-ID reuse were included. All 227 archived payloads passed SHA-256 verification; producer and independent auditor returned zero. Detailed counts and content hashes are in [loopcd-full-tensors.json](loopcd-full-tensors.json). The unchanged original category/lifecycle auditor was checked by AST equality.

The original high-concurrency protocol declares a 64 GiB KV pool, which does not fit on the available H100 partition with model weights. Its status remains **capacity-limited, not run**. A separately named 48 GiB variant preserves the original 512-token inputs, 128-token outputs, B16/C32 and B32/C64 cases, three strategy arms, two fresh orders, and five warmups plus five measurements. That variant passed the raw-trial audits below and cannot substitute for completion of the original 64 GiB specification. Earlier exploratory quality uncertainty is unchanged.

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

## LoopGraph: gate boundaries and actual long prompts

The unchanged CDB cumulative-hazard gate passed an additional real-CUDA boundary oracle: 38 cases, four signal dtypes, eager and captured CUDA Graph execution, and two slot-reuse waves. The independent CPU auditor reconstructed all 4,096 recorded slot-state observations. Exact threshold equality and the adjacent binary64 values, minimum/maximum depth, padding, and stable addresses are covered. This primitive test uses an exactly representable hazard of 0.5; it does not replace the separate real-model math/G3 checks.

The following additional candidate runs use repeated canonical token sequences, 128 generated tokens per request, two request-ID reuse waves, and both fixed-depth-4 and immediate live-gate profiles:

| Prompt length | Capacity | Common KV cap | Candidate requests | Generated tokens | Independently audited result |
| --- | ---: | ---: | ---: | ---: | --- |
| 256 | 32 | 48 GiB | 128 | 16,384 | Exact author tokens, aggregate exit-depth/core-work parity, complete drain |
| 1,024 | 32 | 48 GiB | 128 | 16,384 | Exact author tokens, aggregate exit-depth/core-work parity, complete drain |
| 1,024 | 64 | 48 GiB | 0 | 0 | Candidate admission stopped before generation: its preallocated KV bank exceeds the cap |
| 1,024 | 64 | 56 GiB | 256 | 32,768 | Separately declared variant passed exact token/work/drain checks and independent audit |

The original author completed the B64 reference, but the combined B64 comparison remains a capacity stop. Its producer returned 1 and the initial whole-run auditor rejected it. A separate partial-result audit verified the completed B32 cases and confirmed the two B64 admission failures; it does not turn the failed whole run into a pass. The B32/1,024-token run is a separately named capacity variant. The separately declared 56 GiB B64 functional variant passed; the original 48 GiB failure remains retained.

These are full-generation functional cases, not frozen ShareGPT/Poisson workloads, matched-physical-allocation performance comparisons, or full hidden/KV exports. No throughput or quality gain is claimed. Counts, scopes, natural return codes and archive hashes are in [loopgraph-long-context.json](loopgraph-long-context.json).

A further heterogeneous backlog test used four times the active capacity, mixed input lengths and output lengths 1/3/17/33/64/128. Fixed-depth-4 passed at B32 and B64 with all power-of-two branch buckets exercised. Live-gate runs retained exact generated tokens but executed 16 and 35 extra recurrent steps relative to the author. The producer and initial whole-run audit failed. Independent reconstruction confirmed both the completed cases and these mismatches; it is not an overall pass. The live-gate discrepancy is under diagnosis. See [loopgraph-backlog.json](loopgraph-backlog.json).

## LoopGraph: W1 replay diagnostics

Using the original author's frozen synthetic prompt/depth builders (seed 1729), P256/output128 and capacity 32, both engines allocated exactly 9,764,339,712 physical KV bytes. All-full and all-shallow replay passed two waves of 64 requests each. Bimodal replay retained exact per-request depth plans and recurrent work, but 9 of 64 requests diverged in generated tokens; that producer failed.

All nine divergent requests then passed individually for two waves. Batching those same requests reproduced token differences. A separate prefill-batch diagnostic also found four request differences between the author's own two waves. These results expose batch/scheduling sensitivity; they do not yet establish its numerical or implementation cause. The native engine, gate threshold and comparison tolerances were unchanged. Independent CPU audits verified the complete archives and reconstructed every recorded pass and failure. All four author execution-mode combinations (synchronous/asynchronous, eager/CUDA Graph) passed their own two-wave replay and reproduced 8–9 native token divergences. Raw hidden/logit snapshots at first divergence were independently recomputed on CPU: several recurrent states already differ, and several top-token pairs are tied or separated by one BF16 increment. An additional isolated projection check reused six saved author/native hidden vectors at three divergence points. All 36 B1/B2/B4/B8/B16/B32 projection observations were byte-identical to B1 and the original captured logits. A separate CPU auditor verified every archived payload and recomputed the summaries from raw tensors, matching the inputs to the previously verified first-divergence archive. This narrows the observed difference to computation before the projection for those inputs; it does not identify the earlier operation. Two further routing diagnostics reused the exact verified author output: host-selected ordinary body execution and host-selected captured branches both reproduced the same nine token divergences. Each route retained exact depth/work and drain checks; all 432 saved native hidden/logit records were byte-identical to the prior conditional-router records. An independent CPU auditor checked all 428 archive payloads, reference provenance and raw tensors. This excludes routing execution as the source of the observed saved-state difference in these diagnostics. A separately declared fixed-capacity padding diagnostic also retained the same nine token divergences: 420 of 432 saved hidden/logit records were unchanged, while 12 differed (maximum hidden/logit absolute differences 0.15625/0.1875). Its 410 archive payloads and full raw results passed independent audit; the exact-token case remains failed. The computation producing the state difference is still being located. No W1 speed claim is made. See [loopgraph-w1-diagnostics.json](loopgraph-w1-diagnostics.json).

A further read-only prefill observer captured 2,696 module-input/output records for requests20/58 across four depths and 24 layers. Every compared last-prompt-row input/output, and both first-output hidden/logit vectors, was byte-identical. All 423 previously saved author records and 432 native records were also unchanged; the same nine exact-token failures remain. The complete 420-payload archive passed independent CPU integrity and raw-tensor comparison. This narrows the observed saved-state difference to later decoding for these requests, but does not establish equality of every cached prompt position or identify an operation. The initial observer-interface failure is retained separately. See [loopgraph-prefill-observer.json](loopgraph-prefill-observer.json).

The extended observer independently verified 14,154 prefill/decode module records for the same two requests. The first observed differences were both at layer0 MLP `down_proj`: request20 at output index2 had three differing output elements (maximum 3.814697265625e-6), and request58 at index4 had two (maximum 6.103515625e-5), despite byte-identical input vectors. The generated tokens first differ later, at indices6/7. All previous saved head records and the nine token failures were reproduced. This locates a specific module boundary; an isolated matrix-shape check now reproduces both originally captured outputs exactly for each of these two vectors. See [loopgraph-core-observer.json](loopgraph-core-observer.json).

The isolated down-projection check used the exact saved inputs and unchanged BF16 checkpoint weights. For request20, B1/B2/B4/B8/B12 reproduced the author output and B16/B32 reproduced the native output. Request58 showed the same split with B15 matching the author. A CPU auditor verified all 165 payloads and all 14 saved observations against the earlier core archive. A further 28 observations for requests14/36/39/51 passed independent audit against the all-nine core capture, giving six saved inputs and 42 observations in total. The same B1-to-small-batch versus B16/B32 split exactly reproduces all six captured initial differences. The all-nine observer covered prefill and four decoding outputs: requests7/13/63 showed no difference within that window and are being observed farther. See [loopgraph-core-observer-all9.json](loopgraph-core-observer-all9.json). Remaining requests and operations stay under diagnosis; no exact-token or performance gate has been relaxed. See [loopgraph-down-proj-shape.json](loopgraph-down-proj-shape.json).

## LoopCD: completed native cost variants

Both H10048 cost runs passed independent raw-trial audits: each includes 12 fresh arm processes, 120 trials, 5,760 requests and 737,280 generated tokens. The original 64 GiB configuration remains capacity-limited. The second run isolates CPU cores by case; the first run retains its CPU-contention caveat. Each arm keeps full-depth-4 prefill and fixed 512-token inputs/128-token outputs.

| Case | Depth 4, CD off (tok/s) | Depth 3, CD off (tok/s) | Depth 3, CD on (tok/s) | CD on vs off at depth 3 |
| --- | ---: | ---: | ---: | ---: |
| B16/C32 | 178.84 | 209.31 | 206.94 | −1.13% |
| B32/C64 | 246.21 | 277.68 | 274.16 | −1.27% |

Relative to the depth-4 baseline, depth-3 plus CD is 15.71%/11.35% faster in these instrumented runs; that includes depth reduction and is not a pure CD gain. Candidate quality remains uncertain. All warmup and measured trials, both fresh arm orders and exact-source qualification checks are retained. See [loopcd-cost.json](loopcd-cost.json).

## ScaleRLT: completed numerical training stages

The frozen FP32 controlled-reward GRPO, PPO, DPPO and Flow-DPPO runs each passed three continuous updates and the matching save/fresh-resume path. Rollouts, selected log probabilities, recurrent traces, publication versions and optimizer gradients match the original numerical checks. Final complete checkpoint comparisons cover 202 typed actor chunks for GRPO, and 202 actor plus 202 critic chunks for PPO/DPPO/Flow-DPPO, together with optimizer, scheduler, iteration/version and saved RNG state.

| Algorithm | Response tokens per path | Largest selected-logprob absolute error | Validation status |
| --- | ---: | ---: | --- |
| GRPO | 1,394 | 0.0001022816 | Local numerical/checkpoint audit passed; independent streaming verification of all 58 archived payloads passed |
| PPO | 352 | 0.0001038313 | Local numerical/checkpoint audit passed; independent verification of 80 archived paths / 67 unique payloads passed |
| DPPO | 352 | 0.0001038313 | Local numerical/checkpoint audit passed; archive export pending |
| Flow-DPPO | 352 | 0.0001038313 | Local numerical/checkpoint and categorical/KL checks passed; durable full archive and independent 79-path / 66-payload integrity passed |

Flow-DPPO also passed full old-policy categorical normalization, selected-logprob and KL-metric checks at a 0.01 budget, with two optimizer steps per rollout and exact continuous/resume agreement. No actual blocked step occurred in the 0.01-budget run. A separate 1e-8-budget supplement passed the unchanged full categorical audit and observed blocked fractions of 19.375%, 13.636% and 19.129% on the three second optimizer steps. This establishes actual gate activation under the tight diagnostic budget; independent streaming verification of its 59 logical paths and 55 complete payloads passed; both Flow archives are now durable and passed a separate complete integrity audit on a second node; original RAM copies remain preserved.

The independent GRPO export check verifies all archived bytes and hashes; the typed numerical checkpoint comparison was performed by the local auditor. These controlled-fixture checks establish E2E consistency, not held-out reward improvement or a scaling gain. The DPPO/Flow collectors initially missed the CPU auditor scripts through a path substitution. A separately verified supplementary archive supplies all original auditor scripts and 629 frozen source files; complete checkpoint data and numerical outputs were retained. See [scalerlt-training.json](scalerlt-training.json).

Early concurrent jobs shared a narrow Slurm CPU affinity. Their timings remain diagnostic. Subsequent timing runs use explicit CPU sets; the new one-node baseline and two-node pilot also share publication and checkpoint storage policy. Completed and interrupted attempts are retained. The repeated LoopCD cost matrix with separate CPU sets is complete.

The first matched-storage scaling pair (seed 42, 32 global samples per update) passed local numerical checks with identical response-token counts. After two warmup updates, all five measured updates average **108.17 s on one node/two GPUs** and **121.34 s on two nodes/three GPUs**: a **10.86% throughput regression** in this preliminary pair. The slow 211.89 s two-node update is retained. The second pair (seed 43) passed the same numerical checks and identical response-token counts, averaging **117.82 s** versus **100.67 s**, or **17.03% higher throughput** on two nodes. The opposite signs and co-running I/O leave the scaling conclusion unresolved. The third pair (seed44) also passed, averaging **121.62 s** versus **101.02 s**, or **20.40% higher throughput**. The fourth pair (seed45) passed with identical response-token counts, averaging **114.84 s** versus **95.38 s**, or **20.41% higher throughput**. The fifth pair (seed46) passed with identical response-token counts, averaging **129.74 s** versus **96.05 s**, or **35.08% higher throughput**. All eight complete archives for seeds42/43/44/45 passed independent payload checks; seed46 baseline integrity passed and its two-node independent receipt remains pending; this is not a held-out reward or stable scaling claim. See [scalerlt-scaling-pilot.json](scalerlt-scaling-pilot.json).

## ReLoop: completed native matrix

The five runnable cases completed 30 fresh arm processes, 300 raw trials and 150 measured trials, using the original frozen token inputs and 128 generated tokens per request. The producer and independent raw-trial auditor returned zero. All original trials, both fresh orders and the two capacity stops are retained. At the same depth, guidance changes throughput by +9.27% (T32), +2.37% (L1), −3.49% (T64_short), −0.29% (C8_long) and −0.73% (Q64). Timings used the original narrow CPU affinity and co-running tasks; these are diagnostic results, not a general performance claim.

The initial independent source check rejected three archived files because the packet manifest referenced a later source version. All three actual files were independently matched byte-for-byte to the qualified native revision; every remaining original hash check and all 30 raw-trial audits passed. The original integrity rejections remain recorded. See [reloop-native-matrix.json](reloop-native-matrix.json).

## Remaining work

- ReLoop: verify the completed HTTP export and repeat selected native cases with isolated CPU cores. The original native matrix is independently audited. The small token-ID HTTP input support is on `codex/reloop-http-tokenids-20261011` (`e4f504872975563b8f31e91f02996add158cf8c8`). Capacity-exceeded cases stay explicitly separate.
- ScaleRLT: finish the remaining independent exports and repeated scaling pairs. The matched-storage single-node baseline and two-node actor-DP pilot completed seven updates and local numerical checks; all five seeds completed local numerical checks, with the last independent archive receipt pending. No node-scaling improvement is established yet.
- The attempted two-rollout, four-GPU ScaleRLT configuration was rejected before training: the pinned native backend explicitly supports one rollout GPU/engine. The next supported comparison keeps one rollout engine and increases training data parallelism across nodes. No validator is bypassed.
- Scaling measurements must hold checkpoint, dtype, depth, samples, reward fixture, seed and publication storage fixed. Record actual role placement, full-update wall time, response tokens, training/rollout/publication time, and GPU efficiency. Retain failures and regressions; one pilot does not establish improvement. Five matched-storage pairs are complete; final independent raw-data verification and isolated confirmation remain in progress.

PR #82's accepted coda-only numbers remain +0.34% / +0.85%; they are not combined with any of these experiments. PR #115 was separately updated with real CUDA cleanup/recovery coverage and 88 passing targeted tests, with no skips.
