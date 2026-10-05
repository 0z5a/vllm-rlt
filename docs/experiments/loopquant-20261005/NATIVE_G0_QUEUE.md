# Full-checkpoint native G0 queue

Full-checkpoint CUDA qualification and formal quantized E2E remain incomplete.
The latest H20 attempt and prepared replacements are recorded below.

## LoopFormer download attempt and replacement

The admitted fixed LoopFormer G0v2 packet stops during its first configuration request:
the public mirror redirects then returns HTTP403. Download child71532, controller71525
and launcher71524 finish naturally1; SSH80692 returns1. No checkpoint file is written,
and all64 full-checkpoint cache cases remain **NOT_RUN**.

[All12 raw files](h20-loopformer-g0-v2/archive-index.json) are independently hash-verified offbox.
The retained archive is10577 bytes, SHA256
`6389f3a3cafde246cf5d24bfc492fad4e3d7dd61a6022dcb02a96a65172d69dc`.
[Final audit](h20-loopformer-g0-v2/complete.json) confirms every own process absent,
original boot/UUID/GPU0→IO inodes unchanged and compute empty. The empty own model directory
is removed, and the whole window is explicitly handed back. An initial archive-count assertion
(expected13/actual12) is retained; it made no archive mutation.

The new **LOCALREADY_NOT_ADMITTED** G0v3 packet replaces the mirror with an owner-authorized,
read-only tar stream of8 complete local assets, totaling561155685 bytes. It creates no local
weight duplicate. The receiver holds the original GPU0→IO locks through transfer and the
same64 GPU checks, validates every file size/SHA, and runs the unchanged probe after transfer.
Peer source files remain read-only. Both the probe/requests and frozen runtime source
`8045c32f49a86cfc3fc103195f52057444ec2c59` match v2 byte-for-byte; thresholds stay.03/.02.

G0v3 manifest SHA256:
`067c7e866c640d882c55366f95b5b6a74e7d41939ea6ea02e62f245c831e1f7d`, packet4260602 bytes.
[Packet and local-client receipt](h20-loopformer-g0-v3/h20-loopformer-stream-ready.json),
[controller](h20-loopformer-g0-v3/run.py), [read-only client](h20-loopformer-g0-v3/transfer-client.py).
The stream→standalone meta-loader→tiny full-R8 dense comparison exits naturally0:
all8 states and logits are bitwise exact and no model storage remains meta.
[Preflight](h20-loopformer-g0-v3/loopformer-stream-loader-preflight-attempt1.json).
Actual peer readers, transfer and CUDA checks have not started. This needs a fresh independent grant.

## Ouro-2.6B and Nanbeige preparation

Both new v2 packets remain **LOCALREADY_NOT_ADMITTED**, with no H20 upload, download or remote run.
Each downloads one fixed revision, verifies all inputs, then runs six independent children:
Torch/Triton native cache × eager/automatic-SDPA/math-SDPA reference.
Every child uses the same32 prompts with full prefill and token-incremental cache,
compares all loop states/readouts, preserves prior KV and checks zero KV after drain.
All six probes run even if an earlier probe fails; a failed download stops dependent probes.
Combined budgets stay `atol=0.02, rtol=0.02`.

Existing official CPU bitwise evidence qualifies **eager** attention. Automatic/math SDPA
are separate internal controls and do not replace the original reference.
The earlier unsubmitted four-arm v1 packets are superseded; their generic scope text
incorrectly labeled dense SDPA CPU-qualified. Those frozen artifacts remain retained as provenance,
and must not be deployed. Neither v2 packet proves original-author CUDA attention, quantized
quality, speed, memory reduction or formal E2E. LF's observed mirror403 is also retained;
future download readiness must be verified before claiming deployed checkpoints.

| Checkpoint | Prompts / input tokens | Loop readouts | Planned cache cases | Model bytes | Manifest SHA256 |
|---|---:|---|---:|---:|---|
| ByteDance/Ouro-2.6B | 32 / 1724 | 1,2,3,4 | 384 | 5,340,852,869 | `d69e8bc013094716ab3cf77440428738d045252c862db234769f79aee89e1358` |
| Nanbeige/Nanbeige4.2-3B | 32 / 1756 | 1,2 | 384 | 8,361,039,496 | `f7e1e53fc6590990c9a00e569bedc0e7e2b07485f6acacf6b7c1e50c9caf838a` |

Ouro-2.6B revision: `1ed04250da1a9936042725d302e81c8fa2ab5abd`.
Nanbeige revision: `b82e54bd609793562a75cbf9337970a93369eab5`.
Both runtime archives are from `0a73ba23f4441cdac50b181e4841fba296b61218`;
[all packet files](fixed-loop-g0-v2/ready.json) pass independent SHA/AST checks.

The tiny preparation controller finishes all four direct calls and writes its complete summary;
its tool handle becomes unavailable during a desktop transition, and its PID is independently absent.
The table reports the recorded probe-function return codes, without inferring a missing tool exit.
[Raw control](fixed-loop-g0-v2/tiny-summary.json), [completion audit](fixed-loop-g0-v2/completion.json).

| Tiny architecture | Reference | Cache cases | State comparisons | States bitwise | Largest state/logit absolute difference | Probe return | Argmax / prior KV |
|---|---|---:|---:|---|---:|---:|---|
| Nanbeige | eager | 4 | 28 | No | 0.03125 | 0 | Exact / exact |
| Nanbeige | math SDPA | 4 | 28 | Yes | 0 | 0 | Exact / exact |
| Ouro | eager | 4 | 56 | No | 0.0390625 | 1 | Different / exact |
| Ouro | math SDPA | 4 | 56 | Yes | 0 | 0 | Exact / exact |

All four arms pass the combined state/logit budget; **Ouro eager remains inconclusive because argmax differs**.
The math controls do not resolve that failure. Earlier [automatic-SDPA discrepancies](RESULTS.md#ouro-sdpa-reference-control-on-tiny-cpu-weights)
remain. The previously completed [v1 tiny Nano automatic control](fixed-loop-g0-v1/tiny-summary.json)
passes the combined budget with non-bitwise states, maximum state/logit error.01953125.
No tiny diagnostic is a full-checkpoint CUDA G0.

The latest directly verified shared-resource priority places ready FlashNS/LoopCD work first.
All Quant retry/follow-up packets wait for later separate grants. Original locked quality,
all-model high-concurrency/multi-batch paired E2E, RTN/GPTQ, QAT and NVFP4 remain incomplete;
old5090 INT4 child31380 remains UNKNOWN and is never duplicated or killed.
