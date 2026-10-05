# LoopFormer native causal execution

LoopFormer applies three shared blocks eight times, conditioning each application
on time `t` and step size `dt=1/8`. The adapter retains that clock alongside each
hidden row so continuous batching and resident routing preserve per-request time.
Learned position embeddings are added once, before the first block application.
Each recurrent depth has separate causal KV; the output projection uses the token
embedding directly. RMSNorm preserves the author's dtype-dependent default epsilon.

The public contract is pinned to
[`armenjeddi/LoopFormer-3block-8iterations@2b4fbaaf4e2510353ef7cfe07a07c671b5226739`](https://huggingface.co/armenjeddi/LoopFormer-3block-8iterations/blob/2b4fbaaf4e2510353ef7cfe07a07c671b5226739/modeling_loopformer.py).
Its configuration relies on the defaults defined in that source: hidden width
2,048, 32 heads, head width 64, FFN width 5,120 and context 1,024. All 29 tensor
names and shapes agree with the published safetensors header, totaling
278,169,600 stored parameters. The header came from a range request; this is not
full-weight verification or official inference.

| Validation | Result |
|---|---|
| Independent FP32 state, logits and all KV comparisons | Four storage modes pass at atol 3e-6 / rtol 3e-5 |
| Chunked causal prefill followed by incremental decoding | Pass at every recurrent depth |
| Ragged prompts, sync/async, request ID reuse | All eight mode/storage combinations match dense greedy decoding |
| Strict standard and standalone checkpoint loading | Exact tensor reload |
| Affected CPU suite | 228 passed, 27 GPU tests deselected, 36 subtests passed |
| Actual pinned author GPT forward, tiny copied FP32 parameters | Three lengths pass; max absolute logit error 9.54e-7 |
| Actual pinned author GPT forward, tiny copied BF16 parameters | Three lengths pass at atol 0.03 / rtol 0.02; max absolute error 0.0625; greedy tokens exact |
| CUDA FP32 state/logits/all-KV oracle | Pass at the same 3e-6 / 3e-5 tolerances |
| Tiny BF16 CUDA sync/async/Graph, four storage modes, B1–128/C2B | All 72 arms pass across two request-ID lifetimes |
| Official-weight inference | Pending |

The author comparison uses the original full dense forward against the independent
functional oracle. The paged implementation is separately checked against that
oracle. BF16 discrepancies remain in the raw results and are not claimed to be
bitwise equality. Reproduce the author comparison with
`python -m experiments.loopkv.loopformer_author_gate --author-source <pinned_file> --out <json>`;
the harness checks the source SHA256 before importing it. The CUDA gate is
`python -m experiments.loopkv.loopformer_gate --out <json>` and includes an FP32
oracle followed by 72 BF16 resident/cache arms.

Frozen source `4176b1bfc0a6ca0478cc4d459245f819d38e5e7a` passed that CUDA gate
on RTX5090 #0 with torch 2.12.1+cu130: 11,760 completed requests, 70,560 output
tokens, 164 Graph captures, 4,072 replays and zero fallbacks. Every storage variant
exactly matches native within its execution mode and returns its entire cache
reservation. The controller and children exited naturally with code 0; all eight
evidence files were verified off the node. Archive SHA256:
`484fbd376ba5fc977e4d502352f534e61313ab899c083875ef549a6c96de6c07`.
[Raw CUDA cases and completion receipt](evidence/loopformer-tiny-cuda.json).
The shared-node tiny run does not establish official-model quality or speed.

This checkpoint provides no exit gate. The admitted path requires all eight
steps, threshold 1, LAST_EXITED storage and torch/Triton attention. Prefix reuse,
preemption, speculative decoding and prefill UVA remain outside this qualification.
Fixed-depth execution is a negative control for skipped-depth storage savings.

| Official E2E comparison | Native tokens/s | Candidate tokens/s | Speedup |
|---|---:|---:|---:|
| LoopFormer 3-block / 8-step | — | — | Not measured |

[Raw author comparison](evidence/loopformer-author-tiny.json).
