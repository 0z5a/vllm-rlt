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
278,169,600 stored parameters. The initial shape audit used a range request. The complete 556,342,528-byte
checkpoint is now locally hash-verified as
`077fae449dd3af29d0313412ef50036e1d9a37fb8b11d447fbb41da0462d9185`.
Official-weight CPU arithmetic checks below retain failures; CUDA serving remains
pending.

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
| Official-weight CPU author/dense/cached comparison | 12 diagnostic cases complete; frozen numerical budgets fail |
| Official-weight CUDA E2E | Pending |

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


## Official-weight CPU diagnostic

The complete released weights are loaded strictly into the native model and the
unchanged pinned author GPT. The author model shares the loaded weight values;
forward hooks observe each attention projection without replacing its arithmetic.
The independent dense oracle compares all logits, all 24 layer/depth KV pairs and
every native recurrent state. Inputs are actual tokenizer-produced IDs from the
frozen P128 fixture, truncated to P1/7/33 with two additional teacher-forced IDs.
No free-running generation or task quality is measured.

Two native executions are compared: cached P+1+1 execution, and a single full
prefill of all P+2 tokens. Both retain the prior FP32 `3e-6 / 3e-5` and BF16
`.03 / .02` absolute/relative budgets. **Neither dtype passes the complete gate**
on these official weights; the tiny-model result must not be generalized.

| Dtype | P + continuation | Author max logit error | Cached native max logit error | Full-prefill native max logit error | Cached / full native greedy IDs | Gate | Speedup |
|---|---|---:|---:|---:|---|---|---|
| float32 | 1 + 2 | 2.67029e-05 | 6.58035e-05 | 6.77109e-05 | True / True | Fail | Not measured |
| float32 | 7 + 2 | 2.95639e-05 | 6.29425e-05 | 6.29425e-05 | True / True | Fail | Not measured |
| float32 | 33 + 2 | 3.14713e-05 | 6.77109e-05 | 6.77109e-05 | True / True | Fail | Not measured |
| bfloat16 | 1 + 2 | 0.125 | 0.125 | 0.125 | True / True | Fail | Not measured |
| bfloat16 | 7 + 2 | 0.1875 | 0.21875 | 0.21875 | True / True | Fail | Not measured |
| bfloat16 | 33 + 2 | 0.488281 | 0.1875 | 0.1875 | False / False | Fail | Not measured |

All author argmax IDs match the dense oracle. For BF16 P33, the native path first
differs at absolute position 30: the oracle/author selects token 257 and native
selects 4572, with a 0.0625 top-two logit margin in each path. The same divergence
and BF16 error counts occur in the full-prefill diagnostic. Therefore incremental
decode alone does not explain this discrepancy. The arithmetic diagnostics below isolate attention and conditioning-row effects
on these fixtures. Tolerances have not been widened.

The cached run's maximum all-KV errors are 1.11e-4 in FP32 and 0.25 in BF16;
the author-vs-dense maxima are 3.24e-5 and 0.703125 respectively. The raw report
includes failed-element counts, first differing values and logical locations,
per-depth state errors and argmax margins. The cached numerical measurements
were repeated exactly before adding the full-prefill diagnostic.

```bash
python -m experiments.loopkv.loopformer_checkpoint_gate \
  --model /path/to/verified-loopformer \
  --author-source /path/to/pinned/modeling_loopformer.py \
  --prompts /path/to/loopformer-prompts-p128-v1.json \
  --out checkpoint-cached.json
# Repeat with --prefill-only for the full-prefill diagnostic.
```

Both commands naturally return one after retaining all six cases. They require
an existing compatible Transformers runtime for the author source and do not
install packages. CPU PyTorch 2.13.0 was used.
[Official checkpoint raw results and source hashes](evidence/loopformer-official-cpu.json).
These findings leave author-equivalence and GPU/quality qualification open;
they do not establish a zero-copy storage regression or an end-to-end speedup.


## Attention and conditioning arithmetic isolation

Four diagnostic settings each run the complete official weights on CPU with
all P+2 inputs in one prefill. In addition to logits, the harness now compares
every layer/depth K and V and every recurrent hidden state directly with the
author forward. Exactness is checked through tensor bytes, including signed zero.
The unmodified author setting reproduces all earlier full-prefill measurements.

The attention substitution runs the author's Q/K/V through native CPU paged
arithmetic. The conditioning diagnostic evaluates the author's time/dt and
AdaLN modules with the same repeated row count as native, then selects the first
identical row for the author's broadcast. Neither diagnostic changes the pinned
source file or the production adapter; the changes exist only inside the test
context. The conditioning option requires `--prefill-only`.

| Author diagnostic | Attention substitutions | Conditioning-row expansions | FP32 logits + all KV + states byte-exact | BF16 logits + all KV + states byte-exact | Max FP32 / BF16 native-author logit error | Speedup |
|---|---:|---:|---|---|---|---|
| Unmodified | 0 | 0 | 0/3 | 0/3 | 6.866455e-05 / 0.5351562 | Not measured |
| Native attention only | 144 | 0 | 0/3 | 3/3 | 6.67572e-05 / 0 | Not measured |
| Matched conditioning rows only | 0 | 240 | 0/3 | 0/3 | 4.1008e-05 / 0.5351562 | Not measured |
| Both substitutions | 144 | 240 | 3/3 | 3/3 | 0 / 0 | Not measured |

The original B1-versus-flattened-row FP32 conditioning check observes a maximum
embedding difference of 4.00543e-5 and modulation difference of 3.43323e-5. BF16
conditioning differences are zero for these shapes. Attention alone therefore
reconciles BF16; both substitutions reconcile FP32 as well. This supports an
arithmetic explanation for the observed official-weight differences on these
fixtures, not an original-backend equivalence claim for other inputs or devices.

Reproduce with the official command above plus `--prefill-only`; add
`--shared-attention-diagnostic`, `--matched-conditioning-diagnostic`, or both for
the respective row. The independent dense-oracle budgets remain unchanged and
still fail, so all four processes naturally return one after recording all six
cases. Diagnostic agreement does not turn those original gate failures into passes.
[All four settings, tensor-byte checks and process receipts](evidence/loopformer-arithmetic-isolation.json).
