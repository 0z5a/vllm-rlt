# FlashLoop weight API qualification

The native Ouro loader can supply a separately loaded model to the unchanged
[FlashLoop engine](https://github.com/Superone77/FlashLoop/tree/aeaadee10d75b7aa8c4a42a809f15c75d6d3bdc6).
The adapter exposes the attention metadata and activation callable that engine
uses, and removes the extra head axis from native RoPE outputs. It consumes the
model instance: FlashLoop may fuse projections and release their original
weights. Use a separate instance for a native baseline.

This is an **adapted FlashLoop baseline**, pinned to
`aeaadee10d75b7aa8c4a42a809f15c75d6d3bdc6`. The default author loader requires a
Transformers version unavailable on the target runtime. The adapter does not
replace the engine, codec or reader. No environment changes are required for the
native weight loader; the author comparison below additionally needs an existing
compatible Transformers runtime.

## CPU weight-interface gate

Both paths execute the unchanged FlashLoop engine: one receives the pinned HF
Ouro model; the other receives the adapted native Ouro model. The test copies
identical seeded tiny weights with strict state-dict checking and preserves the
original FP32 RoPE frequencies on both paths. It uses BF16, 24 layers, hidden
width 128, one 128-wide attention head, vocabulary 128 and four loops. Every case
compares prefill plus two teacher-forced decode steps, including logits and every
layer/depth KV, with zero absolute and relative tolerance.

| Engine configuration | Prompt lengths | Logits and all KV | Packed streams at P129 | Speedup |
|---|---|---|---:|---|
| Dense BF16 control | 1, 65, 129 | Bitwise exact, 3/3 | 0 | Not measured |
| Sparse BF16 | 1, 65, 129 | Bitwise exact, 3/3 | 0 | Not measured |
| Default packed int4 | 1, 65, 129 | Bitwise exact, 3/3 | 192 | Not measured |

The P1/P65 cases retain their uncompressed residual tails. P129 exercises actual
int4 packing. CPU execution does **not** qualify the CUDA packed reader or measure
GPU speed. [Raw nine-case results](evidence/flashloop-author-cpu.json).

## Native arithmetic comparison

A separate dense-control comparison uses native paged attention as the reference,
with sparsity and quantization disabled in FlashLoop. It retains the established
FP32 `atol=3e-6, rtol=3e-5` and BF16 `atol=.03, rtol=.02` budgets.

| Dtype | P / D | Max logit error | Max KV error | Failed logit / KV elements | Result | Speedup |
|---|---|---:|---:|---:|---|---|
| FP32 | 1 / 2 | 5.44e-7 | 6.85e-7 | 0 / 0 | Pass | Not measured |
| FP32 | 7 / 2 | 2.02e-6 | 3.49e-6 | 0 / 0 | Pass | Not measured |
| FP32 | 65 / 2 | 1.58e-6 | 4.25e-6 | 0 / 0 | Pass | Not measured |
| BF16 | 1 / 2 | 0 | 0 | 0 / 0 | Pass | Not measured |
| BF16 | 7 / 2 | .0390625 | .0439453 | 1 / 209 | **Fail** | Not measured |
| BF16 | 65 / 2 | .0227051 | .0546875 | 0 / 2130 | **Fail** | Not measured |

All six tiny greedy outputs match. The BF16 failures remain failures; the
weight-interface result above does not establish native arithmetic equivalence.
[Raw six-case results](evidence/flashloop-native-cpu.json).

## Reproduction

Use the pinned FlashLoop checkout with the supplied
[source manifest](evidence/flashloop-source-manifest.json) saved as its
`SOURCE_MANIFEST.json`. Both commands verify that manifest and every listed source
file before importing the engine. Obtain `configuration_ouro.py` and
`modeling_ouro.py` from
[ByteDance/Ouro-1.4B at 574fa66](https://huggingface.co/ByteDance/Ouro-1.4B/tree/574fa66cb8bf5abdc979642d01cf2b79b16bfab1)
in one directory; the author gate checks both file hashes.

```bash
python -m experiments.loopkv.flashloop_author_gate \
  --flashloop-repo /path/to/FlashLoop \
  --author-model-dir /path/to/pinned-ouro-source \
  --out author-interface.json
python -m experiments.loopkv.flashloop_native_gate \
  --flashloop-repo /path/to/FlashLoop \
  --out native-arithmetic.json
```

The first command returns zero. The second records all six cases and returns one
for the retained BF16 discrepancies. These runs used CPU PyTorch 2.13.0 and the
existing Transformers 4.54.1 runtime. They used tiny random weights, not the
released model weights. Official-checkpoint quality, CUDA execution, direct
packed-reader qualification and end-to-end speed remain pending. The pinned
optimized engine supports batch size one and four loops; this interface gate
cannot satisfy the campaign's high-concurrency model matrix.
