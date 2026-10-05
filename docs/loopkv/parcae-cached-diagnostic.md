# Parcae cached CPU arithmetic diagnostic

The released 370m checkpoint produces byte-identical native and author logits,
recurrent states and used KV entries when the author path uses the native CPU
attention arithmetic. This now covers cached execution: P1/P7/P33 followed by
two forced single-token decode steps, both FP32 and BF16, with all eight recurrent
steps. It is an explicit compatibility and arithmetic diagnostic, not unchanged
author qualification, free-running task quality, GPU execution or speed evidence.

The original `ParcaeDynamicCache` constructor fails on the existing Transformers
4.54.1 runtime because its `key_cache` and `value_cache` properties are read-only.
An opt-in subclass supplies writable dictionary properties. Its constructor,
update, reset and sequence-length methods are exactly the pinned author's methods;
their identities are asserted. The author checkout and runtime are unchanged.
The original constructor failure is retained in the result bundle.

| Dtype | Prefill + forced decode | Author attention: max logit error | Author attention: complete check | Shared attention: logits / states / KV | Formal speedup |
|---|---|---:|---|---|---|
| FP32 | 1 + 2 | 0.0000133514 | Fail | Bitwise exact | Not measured |
| FP32 | 7 + 2 | 0.0000324249 | Fail | Bitwise exact | Not measured |
| FP32 | 33 + 2 | 0.0000505447 | Fail | Bitwise exact | Not measured |
| BF16 | 1 + 2 | 0.1875 | Fail | Bitwise exact | Not measured |
| BF16 | 7 + 2 | 0.4375 | Fail | Bitwise exact | Not measured |
| BF16 | 33 + 2 | 0.75 | Fail | Bitwise exact | Not measured |

The complete check covers logits, recurrent states and KV at the existing
FP32 `atol=3e-6, rtol=3e-5` and BF16 `atol=.03, rtol=.02` budgets. No tolerance was
widened. Retaining author attention fails all six complete official cases and
naturally returns 1. BF16 P7/P33 also retain the earlier prefill argmax difference
at position 3. Both forced decode argmaxes match in these fixtures despite the
numerical failures; this does not establish free-running equivalence.

With shared attention, all 18 logit tensors, 144 recurrent-state comparisons and
1,440 key/value comparisons are byte-identical, including the cached prefixes
after each decode. There are 720 observed attention substitutions. Initial random
states match exactly using seed `109 + chunk_index` in both paths. All expected
author cache steps and token positions are checked; both caches drain completely.
The diagnostic naturally returns 0. Its parent controller also returns 0 after
recording both child outcomes.

Tiny cached controls independently pass all three FP32 cases but retain BF16
failures with author attention. Shared attention makes all six tiny cases exact
and records 180 substitutions. The existing tiny prefill comparison and attention
diagnostic reproduce all earlier case values and substitution counts unchanged.

The official runs use source `3595eb55571873668d586825d2752956f08a9f67`,
the pinned author `69284c13746e849104f738d6d1a347b1f457df76`, and checkpoint
`SandyResearch/parcae-370m@439284464ee4999bd1f762da7d044613a4828efe`.
Weights and frozen prompts are hashed before loading; the author model strictly
loads the same tensors and uses the same FP32 RoPE table. Full reports, runtime
versions, file hashes and natural child return codes are in the
[result bundle](evidence/parcae-cached-api-diagnostic.json).

Reproduce each mode with a distinct output file:

```bash
python -m experiments.loopkv.parcae_author_gate \
  --author-repo <pinned-checkout> --official-config <model>/config.json \
  --model <model> --prompts <frozen-prompts.json> \
  --cached-api-diagnostic --out <cached.json>
```

Add `--shared-attention-diagnostic` for the separate arithmetic isolation.
Neither mode qualifies the unchanged author cache API or CUDA backend. Official
GPU, high-concurrency HTTP and task-quality comparisons remain pending.
