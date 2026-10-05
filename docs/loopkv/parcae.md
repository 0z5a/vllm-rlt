# Parcae native execution

The adapter preserves the published diagonal state injection, prelude/core/coda
boundaries, per-depth core KV, gated token-value embeddings and half-scaled
ReLU-squared MLP. The prelude and final output each have their own normalization;
a separate C projection precedes the coda. Q/K normalization follows interleaved
RoPE and retains PyTorch's dtype-dependent default epsilon.

Two byte-valued hidden channels carry token IDs between stages without BF16
rounding. The remaining state contains the recurrent state and fixed prelude
injection. Initialization preserves the author's random-normal draw followed by
truncated-normal initialization; it does not replace stochastic states with zero.
This uses the worker RNG and depends on batch ordering. Tests reset the RNG for
matched cohorts; request-independent stochastic initialization is not implemented.
The official screening driver resets the frozen `state_seed` before each storage
arm and records it in every summary. Different batch schedules still receive
different random draw assignments and must be reported separately.

| Published 370m configuration | Value |
|---|---:|
| Prelude / core / coda modules | 4 / 4 / 4 |
| Fixed recurrent steps | 8 |
| Hidden width / resident width | 1,024 / 2,050 |
| Q and KV heads | 8 |
| Vocabulary / context | 32,768 / 2,048 |
| Author/native constructor state entries | 117, identical names and shapes |
| Unique parameters | 388,003,328 |

The source contract is pinned to
[sandyresearch/parcae](https://github.com/sandyresearch/parcae/tree/69284c13746e849104f738d6d1a347b1f457df76).
The checkpoint is `SandyResearch/parcae-370m@439284464ee4999bd1f762da7d044613a4828efe`.
The complete 1,553,099,635-byte official checkpoint is now locally verified as
`603d9da4a1c1a112c8b6a98bc1e9aac288990ba0d7f5b432aaad9c53940bfcb2`
and executed in the CPU diagnostic below. This does not establish CUDA E2E
qualification. The native loader uses
`torch.load(weights_only=True)`, strict tensor matching and tied-head verification.
It recognizes the author's `_class_name` configuration without requiring an
invented `model_type` in the published file.

| CPU validation | Result |
|---|---|
| Dense FP32 state/logits/all KV, full and chunked causal prompts | Eight cases pass across native/alias/compact/credits at atol 3e-6, rtol 3e-5 |
| BF16 token-ID channels and original random-state draw sequence | Exact |
| Ragged full engine, four storage modes and two ID lifetimes | Outputs/exits match; all reservations returned |
| Strict checkpoint and standalone/Auto dispatch | Exact reload; missing tensor rejected |
| Affected CPU regressions | 227 passed, 5 GPU tests deselected, 36 subtests passed |

The unchanged author generation forward was also compared using copied tiny
weights and the same FP32 RoPE table in all paths. FP32 passes the original
atol 3e-6 / rtol 3e-5 budget. **BF16 does not pass** the separately declared
atol 0.03 / rtol 0.02 budget at P3 and P7; tolerances remain unchanged. The
comparison program writes all six cases and returns 1 for these failures.

| Dtype | P | Native vs author maximum absolute logit error | Native logit check | Dense logit check | Greedy tokens |
|---|---:|---:|---|---|---|
| FP32 | 1 | 0 | Pass | Pass | Exact |
| FP32 | 3 | 0.000004292 | Pass | Pass | Exact |
| FP32 | 7 | 0.000006676 | Pass | Pass | Exact |
| BF16 | 1 | 0 | Pass | Pass | Exact |
| BF16 | 3 | 0.1484375 | Fail | Fail | Exact |
| BF16 | 7 | 0.1875 | Fail | Fail | Exact |

[Raw author comparison and constructor audit](evidence/parcae-cpu.json).
Matching greedy tokens on these tiny inputs does not establish numerical or
quality equivalence. A separate diagnostic replaces only the author's attention
call with the native CPU paged arithmetic. All six native logit comparisons then
have zero maximum logit error, including BF16 P3/P7; 60 substituted attention calls are
recorded. This isolates the observed native discrepancy to the attention path
for these fixtures. It does not qualify the unchanged author backend or CUDA.
The independent dense path still exceeds the unchanged budget at FP32 P7 and
BF16 P3/P7 in this diagnostic, and the program retains return code 1.
[Raw attention diagnostic](evidence/parcae-attention-diagnostic.json).

Reproduce the original comparison with
`python -m experiments.loopkv.parcae_author_gate --author-repo <pinned-checkout> --official-config <config.json> --out <original.json>`.
Add `--shared-attention-diagnostic` and use a separate output path for the
arithmetic isolation. Neither command changes the pinned author checkout.

| Official E2E comparison | Native tokens/s | Candidate tokens/s | Speedup |
|---|---:|---:|---:|
| Parcae-370m | — | — | Not measured |

Only synchronous full-depth LAST_EXITED with torch/Triton is admitted. Async,
prefix reuse, preemption, speculation and prefill UVA are not qualified. Tiny
CUDA, official-checkpoint GPU E2E, broader context/concurrency, task quality and
formal performance remain pending.


## Released checkpoint CPU diagnostic

The optional `--model` and `--prompts` arguments load the complete pinned
checkpoint and fixed tokenizer-produced IDs. The author model is constructed on
meta and strictly assigned the same tensors, preserving tied weights and FP32
RoPE. Both paths reset the same RNG and verify the initial recurrent state
exactly. Existing tiny measurements, including the 60-call attention diagnostic,
reproduce unchanged after this harness extension.

| Dtype | P | Native vs author max logit error | Dense vs author max logit error | Native / dense budget | Native greedy IDs | Speedup |
|---|---:|---:|---:|---|---|---|
| float32 | 1 | 0 | 0 | True / True | Exact | Not measured |
| float32 | 7 | 3.242493e-05 | 4.196167e-05 | False / False | Exact | Not measured |
| float32 | 33 | 5.054474e-05 | 5.626678e-05 | False / False | Exact | Not measured |
| bfloat16 | 1 | 0 | 0 | True / True | Exact | Not measured |
| bfloat16 | 7 | 0.4375 | 2 | False / False | Different | Not measured |
| bfloat16 | 33 | 0.75 | 3.5 | False / False | Different | Not measured |

FP32 keeps `atol=3e-6, rtol=3e-5`; BF16 keeps `atol=.03, rtol=.02`.
P1 passes both paths. **P7/P33 fail the original budgets in both dtypes.**
BF16 first differs at position 3: author token 1761 versus native token 403.
At P7 both top-two margins are 0.0625; at P33 they are 0.0625 and 0 respectively.
These are teacher-forced argmax comparisons, not free-running task accuracy.

The separate attention diagnostic substitutes only the author's attention call
with the native CPU paged arithmetic. Across 240 observed substitutions, all six
native comparisons then have zero maximum logit error and matching argmax IDs.
This isolates the observed native discrepancy to the attention path on these
fixtures; it does not qualify the unchanged author backend. The independent dense
oracle still fails at P7/P33, so that diagnostic also retains return code 1.
No budget was widened.

```bash
python -m experiments.loopkv.parcae_author_gate \
  --author-repo /path/to/pinned-parcae \
  --official-config /path/to/pinned/config.json \
  --model /path/to/verified-parcae-370m \
  --prompts /path/to/parcae-prompts-p128-v1.json \
  --out official-author.json
# Repeat with --shared-attention-diagnostic and a separate output path.
```

Both commands naturally return one after recording all six cases.
[Official-weight raw reports, first argmax margins and source hash](evidence/parcae-official-cpu.json).
Author-equivalence, official CUDA E2E, broader quality and speed remain open.

[Official CPU engine controls and retained cache/admission failures](official-cpu-engine.md)
record the current generation checks and corrected pending GPU capacity.
