# Official checkpoint CPU engine controls

The complete official LoopFormer and Parcae checkpoints each finish all 16 finite engine arms
with identical request objects and work across native, alias, compact and
compact-with-credits storage. These are BF16 CPU controls using torch attention,
two threads, P8/D16 and all eight recurrent steps. They cover real greedy
generation, a 16-token block boundary, queued requests and complete cache drain.

| Model | B / C | Storage arms | Completed requests | Output tokens | Actual peak residents | Requests / work | Formal speedup |
|---|---|---:|---:|---:|---:|---|---|
| LoopFormer | 1 / 2 | 4 | 8 | 128 | 1 | Exact | Not measured |
| LoopFormer | 4 / 8 | 4 | 32 | 512 | 4 | Exact | Not measured |
| LoopFormer | 16 / 32 | 4 | 128 | 2,048 | 16 | Exact | Not measured |
| LoopFormer | 32 / 64 | 4 | 256 | 4,096 | 32 | Exact | Not measured |
| Parcae | 1 / 2 | 4 | 8 | 128 | 1 | Exact | Not measured |
| Parcae | 4 / 8 | 4 | 32 | 512 | 4 | Exact | Not measured |
| Parcae | 16 / 32 | 4 | 128 | 2,048 | 16 | Exact | Not measured |
| Parcae | 32 / 64 | 4 | 256 | 4,096 | 32 | Exact | Not measured |
| Total | — | 32 | 848 | 13,568 | — | Exact | Not measured |

The audit checks every prompt, token, exit depth, finish reason, request ID,
recurrent batch histogram, per-depth row count and prefill count. All allocated
blocks return after drain; compact live and reserved records are zero. B32 has an
effective recurrent batch of 31.8672 because prefill admission is split; its
actual peak residency is 32. All four storage arms have the same work records.
CPU GPU-memory fields are explicitly null.

The model is `armenjeddi/LoopFormer-3block-8iterations` at
`2b4fbaaf4e2510353ef7cfe07a07c671b5226739`. All eight checkpoint assets were
independently hashed before loading. The run uses source
`9a94ef5823d620c909d9f9c00112b171b6a9363f`; its 13 affected capture/HTTP/text-stop
checks pass. Child 78844 naturally returned zero. The original two-model
controller returned one because its subsequent Parcae run failed admission.
All 102 raw JSON files across both models are hashed in the
[result and provenance record](evidence/official-cpu-engine.json).

This fixed-depth control contains no skipped-depth savings. Shared CPU timings
remain diagnostic fields, with no performance ratio. It does not qualify the
unchanged author arithmetic, GPU execution, CUDA Graphs, HTTP serving or task
quality. The original [LoopFormer](loopformer.md) and [Parcae](parcae.md) numerical failures remain.

## Parcae failures and corrected plans

The first official CPU screen naturally returned one before generating any
request: P8/D16 needs 16 KV blocks per resident request, while its plan supplied
four at B1. The failed plan, log and child return code are preserved. The corrected
16-arm plan completes at source `3c1dbdbaf26f51fffcbf0ee86c050de643fb6325`: child
82387 and controller 82380 naturally return zero. All 424 requests and 6,784
output tokens match across storage modes, with complete work equality and cache
recovery. This uses the original global worker RNG reset to seed 83 for each arm;
it does not assert equivalence between different batch schedules. The checkpoint
is `SandyResearch/parcae-370m@439284464ee4999bd1f762da7d044613a4828efe`.

An independent cache-geometry audit also corrects the unexecuted Parcae GPU plan.
At P128/D128 and eight depths, each request needs 128 rectangular blocks.
Each block occupies 786,432 bytes, so the 22 GiB payload ceiling admits 30,037
blocks and at most 234 full-length residents. The former 281-resident estimate
was wrong. The original LoopFormer and Huginn capacity calculations check out;
LoopFormer's unexecuted prompt-hash field was separately corrected. Versioned
plan corrections are local and have not been deployed or executed.

The unchanged Parcae author cache constructor also fails on the existing
Transformers 4.54.1 runtime: `ParcaeDynamicCache` assigns to the read-only
`key_cache` property. The saved isolated child returned one before model loading.
Earlier author prefill comparisons did not instantiate this cache and therefore
do not establish cached-generation compatibility. No environment or author
source was changed.

A separate opt-in [cache API diagnostic](parcae-cached-diagnostic.md) now exercises
the author's inherited cache algorithm through writable dictionary properties.
Official cached logits, states and KV match bitwise after attention arithmetic
is shared. Keeping author attention retains numerical failures. This does not
change the original constructor failure or qualify the unchanged author runtime.

To reproduce an engine screen, use the recorded model/prompt hashes and plan:

```bash
python -m experiments.loopkv.screen --plan frozen-plan.json \
  --out new-output-directory --load-lock task-owned-load.lock
```

The plan explicitly selects `device: cpu`, `attention_backend: torch` and
`cpu_threads: 2`. Omitting these fields preserves the existing CUDA/Triton
defaults. Each run requires its recorded source commit and an unchanged checkout.

The prompt-hash guard was also exercised with a deliberately mismatched frozen
hash: it returns one before loading a model and leaves the output directory
empty. The original failed Parcae admission and constructor probes remain in the
result record alongside the successful corrected generation runs.
