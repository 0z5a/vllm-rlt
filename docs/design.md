# vllm-lt engine design

The first version provides a vLLM-style inference API for Ouro-1.4B. Its scheduling unit is a single traversal of the model's shared transformer layers. Request state and KV allocation are owned by this engine rather than delegated to a conventional token-level scheduler.

## Components and ownership

```text
LLM facade
    -> engine: request lifecycle, tokenization, output collection
        -> scheduler: admission and stage selection
        -> model runner: execute a selected batch
            -> native Ouro: embedding, recurrent core, output head
            -> attention backend: PyTorch reference or Triton
        -> KV manager: reservations, block tables, release
```

The scheduler owns queue membership, request progress, and loop depth. The runner owns tensor execution and the hidden states that survive between stages. The KV manager owns physical block allocation. Model code receives the positions and cache addresses required by a batch; it does not admit requests or decide which request executes next.

Each request has at most one token undergoing autoregressive decoding. Different requests may contribute tokens at different positions and loop depths to the same recurrent batch. Their data remains separate through per-request positions, block tables, and lengths.

## Request lifecycle

Waiting requests enter prefill after obtaining a complete KV reservation. Prefill embeds a prompt chunk and executes its complete recurrent depth before moving to another chunk. Causal attention uses the earlier chunks' depth-specific KV. After the last chunk, the final prompt hidden state enters coda and produces the first generated token.

If generation continues, the sampled token enters prelude for embedding, then recurrent execution. After each recurrent pass, the gate either returns that token to recurrent work or routes its final hidden state to coda. Coda computes logits and samples the following token. A completed or cancelled request releases its reservation and persistent state.

This prefill policy deliberately keeps every prompt token at full depth. It is an engineering extension described separately from the paper in [the source notes](paper-notes.md). It avoids both reprocessing the last prompt token and treating a generated token as already present in the cache before its forward pass.

## Scheduling modes

Refill scheduling allows newly prepared decode tokens to join continuing recurrent tokens. A prelude queued by coda runs immediately to prepare the sampled token for re-entry. Otherwise ready coda work runs first, followed by eligible prompt admission/prefill and recurrent work. Thus a request beginning loop one can execute alongside a different request beginning loop four.

No-refill holds a decode cohort. Tokens leave its recurrent batch when they exit, but its coda waits until the remaining cohort finishes. The next cohort starts after that boundary. Both modes use the same model, gate, and cache semantics.

Scheduling is synchronous: the engine reads results before deciding the next batch. Refill describes queue behavior, not asynchronous host/device overlap. There is no lookahead gate or CUDA-graph execution in this version.

## Gate and model contract

Ouro's embedding is the prelude. One recurrent operation runs all physical transformer layers and the shared end-of-loop normalization. The normalized hidden state persists for the next loop. Coda applies the LM head to the exited state.

Decode maintains a cumulative exit probability per token. Every loop updates it, including loops before the minimum allowed exit depth. A token exits when the cumulative threshold is reached and the minimum depth is satisfied, or when it reaches the maximum depth. Default adaptive bounds are two through four loops. Threshold `1.0` selects fixed depth explicitly. Token position and its RoPE phase stay unchanged while loop depth advances.

## Last-exited paged KV

Keys and values have separate physical tensors, each laid out as:

```text
[physical_block, physical_layer, token_offset, kv_head, head_dimension]
```

A request's block table maps `(loop_depth, logical_token_block)` to a physical block. There is no assumption that a recurrent batch shares one loop depth. Attention selects the depth-specific block-table row for each token and reads only its request's populated context, including the current token after its KV write.

When a token exits at depth `d`, the manager fills its slots at each skipped deeper depth with that token's computed depth-`d` keys and values. This operation copies each physical layer's KV independently. It does not copy the final hidden vector, neighboring tokens, or the whole partially populated block. Shallower computed depths remain intact. Future tokens can consequently attend at any supported depth without encountering missing entries.

The initial allocator reserves enough blocks for every input position the request can execute, at every depth. The last sampled output token needs no forward pass or KV entry. Required physical blocks are:

```text
model.total_ut_steps * ceil((prompt_tokens + max_tokens - 1) / block_size)
```

The depth factor is the model's full prefill depth, even when a request sets a lower decode loop limit. Admission waits until this full reservation fits. A request that cannot fit even in an empty cache must fail with an actionable capacity error. An admitted request can then reach completion without requesting additional blocks; this prevents admission from consuming space needed to finish existing requests. The cost is conservative memory use. Prefix sharing, eviction, swapping, and incremental reservation are outside this version.

Unused reserved positions are not valid KV. Attention lengths and block metadata must exclude them. Release occurs only after task-owned execution no longer references a request's blocks; reusing a block must not expose a previous request's contents.

For the released BF16 Ouro configuration, each token at one depth requires `2 * 24 * 16 * 128 * 2 = 196,608` bytes of KV. Four depths require 768 KiB per token before block rounding. Cache capacity must be explicit rather than inferred from model parameter size.

## Validation and next steps

BF16 weights, ordinary activations, and KV storage are the primary inference target. Accumulation precision is specified independently: retain FP32 in sensitive reductions, normalization statistics, and positional/probability arithmetic as needed. Full-model FP32 is an optional reference/diagnostic path. See the [precision policy](precision-policy.md) for the operation contracts and implementation follow-up.

The validation boundary is numerical and behavioral correctness. Fixed-depth outputs are compared with an independent reference under justified BF16 error bounds. Adaptive execution is compared with a serial implementation with the same last-exited policy. Exact cache/data-movement invariants are separate from floating-point and decision/quality criteria. Scheduler tests should exercise mixed depths, requests completing at different times, exhausted admission capacity, and cancellation. Cache tests should cover partial blocks, copying skipped depths, isolation, and block reuse.

PyTorch provides a portable attention reference. The historical [preview validation](validation.md) and [Q1 results](benchmarks/q1-20260911.md) retain their FP32 passes and BF16 failures. The [M2 result](benchmarks/m2-20260911.md) establishes a bounded FP32 metadata speedup; BF16 performance remains to be measured. Padded batch rows still need explicit handling before CUDA-graph execution is added.

The [roadmap](roadmap.md) prioritizes BF16 measurements alongside numerical and quality evaluation, with profiling selecting runtime, CUDA-graph, attention, and KV work. Asynchronous depth routing additionally requires an available and calibrated lookahead gate; it must not silently reinterpret the released Ouro gate. Serving integrations follow the single-model runtime milestones.
