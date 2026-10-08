# Re:Loop W5 canonical workload inputs

The existing canonical tokenizer resolved all 820 original prompt records,
containing 919,552 token IDs. The original seven cases, four concurrency windows
per trial, arm order, five warmups and five measurements remain unchanged.
Parameters and KV are FP32, P4/R4, block16, with a common 48GiB KV pool.

| Case | B | C | Prompt records | First-B target KV (GiB) | GPU result |
|---|---:|---:|---:|---:|---|
| L1 | 1 | 1 | 4 | 0.9375 | Not run |
| T32 | 32 | 32 | 128 | 30 | Not run |
| Q64 | 32 | 64 | 256 | 66 | Queued case still required; observe actual A/S |
| T64_short | 64 | 64 | 256 | 36 | Not run |
| C8_long | 8 | 8 | 32 | 25.5 | Not run |
| C4_longer | 4 | 4 | 16 | 48.75 | Planned capacity exceeds the common pool |
| A32_large | 32 | 32 | 128 | 102 | Planned capacity exceeds the common pool |

Static reservations exclude model parameters, activations, scratch and guidance
references. Q64 remains in the original queue experiment even though its
first-B target exceeds the pool. The two planned capacity failures remain in
the denominator. No case was shortened, dropped or replaced.

Independent review checked all 173 archived payload files against tar and local
bytes, all 11 original packet inputs, 133 source Git blobs and six tokenizer
metadata files. All 820 records passed exact length, vocabulary integer, common
seed prefix, original ordering and KV geometry checks. This review verifies the
actual canonical tokenizer provenance; it does not independently re-tokenize
locally. CUDA remained uninitialized, checkpoint weights were not read, and
the original child, controller, transport, archive and release waits completed
naturally with zero. Actual resource handoff was verified for the known actors.

This closes CPU input preparation. The original GPU matrix still declares
42 fresh processes, 420 trials, 49,200 requests and 6,297,600 output tokens.
There are no W5 GPU speed, quality, HTTP or memory-saving results in this
receipt; whole-task pass contribution remains zero. The completed native
initial-C32 result is a separate experiment:
[speed comparison](H100_C32_FP32_RESULT-v1.md).

[Scientific settings and immutable hashes](W5_CANONICAL_INPUTS-v1.json).
