# Seven-model independent C128 input supplement

Independent review checked all 59 archive files/58 payload hashes and local
bytes, then ran the original complete input auditor naturally with zero.
It verified seven checkpoints, 39 traces and 39,936 request templates.
Each trace has 1024 requests. All 27 legacy trace prefixes retain their
original first-512 JSON bytes, calibration slices and arrival order.

| Input review | Actual result |
|---|---|
| Complete archive and original calibration/source hashes | Pass |
| Context bounds, IDs, lengths/depths and fixed-prompt uniqueness | Pass |
| Legacy input files and first-512 prefixes | Unchanged |
| Nanbeige default BOS metadata and six trace counts | Static conflict verified: 6140/6144 lack the declared prefix |
| Actual production tokenizer/HTTP roundtrip | Not run |
| Steady 60-second serving, SLO/rates/candidates and GPU tests | Not run |

The pinned Nanbeige tokenizer declares a default `<|im_start|>`/166100 prefix.
The remaining four rows are not established as roundtrip-safe. Actual production
encoding must retain every mismatch; BOS policy and original native banks must
not be changed to force equality. Any canonical-text/native-token counterpart
needs a distinct frozen case with model, tokenizer and source bindings.

This is Loopquant's independent supplement. It does not expand or replace
Re:Loop's original L1/W5/C128 requests, trials or denominators. Execution and G3
readiness remain false; actual B/core rows and live C must be observed.
No model weights were read and whole-task pass contribution is zero.

[Immutable hashes and reviewed counts](LOOPQUANT_C128_INPUT_REVIEW-v1.json).
