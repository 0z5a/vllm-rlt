# Ouro-1.4B: frozen GSM8K100 storage comparison

The corrected compact cache and hybrid rectangular alias produce the same
11,473 output tokens, exit depths and stop locations as Native-S on all 100
requests. Their serialized request SHA256 is identical:
`48bc270ae840e70c16372eb1b252201924398b336ab36a1e6c0974bfe96c24cc`.
The earlier compact failure remains part of the evidence.

| Arm | Strict correct / 100 | Exact requests vs Native-S | Token-identical requests | Missing strict answer | Hit output limit |
|---|---:|---:|---:|---:|---:|
| Native-S, original gate 0.5 | 64 | 100 | 100 | 24 | 1 |
| Hybrid rectangular alias | 64 | 100 | 100 | 24 | 1 |
| Compact before admission fix | 63 | 69 | 90 | 24 | 1 |
| Compact after admission fix | 64 | 100 | 100 | 24 | 1 |
| Native full 4 loops | 63 | Different exit policy | Not an equivalence target | 26 | 1 |

A one-answer difference on this small cohort does not establish that early exit
improves model quality. The storage comparisons hold the gate fixed; the full4
run is a separate policy anchor. All 11,508 cached decode positions in full4
execute exactly four loops. Natural-gate decode exits are 491 at loop2, 9,972 at
loop3 and 910 at loop4. This distribution differs from the controlled throughput
pilot and must not inherit that pilot's memory-saving estimate.

## Protocol

Official Ouro-1.4B revision `574fa66cb8bf5abdc979642d01cf2b79b16bfab1`, BF16,
Triton eager execution, greedy seed17, B16/C100, maximum1,024 output tokens,
natural EOS and text stops `Q:`, `</s>`, `<|im_end|>`. Text stops finish a logical
request through the engine API; the engine's underlying `abort` reason is retained
with the matched stop and byte offset. No process receives a termination signal.
Ninety-nine requests in each arm stop on text; one reaches the output limit.

The [GSM8K dataset](https://huggingface.co/datasets/openai/gsm8k/tree/740312add88f781978c0658806c59bc2815b9866)
is pinned at `740312add88f781978c0658806c59bc2815b9866`. Before inference, select
the 100 smallest SHA256 values of `revision:test:17:source_index` from all1,319 test
rows, then restore source-index order. Use the first three fixed examples from
the frozen `gsm8k_cot` v0.4.9.2 recipe and strict first-match extraction
`The answer is (\-?[0-9\.\,]+).`, with its number normalization rules.
This is a new native-engine cohort, not a reproduction of a historical HF score.

[Compressed frozen prompts, IDs, gold answers and recipe hash](evidence/quality-v1/gsm8k100-protocol.json.gz)
have decompressed SHA256 `632f572c373dd39ad42901eaae39156edf1d9f3490c17919cadfe36c12286ede`.
The standalone text-stop table was [qualified](evidence/quality-v1/byte-stops-qualification.json)
against all49,152 tokenizer entries and500 random token sequences using the
existing Tokenizers0.21.4 runtime. No remote dependencies were installed.

## Failure, correction and limits

Initial native, alias and compact runs use
`32e9c0211c831f26105e53bcd7a008943cec5fee`. Compact's empty rectangular tables
caused the scheduler to count prefill growth a second time, changing batch
composition. All individual processes ended naturally0; the comparison
controller ended1 after detecting31 differing requests and did not launch full4.
The failure includes ten token-different requests, so matching aggregate scores
alone would not have been a sufficient check.

At `deb21b2`, fixed reservations no longer add a second growth budget. The
rerun uses `7ca67d64599b421d0128f062c48325cc2a857b36`, containing that scheduler
fix with the same compact storage/reader. It recovers exact output-file identity.
Full4 runs independently from the initial frozen source. These are non-timing
quality runs, partly overlapping other finite correctness jobs on the other GPU;
their elapsed seconds are not used as performance evidence.

[Manifests, summaries and hashes](evidence/quality-v1/runs.json),
[original mismatches](evidence/quality-v1/compact-before-parity.json), and
[corrected parity](evidence/quality-v1/compact-after-parity.json) preserve both
outcomes. Multiple prompt lengths, more benchmarks, delayed policy and the other
registered checkpoints remain outstanding.

The [completed delayed-policy cohort](delayed-quality.md) adds resident/Graph results: the four non-Graph paths are request-exact at63/100; native Graph scores63/100 and alias Graph62/100, with15 token-sequence differences. This result does not establish quality equivalence.
