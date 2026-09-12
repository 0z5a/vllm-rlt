# Ouro GSM8K-100 accuracy benchmark

Use **100 fixed questions from GSM8K `main`/`test`**, with the original released
Hugging Face Transformers model as the accuracy baseline. Report its measured
score on these same questions, native accuracy, and the paired difference.
The paper's full-dataset score is not the baseline for this subset.

The dataset revision, seed (default 0), SHA-256 selection rule and original
source row IDs are frozen in the prepared protocol. Selection uses only dataset
revision, split, seed and row ID, before generation. It never filters questions
by answer, model correctness or prompt length. Oversized prompts fail preparation
instead of being truncated or replaced. Both backends consume the same saved
prompts and token IDs.

## Protocol

- Pinned `ByteDance/Ouro-1.4B` checkpoint and official release code; BF16,
  fixed four loops, greedy decoding, one request at a time.
- lm-eval-harness 0.4.9.2's `gsm8k_cot`, first three demonstrations and strict
  answer extraction/scoring, following the Ouro paper's stated 3-shot CoT setup.
- No chat template or added special tokens. Natural EOS and identical harness
  stop sequences; at most 1,024 new tokens within a 2,048-token total context.
- Exact-answer accuracy with unparseable answers counted as incorrect. Save
  raw text, token IDs, extracted answers, correctness and stopping reason.
- Default regression threshold: native may lose at most one percentage point
  against measured Transformers accuracy. On 100 questions, one question is
  one percentage point. This is a regression screen, not proof of population
  equivalence; report paired disagreements and uncertainty as well as the score.

The reference calls `AutoModelForCausalLM.from_pretrained(...,
trust_remote_code=True)` and the released model's `generate()`. It uses eager
attention and the standard Transformers `DynamicCache` with 96 depth/layer slots,
plus `exit_at_step=3` for fourth-loop logits. This cache setup accommodates the
pinned release's older cache interface; the model source is not patched.
The candidate uses the existing native engine and Triton attention. Both retain
their existing BF16 reduction precision.

## Prepare and run

Use the same prepared environment and local checkpoint for both backends:

```bash
pip install -e .
pip install -r benchmarks/requirements-accuracy.txt
hf download ByteDance/Ouro-1.4B \
  --revision 574fa66cb8bf5abdc979642d01cf2b79b16bfab1 --local-dir /path/to/ouro
python -m benchmarks.gsm8k prepare --model /path/to/ouro \
  --limit 100 --seed 0 --output /path/to/gsm8k-100-protocol.json
```

Preparation runs without CUDA. It verifies the local checkpoint against the
pinned Hub release, downloads the pinned dataset, and freezes the selected
questions, prompts, token IDs, controls and package/model hashes. Keep generated
protocols and results outside Git. `--all` explicitly selects the complete test
split; changing `--limit` or `--seed` creates a different benchmark.

Freeze GPU/CPU affinity and the run budget before inference. Use two disjoint
training questions (`prepare --split train --limit 2`) for feasibility, then
one pass per backend on the 100 test questions: 4 feasibility generations and
200 scored generations total. Repeated identical greedy passes are not needed.
On a host with the GPU scheduler, run each backend through
`gpu run --gpu-ids <available-id> --timeout 2h --note "GSM8K-100 accuracy" --`
and keep the same exact GPU and CPU/NUMA controls for both passes.

```bash
python -m benchmarks.gsm8k run --backend transformers \
  --protocol /path/to/gsm8k-100-protocol.json --output /path/to/transformers
python -m benchmarks.gsm8k run --backend native \
  --protocol /path/to/gsm8k-100-protocol.json --output /path/to/native
python -m benchmarks.gsm8k compare --transformers /path/to/transformers \
  --native /path/to/native --output /path/to/comparison.json
```

The comparison reports both correct counts and accuracies, native-minus-reference
percentage points, losses/gains and answer disagreements. It rejects incomplete
or mismatched runs. Exceptions and non-finite logits stop execution while
preserving completed records; do not replace examples or reduce the denominator.
An optional `--min-reference-accuracy-pct` floor and `--max-regression-pp` changes
must be declared during preparation. There is no default paper-score floor.
Loading and generation durations are recorded separately; they are not a
throughput benchmark. No GSM8K-100 GPU score has been established by this change.

Dataset: [GSM8K](https://huggingface.co/datasets/openai/gsm8k).
The [Ouro evaluation settings](https://arxiv.org/html/2510.25741v5#A3.T16) do not
pin the exact harness revision, demonstrations or token limits, so the settings
above are explicit project choices rather than an exact paper reproduction.
