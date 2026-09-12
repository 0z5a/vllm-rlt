# Ouro GSM8K accuracy

Compare vllm-lt against the released Transformers implementation in BF16 before
assessing changes to inference arithmetic. The original Ouro paper reports
**78.92%** for Ouro-1.4B with four loops. Its
[evaluation settings](https://arxiv.org/html/2510.25741v5#A3.T16) specify
3-shot CoT, strict match, and lm-eval-harness. The exact harness revision,
demonstrations, and generation limit are not specified there; the choices below
are pinned project settings, not a claim of exact paper reproduction.

The evaluator uses lm-eval-harness 0.4.9.2's `gsm8k_cot` task, its first three
demonstrations, and its strict-match filter and metric. Both backends receive
identical token IDs without a chat template or added special tokens. Generation
is greedy, batch size one, fixed four loops, with natural EOS and the harness's
stop sequences. The default limit is 1,024 new tokens and 2,048 total tokens;
preparation rejects prompt truncation. BF16 weights/activations retain each
backend's existing reduction precision.

## Run

Install the optional evaluation dependencies in an environment with working
PyTorch and Triton, then prepare the pinned checkpoint on CPU:

```bash
pip install -e .
pip install -r benchmarks/requirements-accuracy.txt
hf download ByteDance/Ouro-1.4B \
  --revision 574fa66cb8bf5abdc979642d01cf2b79b16bfab1 --local-dir /path/to/ouro
python -m benchmarks.gsm8k prepare --model /path/to/ouro --output /path/to/protocol.json
```

Preparation downloads the pinned GSM8K data, verifies local model files against
the pinned Hub revision, and saves every prompt, token ID, answer, package
version, and model hash without initializing CUDA. The default covers all 1,319
test examples. `--limit N` creates a bounded screen. Use `--split train --limit 2`
for a disjoint feasibility run; set `--min-reference-accuracy-pct 0` for that
screen. Freeze any threshold or generation-budget changes before inference.

Run each backend with the same protocol on the same reserved GPU. On hosts with
the GPU scheduler, wrap each command in `gpu run --gpu-ids <available-id>
--timeout 8h --note "Ouro GSM8K accuracy" --`. Keep CPU/NUMA affinity fixed.

```bash
python -m benchmarks.gsm8k run --backend transformers \
  --protocol /path/to/protocol.json --output /path/to/transformers
python -m benchmarks.gsm8k run --backend native \
  --protocol /path/to/protocol.json --output /path/to/native
python -m benchmarks.gsm8k compare --transformers /path/to/transformers \
  --native /path/to/native --output /path/to/comparison.json
```

The Transformers reference executes the pinned official Python code with eager
attention and the standard Transformers `DynamicCache`, initialized with 96
depth/layer slots. Explicit `exit_at_step=3` selects the fourth loop's logits.
The native backend uses its existing Triton attention and engine scheduler.
Both stop at the same harness-defined conditions and check generation logits
for non-finite values. The evaluator changes no production inference arithmetic.

## Interpret results

The default observed accuracy gate requires Transformers accuracy of at least
75.92% (three percentage points below the published score) and native accuracy
no more than one percentage point below Transformers. These are project
thresholds, configurable during preparation. `compare` writes its report and
exits nonzero on failure. Its paired standard error describes uncertainty; an
observed pass is not a statistical non-inferiority proof or a kernel equivalence
test. A subset is a regression screen, not a full benchmark reproduction.

Each run retains raw generated text and token IDs, extracted answers, per-item
correctness, stopping reasons, and prompt hashes. A completed summary is written
only after every example finishes. Exceptions stop the run and retain the
completed JSONL prefix; missing or mismatched results cannot pass comparison.
Loading is reported separately, and elapsed times are not performance claims.

For the initial local experiment, the isolated variable is the inference backend.
Controls are the pinned checkpoint/data/task, BF16, four loops, all 1,319 test
examples, greedy decoding, stop rules, budgets, and exact GPU/CPU affinity.
Run two training examples per backend for feasibility, excluded from the test
score, then one full test pass per backend. A single pass is predeclared because
this is deterministic greedy accuracy evaluation; uncertainty comes from paired
test examples rather than repeated identical generations. Stop on execution,
non-finite, context, or cache failures, or at the eight-hour per-backend deadline.
Preserve failed results and report discrepancies without post-hoc rescoring.
