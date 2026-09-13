# Numerical validation

The validation suite compares native Ouro execution with an independent dense
oracle and the pinned official reference. Numerical bounds, token agreement and
exit decisions are separate checks; completion alone is not qualification.
Keep generated plans, tensor dumps and reports outside Git.

## Running the suite


Prepare a compatible environment and local checkpoint before the experiment.
Reuse an existing suitable environment where available. Verify scheduler status
and choose an available exact physical ID, then freeze the plan without device
discovery:

The `validation` package extra pins the official adapter's Transformers/Hub/
tokenizer dependencies: `python -m pip install -e '.[validation,dev]'`. Its
environment must have optional `kernels` absent; probe rejects incompatible
environments before checkpoint loading or CUDA discovery.

```bash
python -m vllm_lt.validation probe \
  --suite benchmarks/fixtures/ouro-q1.json \
  --contract benchmarks/fixtures/ouro-q1-contract.json \
  --model-path /path/to/pinned/ouro-1.4b \
  --gpu-id <available-physical-id> --output artifacts/q1-plan
```

Run the unchanged source/plan in a new artifact directory. The driver verifies
the scheduler-assigned device against the frozen ID before touching CUDA:

```bash
gpu run --gpu-ids <same-physical-id> --wait 10m --timeout 2h \
  --note 'vllm-lt Q1 frozen numerical validation' -- \
  env OMP_NUM_THREADS=1 python -m vllm_lt.validation run \
  --plan artifacts/q1-plan/plan.json --output artifacts/q1-run

python -m vllm_lt.validation report --run-dir artifacts/q1-run
```

The two-hour outer limit wins over ten minutes per execution. Numerical
counterexamples remain recorded while the remaining planned cases run.
Invalid history, nonfinite values, device failure, cleanup failure, or exhausted
budgets stop execution and preserve incomplete evidence. A source fix or new
hypothesis requires a separately recorded plan; the driver never adds trials.
