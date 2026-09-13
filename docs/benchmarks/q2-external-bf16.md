# BF16 native versus Hugging Face latency spot check

The default Q2 external benchmark compares **BF16** native vllm-lt with the
pinned Hugging Face Ouro implementation using one measured request per backend.
It loads weights once, runs one excluded feasibility/warmup request per backend,
then measures native and official once each. There are four requests total,
with no additional repetitions or retries.

Both use the same 128 input tokens, 64 greedy output tokens, four loops and
ignored EOS. The official adapter calls the published model forward with eager
attention and a compatible append cache containing 96 independent depth/layer
slots. This measures the cached adapter, not the full Transformers `generate()`
API used by the separate GSM8K accuracy benchmark.

Weights, activations and KV are BF16. RMSNorm reductions, rotary frequencies,
attention softmax/accumulation and other selective FP32 operations retain each
implementation's existing arithmetic. TF32 and reduced-precision matmul
reductions are disabled for both backends. No inference optimization is added.

The native 3 GiB KV pool stays resident for both backends; official final KV
adds 150,208,512 bytes. Weight pointers are shared and verified. Token-delivery
timestamps include model execution, actual greedy selection/readback, output
history construction and the adapter's checks. Loading, request setup, warmups,
detailed cache inspection and report writing are separate. Final synchronized
latency, TTFT, TPOT and output throughput are recorded independently.

## Prepare and run

Use a clean committed checkout, the prepared local checkpoint, Transformers
4.55.0 and an environment without the optional `kernels` package. The default
[v2 contract](../../benchmarks/fixtures/ouro-q2-external-bf16-contract.json) pins
native production bytes from PR14 `5aa3bf7`; changing production code requires
a separately versioned contract. The probe freezes the actual source, checkpoint,
input tokens, interpreter, dependencies, CPU/NUMA affinity and exact GPU UUID.
It performs no CUDA discovery or tensor loading.

Select an available exact GPU ID using the host scheduler. Set the variables
below to the verified device, UUID, CPU set, NUMA node, interpreter and paths.
Keep the same environment and CPU/NUMA binding for probe and execution. Use
node-local compilation caches and keep unrelated preparation off the worker
cores and their SMT siblings.

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TRITON_CACHE_DIR=/tmp/q2-bf16-triton
export HF_MODULES_CACHE=/tmp/q2-bf16-hf-modules
numactl --all --physcpubind="$BENCH_CPUS" --membind="$BENCH_NUMA" \
  "$BENCH_PYTHON" -m vllm_lt.benchmarks.q2_external probe \
  --model "$BENCH_MODEL" --gpu-ids "$BENCH_GPU" --gpu-uuid "$BENCH_UUID" \
  --output "$BENCH_PLAN"
gpu run --gpu-ids "$BENCH_GPU" --wait 5m --timeout 70m \
  --note "Q2 BF16 single-request comparison" -- \
  numactl --all --physcpubind="$BENCH_CPUS" --membind="$BENCH_NUMA" \
  "$BENCH_PYTHON" -m vllm_lt.benchmarks.q2_external run \
  --plan "$BENCH_PLAN/plan.json" --output "$BENCH_RUN"
"$BENCH_PYTHON" -m vllm_lt.benchmarks.q2_external report --run-dir "$BENCH_RUN"
```

The feasibility requests compile and exercise the complete workload, check
finite outputs and validate cache lifetimes. Each measured history must match
its own backend's feasibility history. Cross-backend token agreement is reported
separately: BF16 rounding can change greedy choices. This is a fixed-workload
latency spot check, not an accuracy test or proof of numerical equivalence.
Use the GSM8K regression for accuracy; its earlier results do not validate this
different source revision and workload.

Before each v2 measured request, the worker verifies every existing thread has
the frozen CPU affinity and samples contention for 200 ms. A failed check aborts
before generation; the sample is outside the delivery interval but inside the
case deadline. The result records this preflight for offline reconstruction.

Each v2 measured request also records CPU busy ticks for the bound cores and SMT siblings,
subtracts worker CPU ticks, and checks the main thread's runnable delay. More
than 0.1 background CPU cores or 5% runnable delay fails the run and blocks later
requests. This bounded contention screen is not a guarantee of exclusive host
resources. The controller's own small CPU cost is included as background work.

Stop on source/control drift, nonfinite outputs, backend history drift, cache,
cleanup, time or artifact limits. The total limit is one hour, each case ten
minutes, with 32 MiB total and 2 MiB per case. Preserve partial results; do not
replace failed rows. Only task-owned processes and memory are cleaned up.

## Results and historical evidence

The [September 12 BF16 result](q2-external-bf16-20260912.md) includes one
measured request per backend and the complete raw evidence archive.

A successful v2 report means the four requests and required controls passed.
Its single timing pair does not establish variability or a general speedup.
`equivalence.passed` reports exact token agreement independently from that
execution result. The report does not close the broader Q2 quality milestone.

The original [FP32 results](q2-external-20260911.md) and their
[final timing invalidation](../reviews/evidence/q2-external-final-qualification.json)
remain historical evidence. FP32 v1 plans can still be audited; the default
probe creates BF16 v2 plans. Controller-polled files use atomic replacement,
and derived reports are excluded from their own input inventory so repeated
report generation is byte-identical.
