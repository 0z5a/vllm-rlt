# Loop-level speculative execution

This opt-in synchronous prototype implements the mixed-depth batching portion of
[RFC #43](https://github.com/ThinkFlowLab/vllm-rlt/issues/43): advance shallow work
for later token positions in the same recurrent call as deep work for earlier
positions. It retains the existing finite speculative window and atomic round
commit. The default remains the original block verifier.

## Enable

```python
SpeculativeConfig(
    num_speculative_tokens=4,
    draft_loops=2,
    target_loops=4,
    loop_level_pipelining=True,
)
```

The offline and serving CLI accept `--speculative-tokens 4 --loop-level-pipelining`.
Greedy-only early suffix pruning can be enabled separately with
`--early-rejection-pruning`; it is off by default. Top-p and top-k sampling are
supported by the same loop pipeline, but sampling still verifies the complete
finite window before applying rejection sampling. Early pruning is intentionally
greedy-only so it does not change the sampling RNG schedule.
Existing restrictions apply: `last_exited`, full fixed target depth, synchronous
refill scheduling, and no preemption or PD.

The implementation has three execution choices:

| Mode | Recurrent schedule | Verification | Intended use |
| --- | --- | --- | --- |
| Baseline | Complete all draft positions, then complete target depth | One batched verifier window | Reference path and default |
| Loop | Advance active positions one depth per mixed-depth wave | Reorder completed target rows into the common verifier | Measure mixed-depth batching |
| Loop + greedy prune | Loop schedule with target-ready rejection checks | Stop future suffix branches after the first mismatch | Optional greedy experiment |

The loop mode is a synchronous pipeline. It does not add CUDA streams, reduce the
number of recurrent rows, or retain hidden state across speculative rounds. Its
possible benefit comes from the shape and timing of mixed-depth batches; metadata,
Python scheduling, Coda fragmentation and dynamic CUDA Graph shapes are additional
costs.

## Schedule and state

For K=2, d=2 and D=4, A is the last committed input, B and C are proposals.
Numbers below are zero-based depths; columns are successive recurrent calls.

| Input position | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | 0 | 1 | 2 | 3 | | | | |
| B | | | 0 | 1 | 2 | 3 | | |
| C | | | | | 0 | 1 | 2 | 3 |

After A completes depth 1, its shallow head proposes B. Call 3 batches A at
depth 2 and B at depth 0. After B completes depth 1, it proposes C while A's
full-depth head produces the distribution that verifies B. These head rows are
batched too. C's full-depth distribution supplies the bonus token.

Each branch stores its item index, position offset, hidden state and completed
loop count. A child starts d calls behind its parent. Therefore, at every layer,
the same-depth KV for earlier positions is already available before the child's
attention. `_prepare_batch` supplies a separate depth plane and causal context
length for every row. The pipeline uses single-query paged attention, not packed
multi-query prefill. No shallow KV is copied into uncomputed deep planes.

Each position/depth is computed exactly once. Proposals remain on the device and
greedy target IDs are read back once per mixed-depth wave. For sampling, the
filtered proposal distribution `q` is retained for each draft and the common
verifier applies rejection sampling with the request's generator from the shared
`RngRegistry`. Different GEMM shapes and the changed draft/verify interleaving can
change BF16 results or the fixed-seed token sequence; same-seed agreement is not a
general sampling correctness guarantee.

Completed target logits are reordered into item/position order for the common
verifier. `SpeculativeResult` and engine commit stay unchanged: keep KV through
the initial input and accepted candidates; invalidate every depth's rejected
suffix; leave correction/bonus unforwarded. Engine EOS/length handling and
exception cleanup remain responsible for the request lifecycle.

## Scope and benefit assessment

The default finite-window path does **not** implement cross-round hidden-state
retention or the complete continuous LoopSpec algorithm. The optional greedy
pruning path checks a target row when its coda output is ready and skips later
suffix branches after rejection. Work that already entered the mixed-depth wave
cannot be cancelled. Sampling still runs the complete window before
verification. Greedy target and draft IDs are read back once per mixed-depth
wave; the host-side rejection decision remains the pruning synchronization
boundary. No new CUDA streams or asynchronous scheduler are introduced.

For a uniform window, both the original block verifier and this pipeline require
`d*K + D` recurrent calls and `(K+1)*D` recurrent rows. Pipelining changes batch
shapes and head batching, not total recurrent work or that critical path. Its
cost also includes extra per-step metadata preparation and loss of the original
verifier's packed FA4 attention. Speedup must be measured; mixing depths alone
does not guarantee an improvement. Reducing round bubbles or wasted suffix work
would require a follow-up design for acceptance and state lifetime.

## Validation

```bash
OMP_NUM_THREADS=1 python -m pytest -q
python -m ruff check .
python -m ruff format --check vllm_rlt tests benchmarks

# On an allocated CUDA device with Triton:
python -m pytest -q tests/test_speculative.py --run-gpu -m gpu -k 'not flash_attn_4'
```

Tests exercise mixed depths including non-divisible d/D, actual hidden/logit/KV
agreement against a serial reference, every rejection boundary and bonus,
greedy/sampling batches, K=1/2/4/8, shortened windows, EOS, prefix reuse,
incremental allocation, cancellation and partial execution failure. CUDA coverage
checks eager/graph parity and one greedy readback per round. The inherited GPU
readback test also replaces its obsolete `OuroConfig.tiny()` call with the current
test helper.

For a bounded real-checkpoint feasibility run:

```bash
python -m benchmarks.loop_speculative \
  --model /path/to/local/Ouro-1.4B \
  --cuda-graphs --k 4 --concurrency 1 --new-tokens 32 --repeats 5 \
  --output /tmp/loop-speculative.json

# Sampling variant (top-p + top-k):
python -m benchmarks.loop_speculative \
  --model /path/to/local/Ouro-1.4B \
  --cuda-graphs --k 4 --concurrency 8 --new-tokens 32 --repeats 5 \
  --temperature 0.8 --top-p 0.8 --top-k 7 \
  --output /tmp/loop-speculative-top-p-top-k.json
```

The harness warms both engines, completes a common prefill barrier before timing,
ignores EOS to commit equal work, synchronizes timing boundaries, alternates trial
order, and saves raw timings, outputs, first divergences and runtime metadata.
For large loop batches, `--cuda-graph-max-batch-size 256` allows Graph capture of
the 2C mixed-depth wave when C=128; the default is 128.
`--prompts` accepts a JSON list of strings; `--backend flash_attn_4` requires
compatible hardware. Peak memory includes both resident engines; the recorded
checkpoint config hash is not a weight hash. This small run does not replace
RFC #43's fixed-input real-model diagnostics, sampling distribution checks,
long-context/serving matrix or BF16 acceptance criteria.

### Rebase verification (2026-10-18)

The current matrix was measured at `9586b4a` on an A100-SXM4-40GB with BF16,
Triton, CUDA Graphs, `K=4`, `d=2/D=4`, 32 new tokens and five alternating
repetitions. It covers C=1/4/8/16/32/64/128. Greedy results include baseline,
loop and opt-in loop + early pruning; sampling results cover top-p (`p=0.8`) and
top-p + top-k (`p=0.8,k=7`) for baseline and loop. The full tables and raw JSON
paths are recorded in `notes/vllm-rlt/loop-speculative-rebase-2026-10-18.md`.

The result is workload-dependent: with the default graph limit of 128, the loop
loses at C=128 because its mixed-depth rows fall back to eager execution. Raising
`cuda_graph_max_batch_size` to 256 recovers C=128 and gives about +3.4% for the
loop and +2.8% for loop + greedy pruning. Sampling results below the graph256
ablation use the original graph limit of 128 and are mixed; they can also produce
fixed-seed divergence because the loop changes RNG consumption order. The loop
remains opt-in pending a broader profile and serving workload evaluation.
