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
Existing restrictions apply: `last_exited`, full fixed target depth, synchronous
refill scheduling, and no preemption or PD. Greedy and temperature/top-k/top-p
sampling use the same acceptance implementation as the original runner.

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
are read back with greedy target IDs once per round. Sampling retains the actual
filtered q distribution and draws all drafts before acceptance, preserving the
existing per-request RNG ordering. Different GEMM shapes can still change BF16
results; same-seed agreement is not a general sampling correctness guarantee.

Completed target logits are reordered into item/position order for the common
verifier. `SpeculativeResult` and engine commit stay unchanged: keep KV through
the initial input and accepted candidates; invalidate every depth's rejected
suffix; leave correction/bonus unforwarded. Engine EOS/length handling and
exception cleanup remain responsible for the request lifecycle.

## Scope and benefit assessment

This version does **not** implement streaming acceptance, early rejection pruning,
cross-round hidden-state retention, or the complete continuous LoopSpec algorithm.
It runs all K+1 positions to target depth even when an earlier proposal will be
rejected. No new CUDA streams or asynchronous scheduler are introduced.

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
```

The harness warms both engines, completes matched prefill and first-token
generation before timing, ignores EOS to commit equal work, synchronizes timing
boundaries, alternates trial order, and saves raw timings, outputs, first
divergences and runtime metadata. `--prompts` accepts a JSON list of strings;
`--backend flash_attn_4` requires compatible hardware. Peak memory includes both
resident engines; the recorded checkpoint config hash is not a weight hash.
This small run does not replace RFC #43's fixed-input real-model diagnostics,
sampling distribution checks, long-context/serving matrix or BF16 acceptance
criteria.

### Local feasibility results (2026-10-07)

Base commit: `ecb1f8b505b7e831815b40aec3b4598619cca23a`, plus this working-tree
implementation. Local Ouro-1.4B, A100-SXM4-40GB, PyTorch 2.14.0+cu130,
BF16/Triton, CUDA Graphs, d=2/D=4, 32 new tokens after prefill/first-token,
five alternating measured repetitions after warmup. GPU 0 was idle before the
experiment. These are the harness's two short default prompts, cycled to fill
the batch; C=1 uses only the first. No workload generalization is implied.

| Concurrency | K | Baseline tokens/s | Pipeline tokens/s | Throughput change |
| --- | --- | --- | --- | --- |
| 1 | 1 | 53.74 | 53.13 | -1.1% |
| 1 | 2 | 57.18 | 55.90 | -2.2% |
| 1 | 4 | 58.59 | 56.92 | -2.8% |
| 1 | 8 | 44.92 | 43.17 | -3.9% |
| 8 | 1 | 383.68 | 380.84 | -0.7% |
| 8 | 2 | 409.14 | 402.28 | -1.7% |
| 8 | 4 | 420.43 | 409.88 | -2.5% |
| 8 | 8 | 324.42 | 316.53 | -2.4% |

Throughput is fixed committed tokens divided by median elapsed decode time.
Every measured sequence agreed with the matching original speculative run.
This does not prove general BF16 equivalence or agreement with native decoding.
FA4, long contexts, native real-model baselines and serving latency were not
evaluated. On this bounded workload the prototype is correct in the exercised
cases but **does not improve throughput**; it remains opt-in for experimentation.
Raw JSON from this local run is retained under `artifacts/loop-speculative/`
(ignored by git), including per-trial timings, outputs and first divergences.
