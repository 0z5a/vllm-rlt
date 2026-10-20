# Cross-round asynchronous self-speculation

This opt-in path uses fixed-depth Ouro, `last_exited` KV, greedy sampling,
Triton attention, eager CUDA execution, and refill scheduling. Historical tests used H20;
FlashAttention is not required. Draft depth is configurable, with `d=2, D=4`
as the initial validation configuration. The target depth must match the model.

```python
from vllm_rlt import LLM, ExecutionConfig, SamplingParams, SpeculativeConfig

llm = LLM(
    "artifacts/models/Ouro-1.4B",
    device="cuda",
    attention_backend="triton",
    execution_config=ExecutionConfig(async_scheduling=True),
    speculative_config=SpeculativeConfig(num_speculative_tokens=4),
)
outputs = llm.generate(
    ["Explain speculative decoding."],
    SamplingParams(temperature=0, max_loops=4, exit_threshold=1.0, max_tokens=64),
)
```

The CLI combines `--async-scheduling --attention-backend triton
--speculative-tokens 4 --draft-loops 2 --target-loops 4 --temperature 0`.
Random sampling, adaptive exit, CUDA Graphs and preemption are rejected.
PD can enable speculation on its decode stage. Incoming prompts wait for transfer
completion and shared scheduler admission before CODA bootstraps local speculative
rounds. Cancellation retains GPU resources until the final round completes.
The speculative runner owns its compute and result-copy streams. Setting
`multi_stream=False` orders the copy on the compute stream as well; cross-round
CPU/GPU submission remains asynchronous, but D2H no longer overlaps computation.

## Execution and ownership

Speculative work enters through `Executor.submit()` like ordinary batches.
The executor dispatches a speculative selection to its local round controller;
the controller uses the same `ModelRuntime` for PRELUDE, RECURRENT and CODA.
It retains its own draft/verify ordering, candidate state and acceptance logic.
There is no cross-stage handoff for each speculative step.

Prompt bootstrap uses the ordinary runner. Refill scheduling can use atomic
prefill or `atomic_prefill=False` to yield between prompt loops. PD can import
the completed prompt into the decode stage. These combinations do not enable
mixed scheduling of prompt rows with speculative verification rows.


The synchronous runner reads the acceptance result before constructing the next
round. Cross-round submission needs a different ownership contract: the GPU must
advance the next token and actual KV position while the CPU reserves a safe upper
bound. `MetadataBuilder.prepare_device()` creates and validates a borrowed
`DeviceKVMetadata` containing allocation identities and device access tensors,
without retaining the manager. The runner owns those tensors and completion
events, and retains allocations until the final GPU reader completes. The batch
does not treat uncommitted positions as populated CPU KV. The scheduler tracks outstanding
output reservations separately from committed tokens.

A simpler alternative is to copy each result asynchronously but wait before
submitting the next round for that request. That can overlap independent requests,
but retains a CPU dependency between consecutive rounds. The two-buffer design
removes that dependency at the cost of device state, reservation slack and up to
one discarded round after EOS. The existing synchronous speculative path remains
available by disabling `async_scheduling`; ordinary decoding remains the default
when no `SpeculativeConfig` is supplied.

Prefill and the first sampled token use the native bootstrap path. Thereafter,
the GPU retains each request's next input token and actual position. A round
drafts candidates, retains shallow hidden states, verifies the deep loops in a
batch, and selects the accepted prefix plus correction/bonus on device. Each
query row has its own context length; no packed FlashAttention path is used.

Two reusable `SpeculativeRoundBuffer` slots allow the CPU to submit the next round before delivering the
previous result. All request computation runs in stream order. CPU positions
include outstanding output reservations and are upper bounds used only for
capacity allocation; GPU positions determine RoPE, KV writes and attention.
Requests near output/context limits may wait for a result to recover rejected
reservation capacity. An effective K of zero is a full-depth single-token step.

For a round beginning at position `s` with `a` accepted candidates, the valid
KV frontier and next input position are `s+a+1`. Invalid suffix bytes may remain
in allocated pages, but attention lengths exclude them. The CPU never truncates
GPU state when collecting an older round. CPU `written` bookkeeping continues
to describe native prefill, not in-flight speculative decode; device descriptors
use the reserved allocation and stream-ordered writes instead. Only completed
full prompt pages are eligible for prefix caching.

Every ticket owns its result buffer and page-table snapshot until compute and
copy events complete. EOS/length stopping is applied in output order on the CPU;
one already-submitted later round can finish but its outputs are discarded.
Cancellation removes scheduling eligibility immediately. KV allocations and
device state survive until their final GPU user completes. Request object
identity isolates stale tickets when a request ID is reused.

## Validation

Inside a Slurm GPU allocation, use the repository environment:

```bash
.venv/bin/python -m pytest tests/test_async_speculative.py --run-gpu -q
.venv/bin/python -m pytest tests/test_serving.py --run-gpu -k speculative -q
```

The tests exercise two GPU rounds before CPU delivery, every rejection position,
K/length boundaries, mixed requests, prefix reuse, EOS, cancellation, ID reuse
and partially submitted failures. CPU-only test results do not qualify the GPU
path. Compare BF16 runs on the same hardware and Triton backend: synchronous
speculation versus asynchronous speculation, and fixed-depth native async versus
async speculation. Record committed tokens, variability, memory and traces;
do not infer a speedup from successful asynchronous submission alone.

The page-table snapshot uses flat reusable pinned/device storage. Each round
views only `batch_size * max_loops * active_page_width` entries as a contiguous
three-dimensional table, clears that region and copies it once. Inactive storage
is untouched; ragged rows are padded within the active width. This reduces the
number of entries prepared and submitted for H2D. The
[recorded measurements](runtime-history.md#active-region-preparation-and-transfer)
separate that reduction from end-to-end effects.

## Historical evidence

The [validation history](runtime-history.md#recorded-gpu-validation-2026-10-10)
retains the dated correctness runs, BF16 measurements, profiling observations
and qualification limits. These results describe their recorded revisions;
they do not qualify the current checkout or establish production serving capacity.
