# M3 prerequisite: inactive rows

This implements the masked-row prerequisite of [M3](https://github.com/hsliuustc0106/vllm-lt/issues/6).
It makes padded eager execution safe enough to validate before introducing
persistent buffers and CUDA graphs. Production dispatch remains compact.
Acceptance for this PR concerns AC-M3-02 and the eager bookkeeping/publication
checks needed to support it. M3 stays open until its remaining storage, capture,
fallback, and end-to-end performance gates pass.

## Constraints and scope

The baseline is the FP32 metadata implementation in [M2 PR #12](https://github.com/hsliuustc0106/vllm-lt/pull/12).
Preserve per-request positions and depth tables, full-depth prefill, actual
synchronous gates, cumulative hazards, LAST-EXITED propagation, lifetime page
reservation, and scheduler order. This is historical FP32 evidence; new GPU
comparisons follow the [BF16 precision policy](precision-policy.md).

Padding is private runner state. It cannot manufacture requests, participate in
coda or sampling, advance an RNG, or change a request waiting in another stage.
There are no persistent buffers, graph capture/replay, bucket selection, public
configuration, asynchronous routing, allocator changes, or performance claims
in this prerequisite.

## Why masked rows

Masked rows keep a single attention launch per layer for mixed valid history
lengths and provide fixed physical shapes for later graph capture. Grouping rows
by equal history length would avoid padding work but split attention into more
launches. The tradeoff here is extra work on padded rows; this PR does not measure
whether that tradeoff improves performance.

Written-position bookkeeping still calls `allocation.written[depth][layer].add(position)`
for each live row and layer. Consolidating those updates into the descriptor
lifecycle belongs to later host-time work and must preserve initialized-prefix
and duplicate-write checks. The padded Torch backend also uses host indexing and
mask conversion per layer; it remains a private eager diagnostic path.

## Borrowed metadata and model boundary

`KVCacheManager._prepare_batch` retains its compact path. Its descriptor captures
live host rows and allocation identities for one traversal. A compact descriptor
has identity row order and no device mask, preserving M2's existing metadata
transfers. The private padding helper adds explicit physical row count,
live-to-physical row mapping, and a device boolean mask from the same validated
host mapping. It rejects invalid or duplicate row indices and insufficient table
width before any KV mutation.

Inactive rows have zero positions and lengths and invalid address sentinels;
no real page is allocated as dummy padding. Ownership, stale-allocation checks,
duplicate active write rejection, and layer-specific initialized-prefix checks
still apply to the compact live rows. Bookkeeping advances only for those rows
after that layer's write is submitted on the existing ordered stream.

The model's private prepared traversal accepts physical hidden rows and this
descriptor. It sanitizes inactive hidden inputs before projections and returns
exact zero hidden states and gate logits for inactive rows, including when the
gate has a nonzero bias. A zero-live-row traversal is safe no-work. The normal
`recurrent` entry point still prepares and executes a compact nonempty batch.

## Memory access and publication

Both attention backends use the mask before indexing query, page-table, or KV
values. Inactive output is finite zero. Triton branches/masks guard loads before
an invalid sentinel can be dereferenced; post-hoc multiplication cannot supply
this guarantee. Padded KV scatter guards address loads and K/V loads/stores.
The compact write path remains unchanged. The reference path indexes only live
rows. Active zero-length metadata remains invalid at the manager boundary.

The runner exposes a private decode hook for validation. Its padded helper
places compact hidden states at the selected physical rows and gathers only
live hidden states and gates back into scheduler order. Prefill remains compact.
Returned active state must not alias scratch padding storage; inactive rows
never reach host routing or result publication. Validation observes physical
model tensors through the same live mapping while retaining logical fixture,
position and depth identities.

Changing GEMM row count can change FP32 rounding. Whole-model A/B checks therefore
retain Q1's final-logit tolerance and exact actual token/exit requirements;
intermediate hidden, gate and KV deltas are finite diagnostics. Held-input
attention and KV scatter tests separately require exact compact/padded agreement
and byte-exact guard isolation. These are distinct checks.

## Acceptance evidence

A resolved contract must freeze source and input hashes, the exact matrix,
GPU/environment/affinity controls, byte limits and stop conditions before device
execution. This is a deterministic correctness comparison with one execution per
declared configuration/case, two excluded feasibility cases, and no timing
estimate or retry on failure. The final report links raw evidence and records
any unavailable memory-access checker. Guard checks and source-level masked
access evidence remain required regardless of checker availability.

A passing prerequisite does not establish graph-safe Python bookkeeping,
persistent pointer stability, asynchronous failure recovery, capture safety,
launch savings, or the remaining M3 acceptance criteria.

## Commands

The M3 experiment harness lives in checkout-only `benchmarks/m3_inactive*.py`,
alongside M2, and is excluded from the installed runtime package. Run these
commands from a source checkout. The checked-in JSON contract is authoritative;
resolved plans and intermediate test logs live with the historical evidence.
The [result report](benchmarks/m3-inactive-20260911.md) links the frozen protocol
and release. Reproduce the original run with its archived sources, since source
and import hashes are part of each plan's identity.

Use the same prepared environment and checkpoint for two clean execution
checkouts. A restores the five production paths named by
`benchmarks/fixtures/ouro-m3-inactive-contract.json` from the M2 snapshot; B
contains the inactive-row implementation. Their validation code, fixtures and
other files must match. The probe rejects dirty sources, unexpected differences,
input drift and incompatible dependencies without initializing/querying CUDA or
loading checkpoint tensors.

```bash
python -m benchmarks.m3_inactive_run probe \
  --baseline-root /absolute/path/control \
  --candidate-root /absolute/path/candidate \
  --contract /absolute/path/candidate/benchmarks/fixtures/ouro-m3-inactive-contract.json \
  --model-path /absolute/path/ouro-1.4b \
  --gpu-id 7 --output /absolute/path/m3-plan
```

Resolve GPU IDs through the scheduler before planning; `7` above is an example.
Use the selected available exact ID for both sides. The probe records the
actual CPU/NUMA binding and runtime variables. Reuse those exact controls for
reserved execution. The resolved contract records the concrete interpreter,
checkouts, source/input hashes and affinity command used for a run.

```bash
gpu run --gpu-ids 7 --nonblock --timeout 1h \
  --note 'vllm-lt M3 inactive-row correctness A/B' -- \
  python -m benchmarks.m3_inactive_run run \
  --plan /absolute/path/m3-plan/plan.json --output /absolute/path/m3-run

python -m benchmarks.m3_inactive_run report \
  --run-dir /absolute/path/m3-run
```

The controller runs two fresh workers with one model load each. Its exact order
is A feasibility, 26 A kernel evaluations, eight A qualification model cases,
B feasibility, 26 B kernel evaluations, and three B qualification model cases:
65 executions. The 27 qualification comparisons and one excluded feasibility
comparison retain their separate counts. Kernel input hashes are computed on the
CPU before the plan is frozen; each execution checks them before device
allocation. Zero memory-checker executions are planned on this host because the
tool is unavailable.

The offline report requires the complete ordered case/kernel set, correct logical
and physical observation identities, original numerical/discrete gates, exact
held-input/guard checks, matching controls and successful cleanup. A recorded
failure stops further cases and remains a failed outcome with incomplete
remaining coverage. Corrupt identities or chronology are invalid evidence.
Neither condition qualifies the implementation. Generated plans, manifests,
typed payloads and reports belong under ignored `artifacts/` and in linked
release assets; the source tree retains the contract and compact manual report.
