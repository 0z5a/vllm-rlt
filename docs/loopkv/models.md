# Model coverage

Coverage is in progress. An official checkpoint gate does not establish a speed
benefit or replace a natural-stop quality benchmark. Every weight set remains
available until its registered tests and independent readers are complete.

## Ouro-1.4B-Thinking

The official checkpoint at revision
`3aaa2224253a92ca45cf2e3d427c360e1ef9c93d` passed21 complete engine captures:
native, compact staging and compact with reclaimed credits at each batch limit
below. Each group has exactly equal complete request objects, including prompt
IDs, output tokens and exit depths. All block credits returned after every arm;
compact records and outstanding reservations returned to zero.

| B limit | Concurrent requests | Output tokens per arm | Native / staging / credits parity | Speedup |
|---:|---:|---:|---|---|
| 1 | 2 | 256 | Exact | Not measured |
| 4 | 8 | 1,024 | Exact | Not measured |
| 8 | 16 | 2,048 | Exact | Not measured |
| 16 | 32 | 4,096 | Exact | Not measured |
| 32 | 64 | 8,192 | Exact | Not measured |
| 64 | 128 | 16,384 | Exact | Not measured |
| 128 | 256 | 32,768 | Exact | Not measured |

These are P128/output-limit128 controlled runs with EOS ignored, greedy seed17,
original gate0.5 and two-to-four loops. The resident maximum equals B in each
arm. Each point has enough worst-future credits for B requests, so reclamation
does not increase the resident cap here. All three arms reuse one loaded model,
use per-step diagnostics and ran in a non-timing resource window. Their elapsed
seconds cannot be compared as performance measurements. The workload repeats
a frozen prompt fixture; it is not506 independent quality questions.

Source `cf70584d60913e1664439fb1b4acf603e2dff105`, RTX5090,
Torch2.12.1+cu130. Peak CUDA allocator reserved28,905,046,016bytes at B128;
native and candidate each finish all256 requests at that point. Controller and
all arms naturally completed; the final timestamp is1791148267.4786336.
[Frozen plan, all summaries and raw-file hashes](evidence/thinking-batch-gate-v1.json).

## Nanbeige and Huginn preparation

The standalone loader now supports native Nanbeige and Huginn without installing
Hub libraries in the GPU runtime. CPU comparisons cover single/sharded checkpoints
and FP32/FP16/BF16 against each native loader. Huginn retains FP32 persistent RoPE
data and verifies/reconnects tied embedding and output weights.

Huginn reuses the existing native implementation from upstream draft82 at
`3b893657f8541a1f5a6125e9703113b8d59e87ff`. Its prelude/coda use depth-zero KV;
only recurrent layers participate in finalization. Alias and compact storage
allow an unwritten coda layer to finish after recurrent finalization, while
rejecting any overwrite of finalized recurrent or already-written boundary
data. CPU tests cover R1/R3, all four cache modes and request-ID reuse.

The combined targeted CPU suite passed117 tests and36 address subtests, with
eight unavailable CUDA/official-checkpoint cases skipped. Official Nanbeige and
Huginn full-weight CUDA E2E remains pending. HRM-Text's public1B checkpoint is
registered for a separate PrefixLM/hierarchical-recurrence adapter; it is not
covered by these results.

### Tiny-model CUDA qualification

Frozen source `5d49b2a8bec75a9866097183dce6002202864df0` passed the finite
RTX5090/Torch2.12.1+cu130 gate and naturally exited0. Nanbeige R1/R2 and
Huginn R1/R3 each pass B1/4/16/32/64/128 with C2B: all four native, alias,
compact and reclaimed-credit modes return identical output/exit sequences and
all block credits. This is24 model/depth/batch points and96 execution arms
using tiny generated weights, not official checkpoint qualification.
[Exact source, raw hash and all24 points](evidence/model-cuda-v1.json).

| Model | B / C | Native–alias–compact–credits | Official E2E speedup |
|---|---|---|---|
| Tiny Nanbeige | B1–128 / C2B | Exact tokens/exits; all credits returned | Not measured |
| Tiny Huginn | B1–128 / C2B | Exact tokens/exits; all credits returned | Not measured |

## Ouro 2.6B official checkpoint

[All24 synchronous original-policy cache diagnostics](ouro-2.6b.md) completed with exact output tokens at B1–128/C2B. Alias/compact request objects match native; credits at B128 raise residency64→75 and change252/256 exit sequences. This is correctness/capacity evidence, not timing or quality.
