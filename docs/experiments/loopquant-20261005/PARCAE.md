# Parcae quantization adapter

This draft adds a differentiable dense adapter and seeded native likelihood on a
frozen integration base `d731275e6f6c224c25722dfae8ea399502f8b6dc`. The base combines
[family quantization PR23](https://github.com/0z5a/vllm-rlt/pull/23) at1b826d3 and
[Parcae runtime PR24](https://github.com/0z5a/vllm-rlt/pull/24) at6477ae0. Its
incremental review contains this adapter and evidence, not either dependency.

The adapter preserves diagonal exp/softplus injection, token-dependent value
embeddings and their gates, adjacent-pair RoPE, dtype-dependent Q/K RMSNorm,
ReLU squared with its0.5 factor, C projection and tied readout. Only explicitly
attached recurrent projections use the common quantizer. Boundary, injection,
value embedding/gate and readout remain protected. Q0 teacher and student share
one explicit initial random state, with gradients through every requested loop.
Native likelihood uses actual boundary/core KV, restores the caller's RNG and
frees the request allocation. Native evaluation requires the registered full
recurrence; dense R1/R4 diagnostics are not fixed-depth native qualification.

## Reference evidence

With unchanged author code at69284c13746e849104f738d6d1a347b1f457df76, the SDPA
adapter matches all12 tiny cases bitwise: FP32/BF16 × B1/B2 × P1/3/7, including
36 recurrent states and12 final logit tensors. The
[raw comparison](parcae-v1/parcae-adapter-author-tiny-attempt2.json) also retains
all12 eager controls, of which only7 pass the unchanged numerical envelope;
the maximum BF16 logit difference is0.25. These are distinct arithmetic paths.
[Reproduction script](parcae-v1/run_author_tiny.py).

The initial [manual-equation CPU comparison](parcae-v1/parcae-quant-cpu-attempt1.log)
retains two BF16 failures. Those remain strict expected failures in regression,
with original0.03/0.02 budgets. They are not counted as qualifications; SDPA
became the adapter default because it matches the unchanged author's path,
without changing error tolerances or substituting its attention implementation.
The independent FP32 equations, seeded likelihood, complete recurrence gradient
and teacher/student initialization checks pass.

The [frozen full CPU suite](parcae-v1/full-cpu-parcae-quant-attempt1.log) reports
676 passed,29 skipped,217 GPU cases deselected,2 strict expected BF16 failures,
and36 passing subtests. [Source manifest](parcae-v1/cpu-source-manifest.json).
Existing native full-checkpoint discrepancies in the dependency remain separate
unresolved G0 evidence. No native cached, quantized or performance gate passed.

## INT4 artifact and native preflight

The INT4 exporter now covers Parcae's six physical core projections per block.
Injection B/Alog/dtbias, C, token value embeddings and gates, boundary layers and
norms stay BF16. RoPE stays FP32. The artifact stores the embedding once and
restores its tied readout alias. RTN and supplied GPTQ codes share this format;
the supplied-code test does not qualify a GPTQ calibration procedure.

The frozen source `1e1f0858151a00b2fc003fc9b32a165ee15270b8` passes
[677 CPU tests](parcae-int4-v1/full-cpu-parcae-int4-attempt1.log), with 29 skipped,
221 GPU tests deselected, 2 retained strict expected failures and 36 passing
subtests. The [focused run](parcae-int4-v1/parcae-int4-cpu-attempt1.log) passes
18 tests. Successful temporary CPU weight payloads are removed.

Four [GPU cases](parcae-int4-v1/gpu-packet-manifest.json) are prepared for Graph
off/on × native/compact KV. They check encoded reload, tied readout, FP32 RoPE,
tokens above 255, repeated requests, fixed full recurrence and KV drain.
They are **not run**: the packet is local and awaits a resource grant. Parcae
uses synchronous scheduling; no mixed-depth or asynchronous support is claimed.
[Probe](parcae-int4-v1/probe.py), [finite controller](parcae-int4-v1/run.py).

## FP8 artifact and native preflight

The FP8 core uses one shared Q/K/V input producer and separate attention-output,
MLP-input and MLP-output producers. DYN, SH1, ST2 and per-loop layouts consume
the runtime's explicit row loop IDs. Each projection owns one encoded matrix;
the exported artifact preserves those bytes, BF16 protected parameters, the
tied readout and FP32 RoPE. Q/K RMSNorm, adjacent-pair RoPE, token value gating
and ReLU²/2 retain the native arithmetic. With fused normalization enabled,
blocks with value gates still compute the BF16 norm needed by that gate; this
has no measured performance benefit yet.

Source `08e636877b1e24a46cc65d028001e69a7867396b` passes the
[full CPU regression](parcae-fp8-v1/full-cpu-parcae-fp8-attempt1.log): 677 passed,
29 skipped, 237 GPU tests deselected, 2 strict expected failures and 36 passing
subtests. This validates import/collection and existing CPU behavior, **not the
CUDA implementation**. The [16-case GPU packet](parcae-fp8-v1/gpu-packet-manifest.json)
is prepared but not uploaded or run. It compares decoded FP8 references across
all four policies × Graph off/on × fused norm off/on, with fixed full recurrence,
encoded reload, tokens above 255, request-ID reuse and KV drain.
[Probe](parcae-fp8-v1/probe.py), [finite controller](parcae-fp8-v1/run.py).

## Full checkpoint and performance

The official checkpoint was independently rehashed and is reused read-only from
the existing local owner. [Verification](parcae-v1/parcae-official-reuse-verified.json).
The frozen96-case official BF16 CPU comparison completed:32 inputs × R1/R4/R8,
all416 dense recurrent states and96 final logits bitwise exact. The controller
and child returned0 naturally; the short-term reader hold is released and peer
weights remain untouched. [Raw results](parcae-v1/official-parcae-adapter-cpu-attempt1.json),
[verified summary](parcae-v1/official-parcae-adapter-cpu-summary.json). [Runner](parcae-v1/run_official_cpu.py),
[manifest](parcae-v1/official-cpu-manifest.json). This is not a serving benchmark.

| Model | BF16 tok/s | Strong quantized baseline tok/s | Candidate tok/s | Paired speedup | Speed change |
|---|---:|---:|---:|---:|---:|
| Parcae370M | Not measured | Not measured | Not measured | — | — |

Full quantization calibration, INT4/FP8 GPU qualification, NVFP4 native export, locked quality,
multiple batch sizes, high concurrency, mixed/open arrivals and formal paired
performance remain unrun. Retained numerical failures do not justify running a
full QAT search before native G0 is resolved.


## Boundary prefill repair after H20 execution

The first H20 FP8 case stopped before completing its request-lifecycle checks:
prelude returned six live states, but the core metadata described eight padded
rows. The error was retained at the original numerical budgets. The executor
now prepares core metadata with the actual prelude row count and records that
submitted size. Prelude and coda keep their existing unpadded boundary KV.
No rotary slicing or fabricated KV rows are introduced.

The added regression compares eager and static-buffer paths bitwise for
FP32/BF16, 3/6/7 live rows, single/multiple requests and two allocation lifetimes.
GPU repair validation is prepared separately; these CPU tests do not turn the
failed H20 attempt into a pass.

[New checks](boundary-prefill-v2/source-parcae-prefill-rows-cpu-attempt1.log): 12 passed. [Model/engine/KV regression](boundary-prefill-v2/source-parcae-prefill-rows-regression-attempt1.log): 97 passed, 3 GPU cases deselected; both test processes naturally exited 0.


## H20 boundary repair verification

The repaired packet completed with a mixed outcome at unchanged criteria.

| Check | Passed | Failed | Unrun | Outcome |
|---|---:|---:|---:|---|
| Huginn FP8 | 4 | 1 | 11 | DYN Graph off/on × norm off/on passed; first SH1 generated sequence differs from decoded FP8 reference |
| Parcae FP8 | 16 | 0 | 0 | DYN/SH1/ST2/LOOP × Graph off/on × norm off/on passed |
| Huginn INT4 | 2 | 0 | 0 | Graph off/on passed |
| Parcae INT4 | 1 | 1 | 2 | Native eager passed; compact with synchronous static buffers rejected by runtime |
| Separate Graph row-depth controls | 0 | 0 | 2 | Not reached after Huginn FP8 assertion |

The controller and all four children finished naturally (controller 1; children
1/0/0/1). The SH1 assertion and unsupported compact configuration remain failures;
no tolerances or sequence equality checks were relaxed. Successful checks cover
tiny native request lifecycles and encoded reload, not full-checkpoint quality
or serving performance. Failed children have per-case progress logs, not final
success JSON. The compact failure is configuration rejection, not an arithmetic
comparison result.

[Summary](h20-boundary-v2/h20-boundary-fix-v2-summary.json),
[natural receipt](h20-boundary-v2/h20-boundary-fix-v2-receipt.json),
[archive manifest](h20-boundary-v2/h20-boundary-fix-v2-archive-manifest.json) and
[completion](h20-boundary-v2/h20-boundary-fix-v2-complete.json) retain the outcome.
The 541,913-byte archive SHA256 is
`4cb2af439863ff7908ede8124d33ef6cb493ceb5e13248faded2f27c5d4bcdd5`.
All 37 files were verified offbox. Two archived temporary failed payloads
(733,942 bytes) were then removed; the unique archive and logs remain. Original
GPU0/IO lock identities were preserved and the finite window was handed back.

| Model | BF16 tok/s | Quantized tok/s | Paired speedup | Speed change |
|---|---:|---:|---:|---:|
| Huginn | Not measured | Not measured | — | — |
| Parcae | Not measured | Not measured | — | — |

Full-checkpoint native G0, locked quality and the multiple-batch/high-concurrency
E2E matrix remain pending. These checks establish no additional memory saving.
