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

Full quantization calibration, INT4 GPU qualification, FP8/NVFP4 native export, locked quality,
multiple batch sizes, high concurrency, mixed/open arrivals and formal paired
performance remain unrun. Retained numerical failures do not justify running a
full QAT search before native G0 is resolved.
