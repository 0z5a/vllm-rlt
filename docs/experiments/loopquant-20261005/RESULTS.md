# Quantization experiment results — 2026-10-05

This is an in-progress numerical qualification. No model has completed G0–G3.
There are no measured E2E speedups yet; missing values are not zero speedup.
The [coverage queue](MODEL_COVERAGE.md) lists immutable model versions and the
remaining original-model, adapter, and serving work for each family.

The [Q0 schedule and pool](Q0_SCHEDULE.md) are registered. Its 8192-token updates
have not run; the earlier two 128-token gradient audits remain a separate result.

| Model | Baseline tok/s | Candidate tok/s | Paired speedup | Speed change | Lower 95% bound | Status |
|---|---:|---:|---:|---:|---:|---|
| Ouro-1.4B | — | — | — | — | — | G0 incomplete |
| Ouro-2.6B | — | — | — | — | — | Official CPU exact; GPU/E2E pending |
| Huginn-3.5B | — | — | — | — | — | Full official CPU96/96 exact; cached GPU pending |
| Nanbeige4.2-3B | — | — | — | — | — | Official CPU64/64 exact; full native GPU pending |

## Native FP8 correctness

RTX 5090/SM120, Torch2.12.1+cu130, CUDA13.0. Each 64×32 projection uses one 2048-byte packed matrix. Native scalar-scale GEMM includes unfused encoding and M-padding; this is not the strong deployment baseline.

| Input rows | Maximum absolute difference vs same quantized values |
|---:|---:|
| 1 | 0 |
| 17 | 0 |
| 32 | 0.001953125 |
| 64 | 0.000000119209 |
| 128 | 0 |

Stable scale-buffer updates followed by CUDA Graph replay passed. [Raw receipt](native-fp8-probe-attempt1.json).

## Full-model numerical failures retained

Official Ouro revision `574fa66cb8bf5abdc979642d01cf2b79b16bfab1`, runtime base `ecb1f8b505b7e831815b40aec3b4598619cca23a`. Weight bytes independently SHA256 verified. Fixed criterion: `atol=rtol=3e-4`.

| Attempt | Comparison | Result |
|---|---|---|
| [1](ouro-g0-attempt1.log) | FP32 vs initial FP32 functional oracle | Fail, smoke-7 R4; abs 0.000334829 |
| [2](ouro-g0-attempt2.log) | FP32 vs official operator-order FP32 oracle | Fail, smoke-7 R4; abs 0.000328362 |
| [Diagnostic](ouro-fp64-diagnostic-attempt1.json) | Only smoke-7 vs FP64 oracle | Pass unchanged threshold; not full G0 |
| [3](ouro-g0-attempt3.log) | FP32 vs FP64 oracle over the smoke set | Adapter failed at smoke-14 R3; abs 0.000332700 |

Attempt3 stopped after the first failed assertion; later inputs were not checked.
The original logs remain retained.

The next [complete audit](ouro-g0-attempt4b.json) collected all 32 inputs at R1–R4
in both FP32 and BF16. Its process naturally exited 0, but the fixed FP32
criterion failed in 36 of 128 combinations across the five checked tensors.
See the [aggregated counts](ouro-g0-attempt4b-summary.json). The first launch
had a [missing prompt path](ouro-g0-attempt4.log) and exited before GPU work;
the corrected launch has a separate [raw log](ouro-g0-attempt4b.log).

| BF16 comparison against official eager arithmetic | Maximum absolute error | Logit argmax disagreements / token positions |
|---|---:|---:|
| Dense adapter state | 0 (bitwise equal at all 128 points) | — |
| Dense adapter logits | 0 (bitwise equal at all 128 points) | 0 / 6,896 |
| Native torch attention state | 6.03125 | — |
| Native torch attention logits | 8.8984375 | 25 / 6,896 |

The native Torch backend computes attention scores and value accumulation in
FP32; the official eager path rounds scores and probabilities to BF16. These
are baseline differences requiring incremental-cache and deployment-backend
qualification, not a passed overall G0 or a quantization quality result. The
FP32 comparison had zero logit argmax disagreements but retained nonzero
fixed-tolerance failures (maximum logit error 0.00420481).

## CPU checks

[Latest full regression](full-cpu-nanbeige-native-attempt2.log): **492 passed, 29 skipped, 196 GPU
cases deselected**; [JUnit](full-cpu-nanbeige-native-attempt2.xml). Earlier full collection stalled
inside an existing macOS dependency load; both processes subsequently completed
naturally. The final run includes serving, GSM8K collection, and the latest
quantization changes. Skipped optional/dependency/GPU paths are not claimed as
passes. The earlier [core-only run](core-cpu-attempt1.log) is retained but does
not need to be added to the final count.

Checks include exact optimizer/RNG/data-position resumption, unchanged frozen
weights, and [BF16 eager oracle equality](bf16-adapter-cpu.log).
[GPTQ checks](gptq-cpu-attempt1.log) verify agreement with RTN for a diagonal
Hessian, consistent blocked/sequential compensation, and masked row/energy
accounting across loops. This signed static-group variant follows the
[GPTQ algorithm](https://arxiv.org/abs/2210.17323); no model-level quality or
native W4A16 performance has been measured.

Ruff lint and formatting pass. The [pre-commit attempt](pre-commit-attempt1.log)
reports that its wrapper is absent; no package was installed or upgraded.

Formal results require exported-model noninferiority and independent paired
E2E trials across C1/8/32/64, multiple batch limits, long/decode/mixed/open
workloads. Failed cohorts and captures during timing cannot establish a speedup.

After that run, FakeQuant's reference was corrected to preserve decoded values
in FP32 until the GEMM output conversion. This avoids an extra pre-matmul BF16
rounding absent from the native FP8 operation. The targeted
[38 CPU checks](loopquant-cpu-attempt3.log) pass; full GPU export parity remains
pending. Recurrent state and residual outputs remain BF16.

An explicit `attention_backend` selects eager or SDPA arithmetic for the adapter;
the Q0 teacher inherits that selection and export records it. The latest
[44 CPU checks](loopquant-cpu-trace-attempt1.log) cover both choices and native
trace metadata. A real tiny-engine cohort confirms that tracing preserves the
generated tokens and accounts for prefill/decode rows at every loop. A separate
reordered-depth/padding check excludes poisoned padding rows from statistics.

## Incremental cache and GEMM row shape

The [complete cache audit](ouro-cache-g0-attempt1.json) naturally exited 0. Both
Torch and Triton preserve every previous K/V prefix bitwise, keep depth tables
disjoint, and free every allocation. Each backend also has zero incremental
versus full-prefill logit argmax disagreements across 512 comparisons. The
registered `atol=rtol=0.02` element criterion nevertheless fails; process success
does not turn those numerical failures into passes.

| Native backend | Incremental/full maximum logit error | Failed logit comparisons | Logit argmax disagreements |
|---|---:|---:|---:|
| Torch | 6.046875 | 511 / 512 | 0 / 512 |
| Triton | 6.8984375 | 511 / 512 | 0 / 512 |

Two calibration smoke inputs with large differences were selected before the
[GEMM diagnostic](gemm-diagnostic-preregister.json). Neither variant changes the
criterion or the checkpoint. Per-token Linear is a diagnostic control, not a
performance baseline.

| Control | Incremental/full maximum logit error | Failed state / logit comparisons |
|---|---:|---:|
| [Disable BF16 reduced-precision reduction](ouro-gemm-fp32-batched-attempt1.json) | 6.34375 | 32 / 32; 32 / 32 |
| [Also fix Linear to single-token GEMM](ouro-gemm-fp32-per-token-attempt1.json) | 0 | 0 / 32; 0 / 32 |

The latter yields bitwise state and logit equality for these two inputs. It
isolates the Linear row-shape path as the cause of their incremental differences;
it is not a complete 32-input result. Both variants still fail the separate
prefill/SDPA comparison. PyTorch documents that batch shapes and BF16 reduction
choices can change floating-point results ([numerical accuracy](https://docs.pytorch.org/docs/2.12/notes/numerical_accuracy.html)).
The original failures remain visible and no threshold has been relaxed.

The unmodified official HF classes completed full-weight CPU comparisons using
existing Transformers 4.54.1 and independently verified weights; see below.

## Nanbeige adapter qualification

The [original-class comparison](nanbeige-official-tiny-attempt1.json) uses
unmodified source from Nanbeige4.2-3B revision
`b82e54bd609793562a75cbf9337970a93369eab5` with tiny random weights. All eight
FP32/BF16 × R1/R2 × loop-final-norm settings have bitwise equal valid-token
logits. This verifies the adapter's residual and normalization placement on
these inputs; the official full checkpoint remains untested.

[47 targeted CPU tests](loopquant-nanbeige-cpu-attempt1.log) pass, with one GPU
case deselected. They include native Nanbeige fixed-loop output/mask checks and
two scale updates through both recurrent steps, using the same architecture
and attention for the BF16 teacher. The first shared Q0 implementation and
Ouro checks remain covered after extracting common projection bookkeeping.

Q0 now reuses one FP32 decoded reference matrix per physical projection during training, so backward does not retain a separate decoded copy per loop. It is a nonpersistent training buffer and is excluded from checkpoints and deployment exports. [48 CPU checks](loopquant-reference-cache-cpu-attempt1.log) pass, including four-loop saved-storage accounting and exact cache reconstruction after loading the packed state. This is a training-memory change, not a native serving-memory claim.


## Unmodified official Ouro-1.4B class

The pinned Hugging Face model loaded through existing Transformers 4.54.1 and
our dense adapter agree **bitwise at all 128 state/logit comparisons**: 32
inputs × R1–R4, BF16 eager attention. The full CPU run naturally exited 0 in
3221.96 s. [Raw receipt](official-ouro-cpu-attempt1.json). This confirms the
adapter against the original class, independently of the earlier functional
oracle. It does not qualify native attention, model quality after quantization,
or serving throughput. Its completed local duplicate weight was removed;
remote weights needed by later experiments remain retained.

## Complete per-token GEMM control

The [full 32-input control](ouro-gemm-full-per-token-attempt1.json) extends the
earlier two-input result. Disabling BF16 reduced-precision GEMM reduction and
using one token per Linear call removes most incremental/full-prefill drift,
but **does not remove all of it**: 53/512 state and 66/512 logit comparisons fail
the unchanged `atol=rtol=0.02`; maximum logit difference is 3.734375. All 512
logit argmax checks agree. Prior KV prefixes remain unchanged and allocations
are released. Prefill against SDPA still fails at all 128 points. Therefore the
two-input bitwise result must not be generalized to the full smoke set.

## Native rowwise FP8

The installed Torch 2.12.1+cu130 `torch._scaled_mm` accepts rowwise A scales
`[M,1]` and columnwise B scales `[1,N]` on RTX 5090/SM120. All 20 combinations
of M=1/17/32/64/128 and (K,N)=(64,32), (2048,2048), (2048,5632),
(5632,2048) pass the registered 0.02 absolute/relative encoding comparison;
maximum absolute difference is 0.015625. Two FakeQuant/native scale checks
also pass. [Raw receipt](rowwise-fp8-probe-attempt1.json). Rowwise graph replay,
producer fusion, model-level quality and performance remain untested.

## Four-window native closed-loop calibration pilot

The pilot completed four 512-token calibration windows, C4, scheduler token
budget 128, R4, and 32 generated tokens per request with native Triton
attention. It produced 69,142 trace calls and 2,896 statistics groups, with zero
failed requests and zero KV allocations after drain. Every loop/projection
received 2,048 prefill and 124 decode rows; the latter counts the 31 generated
tokens actually fed back per request. [Receipt and artifact hashes](ouro-calibration-trace-pilot-summary.json).

| Layer-0 q-projection input | Loop 0 | Loop 1 | Loop 2 | Loop 3 |
|---|---:|---:|---:|---:|
| Prefill amax | 2.125 | 6.625 | 6.6875 | 6.625 |
| Decode amax | 1.71875 | 3.34375 | 3.015625 | 2.921875 |
| Prefill RMS | 0.132093 | 0.142895 | 0.147032 | 0.149037 |
| Decode RMS | 0.132585 | 0.144288 | 0.147717 | 0.149526 |

These are pilot statistics, not the required 128-window calibration or scale
selection on dev data. Raw compressed statistics (65,519,837 bytes) and trace
(1,240,886 bytes) are retained locally with independent SHA checks. The
instrumented 247.98 s process interval is not serving performance.

The next collector accumulates statistics on the activation device and reuses
prepared row indices, removing per-update scalar/boolean-index synchronization.
Reservoir sampling remains uniform and bounded at 4,096 observations; its
seed and device are recorded because CPU/CUDA RNG streams differ. Nonfinite
CUDA data causes rejection when statistics or policies are materialized.
[49 CPU checks](loopquant-device-stats-cpu-attempt1.log) pass; two GPU checks
were deselected. GPU parity now passes as detailed below; formal performance remains pending.

Huginn's separately based adapter is in
[draft 0z5a/vllm-rlt#10](https://github.com/0z5a/vllm-rlt/pull/10), on a frozen
integration of native Huginn PR82 and this quantization harness. Its eight
original-class tiny comparisons are bitwise exact and 59 targeted CPU checks
pass. Its full-checkpoint CPU R8/16/32 comparison is running; its E2E matrix is pending.

The first fused-producer probe passed six cases, then failed the unchanged exact
scale comparison at M17/K64 under DYN (four FP32 scales differ by at most
9.313225746154785e-10). Packed activation values are identical. The retained
sample isolates RN division versus Torch's rounded reciprocal multiplication.
[Raw failure](fused-fp8-probe-attempt1.log), [diagnosis and sample hash](fused-fp8-failure-analysis.json).
The implementation now matches the reciprocal product; the full repeat passes
as detailed below. No threshold was relaxed. Native model integration and a reference
NVFP4 block16/global-scale encoder are also staged, without a performance claim.
Shared scratch uses one allocation per feature width, with row-bucket views,
rather than allocating a separate buffer for every batch shape.

## Mixed-depth metadata and fixed-depth workload checks

Prepared batches now carry an explicit device `loop_ids` vector through ordinary
staging, padded workspaces, asynchronous routing, packed-prefill expansion and
CUDA Graph inputs. Padded rows use -1; graph inputs are refreshed on every replay.
The workspace memory estimate includes the new 8-byte-per-row vector.
[Full CPU regression](full-cpu-loop-metadata-attempt1.log) completed with
**466 passed, 29 skipped, 114 GPU cases deselected**. Added GPU graph checks
make model output depend on the loop vector to detect stale replay metadata;
those two new cases have not run.

A fixed-depth cohort now rejects mismatched prefill/decode depths before any
request is admitted. The engine's prefill uses the model configuration, so
setting only request `min_loops=max_loops` is insufficient for R1–R3 curves.
The registered model recurrence must also match R. Previous R4 receipts are
unaffected. [49 targeted CPU checks](loopquant-fixed-depth-cpu-attempt1.log)
pass with 29 GPU cases deselected, including a real R2 cohort and rejection of
R4-prefill/R2-decode mislabeled as fixed R2.

## Additional full-model and device results

Ouro-2.6B's unmodified pinned official class and adapter agree bitwise at all
128 BF16 eager state/logit comparisons (32 inputs × R1–R4); process natural
exit0, 3372.82s. [Receipt](official-ouro-2.6b-cpu-attempt1.json).

The complete row-shape control fixes both Linear and RMSNorm to one row per
call. All **512/512 incremental/full comparisons are bitwise exact**, with
unchanged prior KV prefixes, disjoint depth tables and zero remaining
allocations. Separate prefill/SDPA comparisons still fail at all128 points,
max logit difference10.4140625 with20 argmax disagreements. The diagnostic
controls establish the source of shape-dependent arithmetic; they are not a
serving baseline or an overall G0 pass. [Receipt](ouro-row-shape-control-attempt1.json).

Device-local statistics pass K64/2048/5632 tests, reproduce their seeded
reservoirs, reject active nonfinite inputs and show no item/local-scalar/nonzero
operation in the profiled updates. [Receipt](device-stats-probe-attempt1.json).
Replaying the original four-window calibration gives exact2896 group keys,
counts, amax, activation scales, token traces and loop depths. Maximum RMS
and channel-energy differences are4.44e-16 and1.14e-13, below the preregistered
criteria. Device-specific reservoir percentiles are intentionally not compared.
[Parity](calibration-device-parity-attempt1.json).

The five-child window has actual return codes0/0/0/1/0; controller/SSH exit0
does not hide the FP8 failure. All14 collected files were independently hashed.
[Window receipt](gpu-window-v2-receipt.json), [completion](gpu-window-v2-complete.json).
The instrumented replay took59.07s, but runs overlapped peer non-timed work and
are not paired throughput trials, so no E2E speedup is inferred.

NVFP4 CPU checks cover all256 packed bytes (including signed zero), nearest-even
midpoints, tail padding, fixed scale reuse and independent swizzle coordinates.
[Three checks pass](nvfp4-format-cpu-attempt1.log). Native installed NVFP4 ABI now passes60 cases as below. The FP8 model/graph
check has a retained constructor failure awaiting corrected-mode repetition.

## Arrival traces and paired task metrics

A native engine trace now supports closed refill cohorts and immutable open
arrival times, varying prompt/output lengths and per-request decode depth.
Arrival latency includes submission delay while the engine is busy; queue
samples include offered but not yet submitted requests. Mixed-depth records
keep the model-configured prefill depth for the first output separately from
later decode depths. Fixed-depth cohorts still require both to match.
The full CPU regression passes476 tests (29 skipped,128 GPU deselected), with
[raw log](full-cpu-arrivals-attempt1.log). The subsequent explicit stream-drain
and finite-SLO checks pass the [50-test subset](arrival-quality-cpu-attempt2.log).

Latency reports include p50/p95 TTFT, TPOT and observable single-token ITL;
batched deliveries retain counts rather than inventing token timestamps.
Goodput keeps failed requests in its denominator and can use a preregistered
steady window. Scheduler shape histograms report host effective/submitted
rows; they are not kernel-profiler measurements. Question-paired bootstrap
accuracy drops complement document-paired PPL intervals. These are tested
measurement facilities; GPU open-arrival and HTTP results remain pending.

## Native formats and full-checkpoint gradients

On the same RTX5090 runtime, all **120 fused FP8 cases and three CUDA Graph
replays pass** the unchanged criteria. Encoded inputs match the independent
reference exactly in every case, including optional RMSNorm. DYN/SH1/ST2,
M1/17/32/64/128 and the four registered K/N shapes are covered. Worst absolute
GEMM error0.25 occurs with large outliers and passes `atol=rtol=0.02`; this is
not an absolute-only0.02 claim. [Receipt](fused-fp8-probe-attempt2.json).

NVFP4's installed native block16/global-scale ABI passes all60 shapes/policies
against independently decoded packed operands, max abs0.015625. Policies
cover dynamic, fixed-global and fixed-global/local scales. The current encoder
is an ordinary Torch reference; no fused NVFP4 producer or speedup is claimed.
[Receipt](nvfp4-probe-attempt1.json).

Ouro1.4B's actual checkpoint completes two128-valid-token, R4 gradient audits:
168 per-loop scale groups have finite gradients/updates; all BF16 weights keep
`grad=None`, storage identities and tensor versions. Gradient norms are0.04456
and0.04998, peak allocation13,085,885,952 bytes. This is **not** the prescribed
8192-token/update20+200 Q0 training run. [Receipt](ouro-gradient-audit-attempt1.json).

The tiny serving probe failed before generation because async scheduling
requires an explicit supported exit mode. The [constructor error](fp8-serving-probe-attempt1.log)
is retained; the test now specifies `ouro_delayed` while keeping fixed requested
depths and all12 registered cases unchanged. It awaits another finite GPU run.
Window3 actual child exits are0/1/0/0, controller/SSH0, with all9 files independently
hashed. [Receipt](gpu-window-v3-receipt.json), [handoff](gpu-window-v3-complete.json).

The Ouro2.6B local duplicate weight was removed only after original-class CPU
completion, peer upload/hash completion, independent verification of all nine
owned remote hardlinks and an empty local reader check. It frees5,336,011,242
bytes; remote inputs still needed for experiments remain retained.
[Cleanup](official-ouro-2.6b-cpu-cleanup.json).

## Fused native FP8 engine qualification

The corrected [original 12-case repeat](fp8-serving-probe-attempt2.json) passes DYN/SH1/ST2 × Graph on/off × fused RMSNorm on/off. Each case runs the same three mixed-depth requests twice, checking token/depth parity against decoded FP8 reference arithmetic, request-ID reuse, complete KV drain and unchanged packed-weight pointers. Graph cases each capture 3 shapes and replay 24 times. There is one packed matrix per physical projection (14 in this tiny model). The child, controller and SSH all naturally exit 0; [completion receipt](fp8-serving-repeat-v2-complete.json). This establishes tiny-model serving correctness, not full-model quality or E2E throughput. The earlier constructor failure is retained.

The immutable-environment checkpoint reader now preserves tied parameter identity across shards and persistent buffer dtypes, as required by Huginn. [47 targeted CPU checks](checkpoint-alias-cpu-attempt1.log) pass, including rejection of inconsistent tied values. Nanbeige's full 12-file download is verified; its full original-class CPU comparison started after Huginn naturally exited, avoiding simultaneous large model loads.

## Native deployment and quality plumbing

The latest [full CPU regression](full-cpu-native-deployment-attempt1.log) passes **482 tests**, with 29 skipped and 164 GPU tests deselected. Fused FP8 export preserves the encoded matrices, their column-major strides, per-row weight-scale ABI and protected BF16 parameters; reload reconstructs the same fused producers without decoding or requantizing W. The new 16-case GPU export/engine roundtrip matrix remains unrun. Native teacher-forced NLL now traverses the actual recurrent core and cache; CPU checks match Ouro/Nanbeige dense references and verify cache release between windows.

The INT4 adapter converts our signed low-nibble-first checkpoint to the installed Torch tinygemm ABI. It explicitly rounds group scales to BF16 and pads activation width where the registered group requires it; the 20-case GPU matrix is pending. No W4A16 speedup is claimed. ABI arithmetic follows the pinned [Torch 2.12.1 CUDA implementation](https://github.com/pytorch/pytorch/blob/v2.12.1/aten/src/ATen/native/cuda/int4mm.cu), including BF16 dequantization.

Nanbeige's default AutoTokenizer route requests an unavailable SentencePiece conversion. Reading its pinned official fast tokenizer JSON directly preserves the existing BOS postprocessor; [all 32 fixed texts encode successfully](nanbeige-tokenizer-json-attempt1.json), totaling 1,756 tokens. The full original model comparison remains queued, using identical IDs for both paths.

## External algorithm track

The [LoopQ reproduction audit](EXTERNAL_BASELINE.md) records `not_reproduced`: no official code revision or exported artifact was identified in the pinned paper, author pages/repositories, or repository-name search as of2026-10-05. Published LoopQ and this strict one-packed-weight track have different transform, weight-sharing and online-operator contracts. No numerical or performance comparison against original LoopQ is claimed.

## Fused-producer scale training

An explicit pre-optimizer binding can share activation log-scales across Q/K/V or gate/up consumers. This keeps the learned scale policy representable by a single fused input producer; it does not silently tie already-divergent scales. The [47 targeted CPU checks](shared-scale-training-cpu-attempt1.log) pass, including equality of the shared gradient to the sum of independent consumer gradients through all four loops, unchanged frozen-weight gradients and rejection of mismatched initial scales. The original168-group full-model gradient audit remains an independent, unbound diagnostic; this new spatial policy has not yet run full-model QAT.

## Native GPTQ row-budget collection

The native hook collector now accumulates first-loop, all-loop, and matched-row
Hessians. The matched policy selects one loop per request/token position using
a fixed seeded hash, so complete fixed-depth execution has exactly the same
row budget as first-loop calibration. QKV and gate/up reuse their common input
Gram matrices; each physical weight remains independently packed once.

The full CPU regression includes reordered mixed-depth rows, poisoned padding,
per-loop counts/energy, incomplete-budget rejection, and unchanged native
generation/KV drain. An initial test used an incorrect cache allocation method;
its [failure](gptq-native-row-budget-cpu-attempt1.log) and the corrected
[targeted checks](gptq-native-row-budget-cpu-attempt2.log) are retained.
Full-checkpoint Hessian collection, GPTQ packing and native W4A16 quality remain
unrun. No speedup is inferred from these tests.

## Native deployment qualification

The independently archived [short-window receipt](native-deployment-v1-receipt.json)
has two natural child exits 0 and controller/SSH exit 0. Frozen source
`72b8634e0d4029fc8106335c2c7861e40eb6c193`, RTX5090/SM120, Torch2.12.1+cu130,
CUDA13.0. No package changes were made.

| Qualification | Cases | Result | Comparison |
|---|---:|---|---|
| Native FP8 export/reload | 16 | Pass | DYN/SH1/ST2/LOOP × Graph on/off × fused norm on/off; encoded state bytes unchanged |
| Mixed-depth native serving after reload | 16 × 6 requests | Pass | Same generated tokens/depths as decoded reference; repeated IDs, full KV drain and stable single packed weight |
| Native INT4 W4A16 | 20 | Pass | M=1/17/32/64/128 across four K/N shapes; unchanged atol=rtol=0.02 vs the same BF16-decoded codes/scales |

Each Graph case captured three buckets and replayed 24 times. Every tiny core
had 14 physical packed matrices (98,304 weight bytes total). All 16 exported
test payloads were removed after successful reload and execution; their
manifests and raw evidence remain. [FP8 records](native-fp8-export-attempt1.json).

The INT4 shapes include K/N=128/32, 2048/2048, 2048/5632 and 5280/32. The last
shape explicitly pads K to 5376. The maximum absolute difference is 0.03125,
with every element satisfying the combined absolute/relative criterion. The
backend uses Torch's installed `_convert_weight_to_int4pack` and
`_weight_int4pack_mm`; group scales round to BF16 as recorded in the ABI.
[INT4 records](native-int4-attempt1.json).

These are numerical and lifecycle checks. Full-checkpoint exported quality,
GPTQ/QAT outcomes and E2E throughput remain unmeasured.

## Nanbeige native deployment preparation

The fused decoder preserves Nanbeige's two-norm residual order; only Ouro uses
the two additional sandwich norms. Scratch covers every physical projection
input width. Nanbeige's attention output is 6144 channels, distinct from its
3072-wide hidden state and10752-wide MLP. Native exports now record and reload
either model family while preserving the existing frozen Ouro entry points.

When `skip_loop_final_norm=True`, Nanbeige still needs its final readout norm.
The native coda now applies it once, matching the already-qualified dense
adapter. The pinned full checkpoint sets this option false; the true branch is
a separate architecture check. Native NLL tests cover both settings.

The first full CPU run found an old scratch-accounting expectation that omitted
the fixture's distinct32-wide attention input. Its [failure](full-cpu-nanbeige-native-attempt1.log)
is retained; the corrected test explicitly checks all32/64/128 widths and shared
bucket storage. The latest full suite passes492 tests. New32 Nanbeige GPU
export/Graph/norm/readout combinations and an Ouro regression repeat are
prepared but unrun; earlier16-case GPU evidence remains tied to source72b8634.

## Full Ouro1.4B calibration

The frozen 128-window run completed naturally on GPU0 with 128/128 requests,
4,096 generated tokens, no failed requests and zero KV blocks after drain.
It recorded 2,199,150 activation calls, summarized as 2,896 groups across
362 physical observation sites, four loop depths and both execution phases.
Each group retains at most 4,096 sampled elements plus exact counts, amax and
channel energy. [Summary and file hashes](ouro-full-calibration-summary.json).

The first query projection observes 62,993 prompt rows per loop and 3,968 decode
rows per loop. The latter is 128 × 31: the final generated token is delivered
but does not re-enter the core. This is consistent with 128 × 32 output tokens.
All 168 physical core projections have SH1, ST2 and LOOP tables for each phase.
The registered ST2 boundary remains 2 and prefill/decode use their shared
maximum in deployment. No clipping search or dev-based choice was made.
[Statistics](ouro-full-calibration-statistics-summary.json).

The six raw files were archived and independently hashed locally; their archive
SHA256 is `8d0a44c8db014f8afc24eb2bed0d2d9770e6c3e11011991a2e05980a7cb90433`.
The child, controller and SSH returned 0, and fresh reader, GPU0 compute and
original-lock checks confirmed [complete handoff](ouro-split-calibration-complete.json).
This instrumented run is calibration evidence, not an E2E speed measurement.

## First full-model engine pilot

The [eight-arm Ouro1.4B pilot](PILOT_E2E.md) completed64 requests and2,048
output tokens per arm with no failures, residual KV, timed Graph captures or
fallbacks. Its single-trial speed comparison is below; formal G2/G3 remain
not run. All FP8 policies are slower than BF16 in this smoke workload.

| Batch token cap | Policy | Output tok/s | Vs BF16 | Speed change vs BF16 | Vs DYN | Speed change vs DYN | TPOT p95 (ms) | SLO coverage |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 128 | BF16 | 180.608 | 1.0000× | +0.00% | 1.0795× | +7.95% | 104.72 | 85.94% |
| 128 | DYN | 167.313 | 0.9264× | -7.36% | 1.0000× | +0.00% | 165.58 | 28.12% |
| 128 | SH1 | 166.168 | 0.9200× | -8.00% | 0.9932× | -0.68% | 166.70 | 28.12% |
| 128 | ST2 | 167.968 | 0.9300× | -7.00% | 1.0039× | +0.39% | 165.59 | 28.12% |
| 512 | BF16 | 387.146 | 1.0000× | +0.00% | 1.0651× | +6.51% | 77.47 | 100.00% |
| 512 | DYN | 363.494 | 0.9389× | -6.11% | 1.0000× | +0.00% | 82.31 | 100.00% |
| 512 | SH1 | 361.615 | 0.9341× | -6.59% | 0.9948× | -0.52% | 82.94 | 100.00% |
| 512 | ST2 | 359.988 | 0.9299× | -7.01% | 0.9904× | -0.96% | 83.55 | 100.00% |

## INT4 model artifact preparation

The INT4 artifact keeps signed low-nibble portable codes, explicitly rounded
BF16 group scales and unchanged protected weights. Loading converts each
physical core matrix once to Torch tinygemm storage; the temporary portable
payload is released, with BF16 activations/state/KV retained. RTN and supplied
first/all/matched GPTQ matrices share this path. Two CPU model fixtures verify
codes, shapes and protected weights; four mixed-depth/Graph native model cases
are prepared but not run. Latest full CPU regression:494 passed,29 skipped,
200 GPU cases deselected. [Log](full-cpu-int4-export-attempt1.log).

## Native model qualification and initial dev quality

The updated Ouro repeat passes16/16 tiny GPU cases, and Nanbeige passes32/32
across all four FP8 policies, Graph, norm fusion and both loop-final-norm
settings. Exported state bytes, mixed-depth token/depth output, repeated
request IDs, packed storage identities and full KV drain agree. All48 tiny
export tensors were removed after success; [manifests and raw results](native-models-v2/native-models-v2-file-manifest.json)
remain. Native INT4 model artifacts also pass all four Ouro/Nanbeige ×
Graph-on/off cases, each with six repeated-ID requests and exact reload state.
Their four temporary payloads were removed after successful validation.
[INT4 records](quant-quality-v1/native-int4-models-v1.json).

The five full Ouro1.4B [native dev quality arms](DEV_QUALITY.md) completed
128 windows each. These are initial-policy dev results, not locked G2.

| Policy | Dev PPL | PPL ratio vs BF16 | PPL change | One-sided 95% upper ratio |
|---|---:|---:|---:|---:|
| BF16 | 11.559050 | 1.000000 | +0.000% | 1.000000 |
| DYN | 11.649150 | 1.007795 | +0.779% | 1.009332 |
| SH1 | 11.640786 | 1.007071 | +0.707% | 1.008543 |
| ST2 | 11.650949 | 1.007950 | +0.795% | 1.009565 |
| LOOP | 11.652965 | 1.008125 | +0.812% | 1.009776 |

## Complete official Huginn CPU comparison

[All96 comparisons](official-huginn-cpu-attempt1.json) passed against the unmodified pinned official full checkpoint: 32 fixed inputs × R8/16/32, CPU BF16 SDPA. Every one of1,792 recurrent states and96 logit tensors is bitwise exact, with an explicit identical seeded initial state. [Verified summary](official-huginn-cpu-summary.json) records source202987db, runner/model hashes and input coverage. Cached GPU semantics, quantized quality and serving performance remain pending; this is not a full G0–G3 pass. The adapter and its reproducible runner are reviewed in the separately based [Huginn draft](https://github.com/0z5a/vllm-rlt/pull/10).

## Complete official Nanbeige CPU comparison

[All 64 comparisons](official-nanbeige-cpu-attempt1.json) pass against the pinned complete BF16 eager checkpoint: 32 fixed inputs × R1/R2. All 96 recurrent states and 64 logit tensors are bitwise exact. The [verified summary](official-nanbeige-cpu-summary.json) fixes the source, runner, model revision and 1,756 input tokens; the reproducible runner uses the official fast tokenizer JSON. Controller and child exited naturally with code 0. This establishes original-class CPU adapter parity; full native GPU semantics, quantized quality and formal serving measurements remain pending.

After all local readers exited and the shared future-read hold was released, both local duplicate weight shards were rehashed against the verified node copy and removed: **8,339,624,720 bytes**. [Cleanup receipt](official-nanbeige-cpu-cleanup.json) retains the exact two filenames and hashes. Configs, tokenizer, author code, raw results and node weights remain available.
