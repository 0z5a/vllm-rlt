# Quantization format, ownership and hardware audit

The October 5 cross-project execution document extends this work through its
Loop Quant Q0–Q3 contracts. These labels are separate from the original plan's
Q0 scale-training schedule. Existing Loop KV work owns aliasing and page
lifetimes; this quantization branch currently retains BF16 state and KV.

The [baseline manifest](associative-io-v1/baseline-manifest.json) pins actual
source files at `8db31484dbbd8ca2e62974460f1cbb092181a1fa`, their hashes,
format/scale axes and the execution-document hash. No proposed document API is
presented as an existing runtime entrypoint.

| Contract | Existing implementation | Evidence or remaining work |
|---|---|---|
| Q0: explicit represented format | `quantizers.py`, `nvfp4.py`, `int4_backend.py` | FP8 row scales are independent of K; INT4/NVFP4 block scales stay inside their native contraction. Portable and execution layouts are recorded separately. |
| Q1: producer conversion | `fused_fp8.py`, `fp8_kernels.py`, `fused_nvfp4.py` | FP8 fuses row reduction/scale selection/cast and optional affine RMSNorm. Quantized KV publication is unimplemented; no KV-memory reduction is claimed. |
| Q2: one packed copy | `native_export.py`, `int4_export.py`, native linear buffers | Encoded reload and stable pointers are tested. Payload and scales share one hashed artifact. Q0 freezes model weights; general optimizer weight-version invalidation/repacking remains unimplemented. |
| Q3: architecture and represented-value comparison | Existing Torch GEMM, Triton producers and frozen probes | RTX5090 evidence is separate from the new H20 run. There is no B200/TMEM measurement or custom WGMMA implementation here. |

FP8 weight scaling uses one scalar per physical matrix, expanded to the ABI's
output-column scale vector. Activation scales are dynamic per row or selected
from the registered outer-loop stage. These scales never move across softmax,
SiLU or sigmoid. INT4's signed portable bytes are translated to the installed
tinygemm ABI with BF16 group-scale rounding. NVFP4 has an independent block16
scale contract; no global epilogue scalar substitutes for its K-block scales.

Memory reporting separates packed weight bytes, scale bytes, scratch, retained
BF16 tensors and the preallocated KV pool. The existing Ouro pilot saves
1.076–1.096 GiB of peak allocated memory with unchanged 20 GiB BF16 KV; it is
slower than BF16. Those observations establish neither copy elimination nor
capacity-qualified throughput.

## H20 qualification

The new node is one NVIDIA H20, compute capability 9.0, driver 580.105.08,
Torch 2.12.1+cu130 and Triton 3.7.1. [Read-only inventory](associative-io-v1/h20-inventory.json).
The existing Python is used without package installation or upgrade; artifacts
use local XFS under the task's `0z5a` directory. This node is distinct from the
old two-RTX5090 host; the interrupted INT4 job there remains of unknown status.

[The first H20 packet](associative-io-v1/h20-packet-manifest.json) selects twelve
physical shapes from the existing FP8 probe: four K/N pairs and M=1/17/128,
three scale policies, norm fusion off/on, and three Graph replay updates.
It checks encoded values, scales, represented-value GEMM, outliers, zero rows,
padding and pointer stability at the existing numerical budgets. The controller
holds the original GPU0 and IO locks until its child exits naturally.

| H20 result | Status |
|---|---|
| 72 producer/GEMM cases and 3 Graph replays | All pass; both processes exited naturally 0 |
| Full-checkpoint quality | Not measured |
| Fixed-workload speed and capacity | Not measured |
| NVFP4 or B200-specific path | Not submitted on H20 |

[Raw cases](associative-io-v1/h20-raw.json),
[summary](associative-io-v1/h20-summary.json),
[natural process receipt](associative-io-v1/h20-receipt.json) and
[offbox archive manifest](associative-io-v1/h20-archive-manifest.json) are retained.
All five file sizes and hashes match the independently copied archive. The
temporary tensor payload was removed after verification; its archived copy
remains. [Completion record](associative-io-v1/h20-complete.json).
At handoff a separate GPU job was present, so the node was not reported idle
and the next quantization packet was not submitted. This qualification provides
no throughput or full-model quality result.

Formal timing will keep non-profiled paired runs separate from profiling and
will account for conversion, scale metadata, packing and all participating
streams. The original quality and E2E gates remain in force.


## H20 family deployment checks

The next finite packet ran four sequential children under the same original
GPU0/IO locks. Its [manifest](h20-native-v1/manifest.json) freezes the three
source revisions; numerical budgets are unchanged.

| Family | Native FP8 cases | Process result | Full-checkpoint E2E speedup |
|---|---|---|---|
| HRM | 16 pass | Natural exit 0 | Not measured |
| LoopFormer | 16 pass | Natural exit 0 | Not measured |
| Huginn | First case failed; 15 unrun | Natural exit 1 | Not measured |
| Parcae | First case failed; 15 unrun | Natural exit 1 | Not measured |

The two additional Graph row-depth controls were also unrun. The complete
controller exited 1 after waiting for all four children. The two failures
exposed boundary-prefill metadata that used a padded workspace length while
the model's prelude returned only live rows. They are shape errors, not
relaxed numerical comparisons. A separate repair sets core metadata from the
actual hidden row count; repair verification is not counted in this attempt.

[Summary](h20-native-v1/h20-native-families-v1-summary.json),
[Huginn failure](h20-native-v1/h20-native-families-v1-huginn.log),
[Parcae failure](h20-native-v1/h20-native-families-v1-parcae.log) and
[completion receipt](h20-native-v1/h20-native-families-v1-complete.json) preserve
the mixed outcome. All 46 archived files were independently hash-verified
offbox. Temporary tensor payloads were cleaned after preserving the archive.
Both locks released naturally; the finite window was explicitly handed back.


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
