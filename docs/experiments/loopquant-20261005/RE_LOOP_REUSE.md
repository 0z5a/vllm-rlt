# Quantization evidence available to Re:Loop

This handoff reuses existing code and measurements. It does not establish a new G0–G3,
quality or speed claim. The full original all-model execution goal remains active.

## Source and data

| Component | Exact local source | Scope |
|---|---|---|
| FP8 row producers and installed rowwise GEMM | `source/loopquant/fp8_kernels.py`, `source/loopquant/fused_fp8.py` | DYN/SH1/ST2/LOOP, per-row scales, fused QKV and gate/up, represented-byte checks |
| Native FP8 model execution/export | `source/loopquant/serving.py`, `source/loopquant/native_export.py` | Ouro/Nanbeige protected BF16 operations, persistent Graph workspaces, row-depth metadata, exact packed reload |
| Immutable-environment checkpoint loader | `source/loopquant/checkpoint.py` | PyTorch-only safetensors streaming, persistent buffer dtype and tied parameter identity |
| Native INT4 | `source/loopquant/int4_backend.py`, `source/loopquant/int4_export.py` | Packed core projections; tiny/native export correctness, full quality still pending |
| Family adapters | `source-families` at `d180fea3ab8af6efc2df504781c575fecff7413a` | HRM prefix/H/L and LoopFormer clock/AdaLN; full CPU recurrence qualification and tiny CUDA |
| Huginn stack | `source-huginn` at `0b2926396bbea10520e3f5f88ba591e29b0f7424` | Boundary row-count repair and FP8/INT4; SH1 generated-output mismatch retained |
| Parcae stack | `source-parcae` at `a8d79443ebf075aeaa37d6e22ed6608e56c6db16` | Depth-weighted logits, shared KV representation; compact/static rejection retained |

All paths are under `/Users/0z5a/Documents/infra/loopquant-20261005`.
Main source HEAD is `55b4f22e2bcc9a4c0f2df3c66f62991d4d940522`,
[open fork Draft28](https://github.com/0z5a/vllm-rlt/pull/28).
The [index](re-loop-reuse-index.json) records exact native-file and completed-summary SHA256s.
Use each result's own frozen runtime source for reproduction; later report-only HEADs are not
substitutes for that source provenance.

## Existing full-weight engine pilot

Model is **ByteDance/Ouro-1.4B Base@574fa66cb8bf5abdc979642d01cf2b79b16bfab1**,
RTX5090, fixed R4, C32, prompt256/output32, 64 requests per arm, same20GiB BF16 KV budget.
These results cannot be relabeled as Ouro Thinking, another checkpoint, H20 or a formal paired steady-state test.

| Batch token cap | BF16 tok/s | DYN tok/s | SH1 tok/s | SH1 vs BF16 | Saved PyTorch allocated GiB |
|---:|---:|---:|---:|---:|---:|
| 128 | 180.608 | 167.313 | 166.168 | -8.00% | 1.0962 |
| 512 | 387.146 | 363.494 | 361.615 | -6.59% | 1.0763 |

SH1/ST2 do not beat dynamic FP8 materially in this smoke workload; FP8 is slower than BF16.
Zero failures, residual KV, timed Graph captures and fallbacks do not establish accuracy.
Detailed inputs, all8 arms and raw metrics are in [PILOT_E2E.md](PILOT_E2E.md),
`evidence/ouro-pilot-v1-summary.json` and `evidence/ouro-split-pilot-archive/`.
Avoid repeating this same8-arm FP8 pilot merely to recover its already saved data.

Dev PPL: BF1611.5591 vs SH111.6408; paired one-sided95% ratio upper bound1.008543.
[Dev results](DEV_QUALITY.md) are unlocked, and no Wiki/GSM8K/ARC/long-context gate is passed.
Three Nsight profiles and15 stats reports are saved: the producer is about4% of GPU kernel-duration
sum. This is not a critical-path or removable-cost estimate; [G1 remains inconclusive](COST_PROFILE.md).

## Completed CUDA qualifications and failures

| Evidence | Scope and observed result | Exact raw summary |
|---|---|---|
| H20 primitives | 72 represented FP8 producer/GEMM comparisons +3 Graph checks pass; all encoded inputs exact, combined budgets only | `evidence/h20-fp8-primitives-v1-complete.json` |
| H20 native families v1 | HRM16 + LoopFormer16 tiny FP8 cases pass; Huginn/Parcae stop on boundary row metadata error | `evidence/h20-native-families-v1-complete.json` |
| H20 repaired boundary v2 |23 native pass /2 fail /13 native +2 controls NOT_RUN: ParcaeFP8 all16, HuginnINT4 both2, HuginnDYN4 pass; SH1 sequence mismatch and compact/static INT4 rejection retained | `evidence/h20-boundary-fix-v2-complete.json` |
| Full LF G0v2 attempt | Mirror config302→403 before any model bytes;64 full GPU checks NOT_RUN; all12 raw files verified offbox and window handed back | `evidence/h20-loopformer-native-g0-v2-complete.json` |

No row is a full-checkpoint quantized quality or G3 serving qualification.
All unique H20 raw archives are already saved locally and independently verified;
failed tiny tensors are removed only after archival. Native NVFP4 F2 on SM90 is not qualified.
Ouro's earlier full BF16 elementwise G0 failures and tiny eager argmax discrepancy remain;
math-SDPA exactness is an internal control, not permission to replace the eager/original reference.
[Current full-weight diagnostic queue](NATIVE_G0_QUEUE.md).

## Resource handoff

Direct human deadline in LoopKV message`01a10d98-59cc-7712-8fd9-b4db54d3dcfd`:
archive before **2026-10-06 08:30 Asia/Shanghai**, H20 shutdown/deletion at **08:44:10**.
Only `region-42.seetacloud.com:33885`, GPU`GPU-edf64e5f-21ef-a1f2-6601-e8620b5664ff` is in that scope.
A fresh read-only node inventory confirms all prior Quant controller/children/finalizer PIDs absent.
No Quant CUDA/model work is active, no original GPU/IO lock is held, and LF peer transfer reader is inactive.
The transient own SSH master has a600-second idle expiry; no process is killed or signaled.
Only separate, deadline-fitting future grants may admit queued work, behind the verified FlashNS/LoopCD priority.
All originals remain preserved offbox; the old5090 INT4 child31380 staysUNKNOWN and is neither migrated nor restarted.
Quant has no verified cloud instanceID/control-plane entry; only the verified SSH endpoint and GPU identity.
