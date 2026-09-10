# Validation record — 2026-09-10

This record covers correctness and packaging of the first implementation. It is
not a throughput benchmark, an adaptive-depth task-accuracy evaluation, or a
reproduction of the paper's measured speedups.

## Environment and source

- Host: `dedicated-developjob-8gpu2-a029z-64896bc8cf-8p2lw`.
- Account: `hsliu2`; checkout: `/home/hsliu2/tmp/vllm-lt`, branch `codex-vllm-lt`.
- Python 3.12.13, PyTorch 2.13.0+cu130, Triton 3.7.1,
  Transformers 5.14.1, pytest 9.0.3; existing `/usr/local/bin/python` reused.
- New project bootstrapped from empty signed commit
  `3c09e6d`; implementation was uncommitted during initial validation.
  [Kernel source manifest](validation/kernel-validation-manifest.json) and
  [BF16 source manifest](validation/checkpoint-bf16-manifest.json) and
  [FP32 source manifest](validation/checkpoint-fp32-manifest.json)
  record SHA-256 hashes of the exact Python sources and tests.
- Ouro checkpoint and tokenizer:
  `ByteDance/Ouro-1.4B@574fa66cb8bf5abdc979642d01cf2b79b16bfab1`.
  All 269 parameter names/shapes match the published safetensors header:
  1,434,652,673 parameters. The 2,869,336,434-byte checkpoint was downloaded
  before inference, into `/home/hsliu2/tmp/vllm-lt-models/ouro-1.4b`.

## Completed checks

| Check | Result |
| --- | --- |
| CPU suite, `OMP_NUM_THREADS=1 python -m pytest -q` | 54 passed; 15 opt-in GPU cases skipped |
| Reserved GPU, `python -m pytest -q tests/test_attention.py --run-gpu` | 20 passed, including all 15 GPU cases |
| `python -m ruff check .` | Passed |
| `python -m ruff format --check .` | Passed |
| `git diff --check` | Passed |
| CPU toy CLI, four adaptive output tokens for two requests | Passed; exits vary between two and four loops |
| Wheel build with `pip wheel --no-deps --no-build-isolation` | Passed; verified native model, LICENSE, and NOTICE in wheel |
| Real checkpoint FP32 CPU smoke | Generated ` Paris.`; first token matched dense oracle; zero KV blocks retained |
| Real checkpoint FP32 GPU diagnostic | Passed all logits, token/exit equivalence, and memory cleanup checks |
| Real checkpoint BF16 GPU smoke | Tokens and exits matched; strict logit tolerances failed (details below) |

GPU kernel testing ran through `gpu run --gpu-ids 0 --nonblock --timeout 10m
--note 'vllm-lt Triton correctness feasibility' -- ...` after scheduler status
checks. The reservation was automatically released when the command finished.
No other reservations or processes were modified.

The CPU suite checks each loop of tiny Ouro against an independent functional
dense implementation, as well as token-by-token versus full causal execution,
mixed request/depth batches, checkpoint loading, and stage transitions. Engine
tests compare serial generation with packed refill and no-refill execution,
including cumulative gate probabilities, heterogeneous exit policies, bounded
KV admission, EOS, cancellation, and deterministic per-request sampling.

Kernel tests compare against dense PyTorch SDPA for FP32, FP16, and BF16, GQA,
noncontiguous layer strides, fragmented pages, head dimensions 7/32/64/128,
and causal contexts spanning multiple online-softmax tiles. Each row uses its
own request/depth page table. Separate tests exercise skipped-depth propagation
and detect uninitialized cache reads after reuse.

The [CPU real-checkpoint result](validation/checkpoint-cpu-probe.json) records
prompt token IDs `[504, 3575, 282, 4649, 314]`, output IDs `[7042, 30]`, and depths
`[4, 4]`. Only the first generated token was compared against the independent
dense oracle in this smoke. The second token exercised the incremental cache.

Raw kernel logs, wheel artifacts, and full checkpoint-validation logs/results
are retained in the worktree's ignored `artifacts/` directory. The manifests and
small CPU result above are included in version control.

## Real-checkpoint GPU validation procedure

`scripts/validate_checkpoint.py` loads the already downloaded weights once and
uses the same model for all checks. It compares packed Torch and Triton prefill
logits with an independent dense implementation at all four depths. Declared
BF16 tolerances before execution are `atol=0.25`, `rtol=0.02`, plus exact final
top-1 token agreement. The reference uses the published eager BF16 attention
arithmetic; both paged backends accumulate attention in FP32, so bitwise logit
equivalence is not expected.

Three short prompts use decode thresholds `[0.0, 0.7, 1.0]` and token limits
`[4, 8, 4]`. Both backends run serial, refill, and no-refill generation. The check
requires identical output token IDs and exit depths for those inputs and zero
retained KV blocks. It records GPU visibility, actual device, numerical errors,
stage traces, and CUDA allocation cleanup. The run budget is one complete smoke;
only identified failures or source changes justify another run.

Reproduce the passing FP32 diagnostic with:

```bash
gpu run --gpu-ids <available-id> --wait 20m --timeout 15m \
  --note 'vllm-lt Ouro checkpoint correctness' -- \
  env OMP_NUM_THREADS=1 python -m scripts.validate_checkpoint \
  --model-path /path/to/pinned/ouro-1.4b --device cuda --dtype float32 \
  --atol 0.001 --rtol 0.0001 --output artifacts/checkpoint-fp32.json
```

## GPU checkpoint results

Both real-checkpoint runs used physical GPU **5**, an **NVIDIA L20X**. The
scheduler exposed exactly that device as `cuda:0`. An earlier queued GPU-0
request (`38893bd0`) was cancelled before execution when GPU 5 became available.
Both executed reservations automatically released on exit; no task-owned queue
or GPU reservation remains.

The first [BF16 run](validation/checkpoint-bf16.json) failed its declared logit
tolerance at deeper loops. At depth four, maximum absolute differences were
0.5625 (Torch vs dense), 0.625 (Triton vs dense), and 0.28125 (Triton vs Torch).
All final-depth top-1 predictions and all six generation configurations agreed
on the tested output tokens and exit depths. For example, the three prompts
produced continuations beginning ` Paris.`, ` 4`, and ` cold.`. This small smoke
does not establish general batch invariance or adaptive-depth task accuracy.

The BF16 run also initially reported a 32 MiB memory leak after Python cleanup.
Inspection of the installed PyTorch `CudaMemoryLeakCheck` showed that it clears
persistent cuBLAS workspaces before leak accounting. The diagnostic script was
corrected to record allocations after GC, clear only this process's cuBLAS
workspaces, then check allocations again. This changes validation accounting;
it does not change model execution or the original recorded BF16 failure.

One [FP32 diagnostic](validation/checkpoint-fp32.json) followed to distinguish
an algorithmic mismatch from low-precision rounding. It used the same physical
GPU, weights, inputs, scheduler settings, and reference implementation; dtype
changed to FP32, and tighter tolerances `atol=0.001`, `rtol=0.0001` were declared
before execution. Every logit comparison at every loop passed; the largest
absolute difference was **6.2943e-5**. All top-1 predictions, generated token IDs,
and exit depths matched across both backends and all three scheduling modes.
After GC, 32 MiB remained; after clearing the cuBLAS workspace, allocated and
reserved bytes both reached **zero**. All request KV blocks had already been
released before that cleanup.

These results support the FP32 implementation's correctness on the tested
inputs. Different reduction orders and BF16 rounding are consistent with the
observed reduced-precision drift; this is an inference, not proof of BF16
equivalence. The BF16 tolerance was not relaxed and its failure remains open.
The default dtype and README GPU example use FP32. Further reduced-precision
validation and performance work are separate follow-ups.

Full stage traces are preserved in `artifacts/checkpoint-bf16.json` and
`artifacts/checkpoint-fp32.json`; the versioned copies omit only stage traces.
