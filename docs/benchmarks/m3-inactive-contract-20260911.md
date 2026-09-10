# M3 inactive rows: resolved correctness contract — 2026-09-11

This protocol is committed before device execution. The separate clean execution
checkouts below remain fixed; later documentation commits do not change their code.

Resolved at `2026-09-10T19:10:20.188027+00:00`.
Plan: `/data/hsliu2/tmp/vllm-lt-m3-inactive/artifacts/m3-plan-20260911/plan.json`.
Canonical plan SHA-256: `39dec46e21eedb198738cdf2d3ca6483239eb7e44c00872c61feb06a40c0f282`.
Plan file SHA-256: `ca959522ec6621db2b6bdae70bd8f95abf21ad79ef4ce2f80866a228c3b23c7d`.

## Hypothesis and source controls

Masked inactive recurrent rows preserve live numerical behavior and cannot access or mutate live state.

Compact M2 recurrence A versus padded eager decode B; physical dense row count is a necessary consequence.

Production remains compact by default. B qualifies the private padded eager decode
path. This is the AC-M3-02 prerequisite, with relevant eager bookkeeping/publication
checks; it cannot close M3 or qualify graph capture, stable buffers or performance.

| Side | Frozen commit | Clean execution checkout |
| --- | --- | --- |
| A | `94320f9a4b167e9ed3e098d8d8c12a4bcb68f57d` | `/data/hsliu2/tmp/vllm-lt-m3-inactive-control` |
| B | `1583e1d34cf121ecd1d781ea0193775ab1ef2b6e` | `/data/hsliu2/tmp/vllm-lt-m3-inactive-candidate` |

A restores the five production files below from M2 commit
`25c2c10e3ed819d2104ac92b713217eca71abf20`. Their implementation bytes match
its predecessor `3b410442923637ea1183ad6d61d73a60f603fd47`; the newer commit
corrects report wording only. Every other A/B file matches, including the
unused-on-A masked-write module and the common validation code/fixtures/tests.
Common harness SHA-256: `c9df2623a8816e09cc9466c6b63fba35a086aaf7099ba4b15c550e05d4751106`.

| Production path | A SHA-256 | B SHA-256 |
| --- | --- | --- |
| `vllm_lt/core/kv_cache_manager.py` | `bae7e9c442f855b94de07ccb497544033a28a396d265ac5eaa10c6f6025c8017` | `3afb020da127d24a1c579fc097ffa0313b39ab476f92fd89dcd711fbd8b65daa` |
| `vllm_lt/kernels/paged_attention.py` | `273e34770e2e85bfb3f39d345b99b538720c54dcd0fd04806ee85694105198fb` | `2f8ff17f1abbb5bbd50960787d83fa318e41b1887cb014ed0921419e4e26d492` |
| `vllm_lt/kernels/triton_attention.py` | `404b7b079cd0717334662952c09d7bb44a1a9c0d71860c170e20421e63190eda` | `0861c57c25d7e5b284cfe4c259ae324a2a420a42c72fa3d3124d0fd13e63098e` |
| `vllm_lt/models/ouro.py` | `e5a4593539789efb108b8535a72e63e12fbfc6fae245e146ad850aa96979df27` | `a928df915b3ff57b0fb8e041e89f0f141bce8a1cfc4c6f2616e7eb0b2ef620ec` |
| `vllm_lt/worker/model_runner.py` | `e52f8784b62a4df9cd805d4ddb3eefae5bac6bdd9b3d172785ff434a2b6218ee` | `6704a2884480b99d95e040a34aa89fdff300e90ac6474f709d1e1ac85b21822a` |

## Fixed controls

Actual checkpoint: `ByteDance/Ouro-1.4B@574fa66cb8bf5abdc979642d01cf2b79b16bfab1`, FP32.
Model directory: `/data/hsliu2/tmp/vllm-lt-models/ouro-1.4b`.
Interpreter: `/home/hsliu2/tmp/venvs/vllm-lt-q1/bin/python`.
Both sides reuse Python 3.12.13, Torch 2.13.0+cu130, Triton 3.7.1,
Transformers 4.55.0 and the prepared Q1 environment; optional `kernels` is absent.
Exact dependency versions and file identities are retained in the plan.

Host: `dedicated-developjob-8gpu2-a029z-64896bc8cf-8p2lw`; account: `hsliu2`.
Physical GPU 7 is an NVIDIA L20X. Its previously verified UUID is
`0e488755-d8ed-b688-81e7-8e377dc10fd5`; recheck the recorded runtime UUID against
it before accepting the result. The scheduler owns device visibility. CPU cores
56–63 and active NUMA memory policy `bind` to node 1 are fixed. The allowed
memory mask remains 0–1; active binding is independently checked with `numactl`.
No GPU-locality claim is made. Seed 0, one intra-op/inter-op thread, and disabled
TF32/BF16/FP16 reduced-precision reductions remain fixed. Only `OMP_NUM_THREADS=1`
is set among the plan's recorded optional runtime variables; the rest are unset.

One fresh worker/model load per side; one resident per-case pool. A and B run
sequentially with task CUDA cleanup before the next worker. No downloads, shared
cache drops or model reloads between that worker's cases. Loading, fixture setup,
compilation, export and cleanup are recorded; no time is reported as speedup.

| Input | SHA-256 |
| --- | --- |
| contract | `61622b6e50670b1a46808acb913dc29be5e86f57332882f5784c35f5378057a5` |
| suite | `9b1fee4778fe77c20f937417946ce4ad0b5f21510659221d6cdedffaf92da197` |
| M3 contract | `0c40b0ea94ec1ff4304bf0e298ff9daa3404d0a2514401d5e153b3c9139c7f58` |
| Model `config.json` | `ce9cc13da41591b8b4deca053d7dfee06424c0228628ee862ea86d725bc163f3` |
| Model `merges.txt` | `0b54e8aa4e53d5383e2e4bc635a56b43f9647f7b13832d5d9ecd8f82dac4f510` |
| Model `model.safetensors` | `58872a72616c736595b8b7662079c5b12c5a162ec16eae94f21c348dfa9885af` |
| Model `special_tokens_map.json` | `55087bb8409060d9cb0f80495e34f4e0d8a84f68a799a6b0cab3132be0aae319` |
| Model `tokenizer.json` | `fcb808fe5e7642f5299be28aea07fc7f6d4f4364c3ac5e408e15a772cbc8fa8d` |
| Model `tokenizer_config.json` | `7e010da95d71b0fa1aa809552922c74cc1dd98d628f21ba39ca6c35aa18d5d91` |
| Model `vocab.json` | `7b9de3f47796abf8d00ab96be299fea0dc9afdf1827f34e7e0b9fb44593efe5c` |

## Exact execution order and model gates

Exactly **65 executions**, one deterministic pass per declared case/configuration:
A feasibility → 26 A kernel evaluations → five oracle and three compact native
qualification cases → B feasibility → 26 B kernel evaluations → three padded
native qualification cases. No retries, replacement trials or unplanned cases.
This correctness prerequisite explicitly uses one pass, with no performance
estimate and no benchmark warmup/measured repetitions.

The 13 model cases contain two excluded feasibility and 11 qualification cases.
The 27 qualification comparison streams remain separate from one excluded
feasibility B/A stream. Required failures in either stop subsequent cases.

The unchanged main fixtures are Q1-L16-F0, Q1-L256-F0, Q1-L64-F2 and Q1-L128-F3.
The independent oracle executes each serially plus L16-F0 with live gates.
Each native side runs the four-fixture group with refill, then no-refill, then
L16-F0 live. Each trajectory records nine genuine predictions. Fixed/forced
histories consume eight supplied inputs; live histories consume generated tokens.
Live routing uses threshold 0.7, minimum depth 2 and maximum depth 4.

B pads decode to eight physical rows, logical row i at physical row 2*i+1,
and block-table width 32. Prefill/coda remain compact. Actual scheduled request,
position and depth identities are checked before applying the physical row map.
Inactive hidden/gate outputs must be finite zero and omitted from publication.
The pool remains 160 pages × 16 tokens; scheduler maximum four requests and
64 batched tokens. The changed dense GEMM row count is a necessary consequence.

All oracle/native and B/A streams retain Q1 FP32 final-logit bounds
`atol=0.001`, `rtol=0.0001`, exact actual top1/exit/history requirements,
finite selected intermediates and full populated-KV coverage. Hidden/gate/KV
deltas are diagnostic; full-model bitwise identity is not required. Preselected
paired FP32 B/A dumps are L64-F2 and L256-F0. This does not qualify BF16 or quality.

## Held-input kernel matrix

Kernel plan SHA-256: `c29b94a25cbe103e96a7aa201f2bf6920b5c8bfa9ae5f5d253321c1a43c76106`.

| Layout | Physical rows | Table width | Live physical indices | Active context lengths |
| --- | ---: | ---: | --- | --- |
| L00 | 0 | 1 | `[]` | `[]` |
| L01 | 1 | 1 | `[]` | `[]` |
| L02 | 1 | 1 | `[0]` | `[1]` |
| L03 | 4 | 1 | `[]` | `[]` |
| L04 | 4 | 1 | `[0]` | `[16]` |
| L05 | 4 | 1 | `[3]` | `[15]` |
| L06 | 4 | 1 | `[0, 2]` | `[1, 16]` |
| L07 | 4 | 1 | `[0, 1, 2, 3]` | `[1, 2, 15, 16]` |
| L08 | 4 | 2 | `[0, 1, 2, 3]` | `[15, 16, 17, 31]` |
| L09 | 8 | 2 | `[0, 1, 2, 3]` | `[16, 17, 31, 32]` |
| L10 | 8 | 2 | `[1, 3, 5, 7]` | `[16, 17, 31, 32]` |
| L11 | 8 | 3 | `[1, 3, 5, 7]` | `[17, 31, 32, 33]` |
| L12 | 8 | 32 | `[1, 3, 5, 7]` | `[33, 511, 512, 16]` |

Each layout runs compact A and masked B once on Torch and once on Triton:
52 evaluations. The plan freezes CPU-generated physical and logical Q/K/V and
metadata hashes before device work. Deterministic cache/row formulas use FP32,
16 query/KV heads, head dimension 128 and page size 16. Each direct pool has
160 pages and two physical layers (target layer 1), exercising strided views
and protecting the other layer. Both cache tensors total 83,886,080 bytes.
Logical pages use distinct odd physical IDs; neighboring even pages are guards.
Inactive inputs are NaN, lengths zero, and addresses alternate -1 / 2**30.
Unused table columns also have invalid sentinels.

Require exact active compact/padded attention and writes within each backend,
finite positive-zero inactive attention, and exact whole-cache agreement with
CPU-constructed expected writes/untouched guards. Every evaluation checks all
40 K/V chunks of eight pages, including neighboring pages and the other layer.
Torch/Triton arithmetic differences are not an invented cross-backend gate.
All raw attention outputs are typed and hash checked, with exact metadata/file
coverage checked offline.

`compute-sanitizer` and `cuda-memcheck` are absent from PATH, conventional locations
and the actual `/usr/local/cuda-13.0` installation. Zero checker executions are
planned; this missing coverage is explicit. CPU tests, invalid-address/NaN cases,
whole-cache guards and source-level masked access remain required.

## Budgets, stops and acceptance

Overall 3,600 seconds includes verification/loading/exports/cleanup, with 600
seconds per whole model/kernel case. The controller watches active markers and
cleans up only its child process group on timeout/SIGINT/SIGTERM. Each completed
case must retain zero task requests/reserved pages, and each worker must finish
with zero allocated/reserved CUDA bytes. Check and record scheduler release.

Live matching ≤ 64 MiB, retained group ≤ 2 GiB, cumulative tensor writes ≤ 8 GiB,
paired dumps ≤ 256 MiB, kernel evidence ≤ 256 MiB, all artifacts ≤ 12 GiB.
No profiler captures are planned or permitted. Model retained-payload upper
bound: 3,067,709,328 bytes; largest case: 939,262,464 bytes; retained index bound:
71,583 records × 1,024 bytes; comparison bound: 133,776 records × 2,048 bytes.
Kernel output payload: 1,425,408 bytes, plus at most 8 MiB auxiliary evidence.

Execution/address/nonfinite/isolation/required-logit/actual-decision failures,
control drift, cleanup failures and budget violations stop subsequent cases.
Finite model mismatches retain the current case's bounded diagnostics. Preserve
all partial evidence; valid failed prefixes are incomplete failed outcomes,
and corrupted identities/chronology are invalid. Neither is acceptance.

Acceptance requires every declared numerical/discrete/kernel/isolation/control/
cleanup gate and complete offline coverage. Link raw artifacts, individual
outcomes, source/input hashes and the missing-checker limitation in the PR.
The result supports only the inactive-row prerequisite; all remaining M3 graph
and performance gates stay open. See [interfaces and commands](../m3-inactive-rows.md).

## Command used for the reserved comparison

Run from the frozen candidate checkout with the exact controls used for probing:

```bash
gpu run --gpu-ids 7 --nonblock --timeout 1h --note 'vllm-lt M3 inactive A/B 1583e1d' -- \
  numactl --physcpubind=56-63 --membind=1 -- env OMP_NUM_THREADS=1 \
  /home/hsliu2/tmp/venvs/vllm-lt-q1/bin/python -m vllm_lt.validation.m3_inactive_run run \
  --plan /home/hsliu2/tmp/vllm-lt-m3-inactive/artifacts/m3-plan-20260911/plan.json \
  --output /home/hsliu2/tmp/vllm-lt-m3-inactive/artifacts/m3-run-20260911
```
