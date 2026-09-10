# Resolved M3 CUDA capture contract — 2026-09-11

This is the pre-run contract for issue #6. It records one bounded correctness and performance experiment; it contains no measured result or qualification claim.

## Frozen identity and chronology

The common execution source is `0d9284a849c3ecdc2bbb5fbc4103729c98ce998d`; A and B have identical clean tracked source manifests (143 files), with no restore allowlist or production-source difference. Only `use_graphs=false` (A) / `true` (B) changes.
A root: `/data/hsliu2/tmp/vllm-lt-m3-capture-a`. B root: `/data/hsliu2/tmp/vllm-lt-m3-capture-b`. Preparation/document root: `/data/hsliu2/tmp/vllm-lt-m3-capture`.
The common harness SHA-256 is `dc3f72a25920a470957ae4331be6e43c1aefe9c589b5ba1c83197119e0f42599`; the full dependency manifest SHA-256 is `a6818e6898e16dbb61c17630b5871de0626b6528599d4544bde22dc199169d60`.
Resolved plan: `/data/hsliu2/tmp/vllm-lt-m3-capture/artifacts/capture-plan-20260911/plan.json`. Canonical identity: `528e2665c0bf5588a4cb5ea100c7c46188f7fc3600b8b2c7eb54e5e46c5e78a6`. Exact plan-file SHA-256: `d96cd7b0bf4b8a22cb09522c7623e4442075e1d7b78b65a56fede7f585fbf8d8`.
Host selection record: `/data/hsliu2/tmp/vllm-lt-m3-capture/artifacts/capture-host-before.json`, SHA-256 `ba1b04cc3f2eb107d5eee836fc7daa3e29570ca73cd89f83c2ec2aa60b67b500`; recorded at `2026-09-10T21:30:59.637634+00:00`.
Retained generator SHA-256: `44da20a56d8028d69d4b16d5fc23e8b7ca31bce4c397e42bddfbf3ced10fe804`; CUDA-blocking wrapper SHA-256: `0f9991e9180cf7b9655f678292805959ab50e55e6b2cc9383ee3b29041781003`.
Numerical nested hash: `511eb4f8a9923159ba3630e002500e60afd45ba6102a95c3066779c482a6405f`; lifecycle: `582fe122293ff8cca41e25813c43ee878118d83540406d580e411b6d2874eff8`; kernels: `2460005af8b85cbaaac41c5a2c727cd7b52219f48eba9c3122bcdcad677da37f`.
Source is frozen before the CUDA-blocked CPU probe. Review and commit this resolved document before reserving/starting device work; the later document commit does not alter either detached execution checkout. Preserve probe/check logs and refresh scheduler status before execution. These are required sequencing steps, not assertions of completed GPU work.

| Frozen input | Bytes | SHA-256 |
| --- | ---: | --- |
| capture contract (`/data/hsliu2/tmp/vllm-lt-m3-capture-b/benchmarks/fixtures/ouro-m3-capture-contract.json`) | 4657 | `f947e611a49698d0ca47936a1d239171f6c92e82d3c095099e6d1ad95e465fab` |
| benchmark_contract (`/data/hsliu2/tmp/vllm-lt-m3-capture-a/benchmarks/fixtures/ouro-m1-contract.json`) | 2034 | `aec0de4c640ee29e1904ff83edb944aeff684b9f3cc29ab5c092fd70d9a83486` |
| benchmark_suite (`/data/hsliu2/tmp/vllm-lt-m3-capture-a/benchmarks/fixtures/ouro-m1.json`) | 90602 | `c43442bceec62a8978dd73f6e2d5754df26050e0fad195f34ede7b351681a6d9` |
| numerical_contract (`/data/hsliu2/tmp/vllm-lt-m3-capture-a/benchmarks/fixtures/ouro-q1-contract.json`) | 5523 | `61622b6e50670b1a46808acb913dc29be5e86f57332882f5784c35f5378057a5` |
| numerical_suite (`/data/hsliu2/tmp/vllm-lt-m3-capture-a/benchmarks/fixtures/ouro-q1.json`) | 38154 | `9b1fee4778fe77c20f937417946ce4ad0b5f21510659221d6cdedffaf92da197` |

Model `ByteDance/Ouro-1.4B`, model revision `574fa66cb8bf5abdc979642d01cf2b79b16bfab1`, tokenizer revision `574fa66cb8bf5abdc979642d01cf2b79b16bfab1`, prepared path `/data/hsliu2/tmp/vllm-lt-models/ouro-1.4b`.

| Checkpoint/tokenizer/config file | Bytes | SHA-256 |
| --- | ---: | --- |
| `config.json` | 1472 | `ce9cc13da41591b8b4deca053d7dfee06424c0228628ee862ea86d725bc163f3` |
| `merges.txt` | 466391 | `0b54e8aa4e53d5383e2e4bc635a56b43f9647f7b13832d5d9ecd8f82dac4f510` |
| `model.safetensors` | 2869336434 | `58872a72616c736595b8b7662079c5b12c5a162ec16eae94f21c348dfa9885af` |
| `special_tokens_map.json` | 825 | `55087bb8409060d9cb0f80495e34f4e0d8a84f68a799a6b0cab3132be0aae319` |
| `tokenizer.json` | 3522644 | `fcb808fe5e7642f5299be28aea07fc7f6d4f4364c3ac5e408e15a772cbc8fa8d` |
| `tokenizer_config.json` | 4211 | `7e010da95d71b0fa1aa809552922c74cc1dd98d628f21ba39ca6c35aa18d5d91` |
| `vocab.json` | 800656 | `7b9de3f47796abf8d00ab96be299fea0dc9afdf1827f34e7e0b9fb44593efe5c` |

## Fixed execution controls

Host `dedicated-developjob-8gpu2-a029z-64896bc8cf-8p2lw`, account `hsliu2`; exact physical GPU 0, NVIDIA L20X, UUID `GPU-006eb78c-23cb-f37d-eb7c-0ccb578b8f11`. The recorded selection showed `[{"details":"free for 1h 3m 47s","gpu_id":0,"gpu_model":"L20X","last_released":"2026-09-11T04:27:11.875926976+08:00","status":"AVAILABLE","validation":"0MB used"}]`; driver-reported memory was `0 MiB`.
Interpreter `/home/hsliu2/tmp/venvs/vllm-lt-q1/bin/python`; Python `3.12.13 (main, Mar  4 2026, 09:23:07) [GCC 11.4.0]`; Torch `2.13.0+cu130` / CUDA build `13.0`; Triton `3.7.1`; Transformers `4.55.0`; tokenizers `0.21.4`. All installed distributions are frozen in the plan.
Threads: intra-op 1, inter-op 1; seed 0. CPU IDs `[56, 57, 58, 59, 60, 61, 62, 63]`; active NUMA policy `{"cpubind":"1","membind":"1","nodebind":"1","physcpubind":"56 57 58 59 60 61 62 63","policy":"bind","preferred_node":"1"}`; cpuset evidence `['Cpus_allowed_list:\t56-63', 'Mems_allowed_list:\t0-1']`. Memory-node allowance is not the active memory policy; no GPU-locality claim is made.
Frozen environment: `{"CUBLAS_WORKSPACE_CONFIG":null,"CUDA_MODULE_LOADING":null,"HF_HOME":null,"HF_MODULES_CACHE":null,"MKL_NUM_THREADS":null,"NVIDIA_TF32_OVERRIDE":null,"OMP_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":null,"PYTORCH_ALLOC_CONF":null,"PYTORCH_CUDA_ALLOC_CONF":null,"TORCHINDUCTOR_CACHE_DIR":null,"TRITON_CACHE_DIR":null}`.
Numerical arithmetic controls: `{"allow_bf16_reduced_precision_reduction":false,"allow_fp16_reduced_precision_reduction":false,"allow_tf32":false,"cudnn_allow_tf32":false}`. FP32 only; optional Hub kernels are absent. This reuses the prepared Q1 environment, whose versions and binding must not be compared as identical to the earlier M1 system environment.
One reservation owns the same physical GPU for all eight sequential fresh worker processes. Each loads immutable weights once; each inference execution uses a fresh engine/cache. Preserve the worker allocator cache; do not drop shared caches. Loading, graph setup, warmups, and diagnostic captures are excluded from measured arrival-to-final-sync time.
Actual gate accumulation/readback and token sampling remain real. W4/W5 scheduling replay overrides only frozen exit/output decisions after that work; it is not an adaptive-quality evaluation. All requests use the frozen token IDs, arrival offsets and fixed EOS/output-length policies.
Benchmark engine/sampling controls: `{"attention_backend":"triton","cache":{"block_size":16,"num_blocks":1024},"dtype":"float32","sampling":{"exit_threshold":1.0,"ignore_eos":true,"max_loops":4,"min_loops":2,"seed":0,"temperature":0.0,"top_k":-1,"top_p":1.0},"scheduler":{"max_num_batched_tokens":128,"max_num_seqs":8,"min_coda_batch_size":1}}`.

## Exact execution budget and order

| Worker | Graph option | Model / kernel / lifecycle / benchmark executions | Total |
| --- | --- | --- | ---: |
| N-A | `False` | 10 / 3 / 1 / 7 | 21 |
| N-B | `True` | 5 / 3 / 1 / 7 | 16 |
| A1 | `False` | 0 / 0 / 0 / 14 | 14 |
| B1 | `True` | 0 / 0 / 0 / 14 | 14 |
| B2 | `True` | 0 / 0 / 0 / 14 | 14 |
| A2 | `False` | 0 / 0 / 0 / 14 | 14 |
| P-A | `False` | 0 / 0 / 0 / 4 | 4 |
| P-B | `True` | 0 / 0 / 0 / 4 | 4 |

Total: **101 executions / 8 workers / 8 model loads**. Correctness: 15 model cases (13 qualification + 2 excluded feasibility), 2 lifecycle evaluations, 6 direct-kernel evaluations = 23. Model comparison streams: 31 (30 qualification + 1 excluded feasibility).
Performance: 14 excluded feasibility executions; 28 warmups; 28 measured executions in A1→B1→B2→A2 order (seven cells per period); 4 diagnostic warmups + 4 profile captures for W1/W4-refill, each bounded to 16 decode outputs.
Within each N worker: first model feasibility → three direct-kernel evaluations → lifecycle → remaining model cases → seven performance feasibility cells. N-A must pass before N-B; both must pass before A1. P workers follow all four measured periods.
B owns 44 graph executors: four Triton model cases + one lifecycle + seven performance feasibility + fourteen warmup + fourteen measured + four diagnostic executions. Torch model cases and direct kernels create no graph executor.
Each executor prepares two buckets with three actual warmup traversals, one capture recording, and one verification replay each: 88 recordings, 264 warmup device traversals + 88 verification device traversals = 352 actual scratch GPU traversals. Capture records commands and does not execute the captured kernels; replay does not rerun Python tensor-body hooks.

## Numerical and lifecycle qualification

| Model case | Implementation / backend / scheduling | Fixtures | Phase / storage |
| --- | --- | --- | --- |
| `m3-capture-A-feasibility` | native / triton / serial | Q1-feasibility-float32 | feasibility / eager_tensor_body |
| `m3-capture-A-main-oracle-dense-serial-Q1-L16-F0` | oracle / None / serial | Q1-L16-F0 | validation / oracle_compact |
| `m3-capture-A-main-oracle-dense-serial-Q1-L256-F0` | oracle / None / serial | Q1-L256-F0 | validation / oracle_compact |
| `m3-capture-A-main-oracle-dense-serial-Q1-L64-F2` | oracle / None / serial | Q1-L64-F2 | validation / oracle_compact |
| `m3-capture-A-main-oracle-dense-serial-Q1-L128-F3` | oracle / None / serial | Q1-L128-F3 | validation / oracle_compact |
| `m3-capture-A-live_gate-oracle-dense-serial-Q1-L16-F0` | oracle / None / serial | Q1-L16-F0 | validation / oracle_compact |
| `m3-capture-A-main-native-triton-refill-mixed` | native / triton / refill | Q1-L16-F0, Q1-L256-F0, Q1-L64-F2, Q1-L128-F3 | validation / eager_tensor_body |
| `m3-capture-A-main-native-triton-no_refill-mixed` | native / triton / no_refill | Q1-L16-F0, Q1-L256-F0, Q1-L64-F2, Q1-L128-F3 | validation / eager_tensor_body |
| `m3-capture-A-live_gate-native-triton-serial-Q1-L16-F0` | native / triton / serial | Q1-L16-F0 | validation / eager_tensor_body |
| `m3-capture-A-main-native-torch-serial-Q1-L16-F0` | native / torch / serial | Q1-L16-F0 | validation / backend_fallback |
| `m3-capture-B-feasibility` | native / triton / serial | Q1-feasibility-float32 | feasibility / graph_replay |
| `m3-capture-B-main-native-triton-refill-mixed` | native / triton / refill | Q1-L16-F0, Q1-L256-F0, Q1-L64-F2, Q1-L128-F3 | validation / graph_replay |
| `m3-capture-B-main-native-triton-no_refill-mixed` | native / triton / no_refill | Q1-L16-F0, Q1-L256-F0, Q1-L64-F2, Q1-L128-F3 | validation / graph_replay |
| `m3-capture-B-live_gate-native-triton-serial-Q1-L16-F0` | native / triton / serial | Q1-L16-F0 | validation / graph_replay |
| `m3-capture-B-main-native-torch-serial-Q1-L16-F0` | native / torch / serial | Q1-L16-F0 | validation / backend_fallback |

Native cache: 160 blocks × 16 positions; scheduler `{"max_num_batched_tokens":64,"max_num_seqs":4,"min_coda_batch_size":1}`. Nine predictions per request: the last prompt query and eight continuation inputs; forced histories/depths and real live-gate histories are distinct declared cases.
Projection `loop_gate_logits_full_kv_v1` retains the actual normalized loop output and actual gate logits at every selected query's executed depth, plus final CODA logits. Final populated K and V cover every P+8 position, all four materialized depth planes, and all 24 physical layers, in 4-position chunks. There is no per-loop full-KV snapshot or claim of the old six per-layer hidden/Q/K/V hook coverage.
Original FP32 final-logit allclose: atol=0.001, rtol=0.0001; actual top-1/token, executed exit depth, input history, and cache/lifecycle invariants must agree. All observed intermediate and populated-KV values must be finite; their arithmetic deltas are diagnostic, without a new tolerance. A live decision divergence fails at the matched prefix; subsequent incompatible trajectories are explicitly incomparable. BF16 is outside this experiment.
A runs the common tensor body eagerly; B must actually replay both 4×32 and 8×32 buckets with odd-slot mapping, up to four live rows. The extra real-model A/B Torch forced L16-F0 pair uses counted compact backend fallback and zero graph setup. Storage/publication and dispatch/generation evidence is collected after replay, without capture-unsafe Python layer hooks.
Lifecycle: two held-input Triton evaluations preserve exact active outputs, fresh publication, page reuse, transaction/stale-lease behavior, both bucket visits, empty dispatch, and counted live-count/table-width fallbacks. Each has 173 typed records and 480 whole-pool guard chunks (960 total). The long four-row case reaches position 511; the later reused r1 input is position 2.
Direct kernels: A/B × K4-zero, K4-one-limit, K4-two-crossing = six evaluations; 40 full-pool guard chunks each (240 total). The prior inactive-row/persistent prerequisite evidence remains required; this subset does not rename old layer-hook counts or turn held-input coverage into full-model quality coverage.

## Frozen resource bounds and stop rules

| Resource | Bound / exact planned amount (bytes unless marked) |
| --- | ---: |
| Common device payload, both buckets | 262144 |
| Common CPU staging, both buckets | 16384 |
| Actual common device / CPU staging payload | 198588 / 1884 |
| Temporary model scratch restoration (four pages) | 25165824 |
| Graph retained allocated / reserved, independently | 268435456 / 268435456 |
| Setup peak allocated / reserved, independently | 536870912 / 536870912 |
| Each measured B−A peak allocated AND reserved increase | 536870912 |
| Retained numerical tensors / largest retained case | 2536572960 / 788267520 |
| Retained index / comparison record upper bounds (counts) | 1980 / 3879 |
| Per index / comparison serialized record | 1024 / 2048 |
| Transient matching / live spool group | 67108864 / 2147483648 |
| Cumulative tensor writes / numerical artifact cap | 8589934592 / 12884901888 |
| Persisted diagnostic dumps / per fixture | 268435456 / 67108864 |
| Pointer-record count / derived bytes / hard cap | 3046 / 199622656 / 209715200 |
| Lifecycle tensor / JSON / total suite caps | 16777216 / 8388608 / 50331648 |
| Kernel exact output tensor payload / suite cap | 122880 / 268435456 |
| Each profile trace / all four traces | 2147483648 / 4294967296 |
| Auxiliary evidence allowance | 1073741824 |
| Derived total artifact upper bound / hard cap | 8712171552 / 17179869184 |

Complete setup must finish within 60 seconds to qualify; phase checks are cooperative. Independent parent watchdogs bound each case to 600 seconds and the controller to 7200 seconds, including setup, export, and cleanup. A blocked device call is subject to case/global interruption; no hard 60-second setup interruption or earlier M2 relative 100-ms setup cap is claimed.
Configured capture-budget decline is a product compact fallback only after confirmed completion and safe scratch/resource restoration. This experiment requires an actual prepared executor: any decline fails qualification before inference. No device/OOM/correctness failure, uncertain completion, or cleanup failure may become an eager retry or a new capture attempt.
Stop on the first execution, required numerical/behavior, ownership, nonfinite, source/input, dependency, GPU/binding, deadline, or resource-cap failure. Preserve the failed prefix. Confirm task-owned worker exit and zero allocated/reserved device memory after workspace release before the next worker. Never add a replacement case, retry, or unbudgeted trial.
The recorded host search found neither compute-sanitizer nor cuda-memcheck on PATH or the listed CUDA locations: zero checker executions. Guard/source checks and CPU failure-injection tests do not claim sanitizer coverage or observed GPU fault recovery.

## Measured acceptance and evidence

For each frozen pair `[['A1', 'B1'], ['A2', 'B2']]`, W1-refill B/A throughput must be ≥ 1.1; each of the six controls `['W2-refill', 'W3-refill', 'W4-refill', 'W4-no_refill', 'W5-refill', 'W5-no_refill']` must be ≥ 0.95. W1 TTFT B/A must be ≤ 1.05; both measured peak-memory increase checks must pass in every matched cell/pair.
W1 must also satisfy min(B throughput) > max(A throughput). The two measured samples per configuration expose observed ranges, not a confidence interval. A failed pair cannot be rescued by an average. Range overlap is inconclusive if every other required gate passes; no further samples may be added under this contract.
All original required numerical gates, held-input/lifecycle guards, actual bucket replay and post-gather publication proof, setup/cap controls, matched work/counter identities, cleanup, and correlated profile replay-launch evidence for both W1 and W4-refill are required. Timing uses lightweight host events/counters; rich tensor descriptors and profiler collection are restricted to correctness/profile executions and use the same policy per A/B pair.
Corrupt/unverifiable evidence is invalid and unqualified. An independently validated stopped prefix with a known required failure is failed with incomplete coverage. Missing-only coverage is inconclusive. No failed, missing, or variable result expands the frozen budget.
Raw run directory: `/data/hsliu2/tmp/vllm-lt-m3-capture/artifacts/capture-run-20260911`. Preserve plan, manifests, case/comparison streams, retained typed payloads/indexes, lifecycle/kernel guards, all profile traces, scheduler/host records, commands, CPU/probe/console logs, cleanup, and offline report outside Git. Publish a compact report and checksummed raw archive separately after completion; this pre-run document neither predicts the result nor asserts archive availability.

## Reproduction commands (not executed by this generator)

Use the retained CUDA-blocking wrapper for probe/report. The two clean detached checkouts must already exist at the common SHA; the probe destination must not exist. The GPU command is a reservation template to be reviewed after a fresh scheduler check.

```bash
cd /data/hsliu2/tmp/vllm-lt-m3-capture-b
numactl --physcpubind=56-63 --membind=1 env OMP_NUM_THREADS=1 PYTHONPATH=/data/hsliu2/tmp/vllm-lt-m3-capture-b /home/hsliu2/tmp/venvs/vllm-lt-q1/bin/python /data/hsliu2/tmp/vllm-lt-m3-capture/artifacts/capture_cpu_cli.py probe --baseline-root /data/hsliu2/tmp/vllm-lt-m3-capture-a --candidate-root /data/hsliu2/tmp/vllm-lt-m3-capture-b --contract /data/hsliu2/tmp/vllm-lt-m3-capture-b/benchmarks/fixtures/ouro-m3-capture-contract.json --model-path /data/hsliu2/tmp/vllm-lt-models/ouro-1.4b --gpu-id 0 --output /data/hsliu2/tmp/vllm-lt-m3-capture/artifacts/capture-plan-20260911
gpu run --gpu-ids 0 --nonblock --timeout 130m --note 'M3 capture 0d9284a849c3ecdc2bbb5fbc4103729c98ce998d' -- numactl --physcpubind=56-63 --membind=1 env OMP_NUM_THREADS=1 PYTHONPATH=/data/hsliu2/tmp/vllm-lt-m3-capture-b /home/hsliu2/tmp/venvs/vllm-lt-q1/bin/python -m vllm_lt.benchmarks.capture run --plan /data/hsliu2/tmp/vllm-lt-m3-capture/artifacts/capture-plan-20260911/plan.json --output /data/hsliu2/tmp/vllm-lt-m3-capture/artifacts/capture-run-20260911
numactl --physcpubind=56-63 --membind=1 env OMP_NUM_THREADS=1 PYTHONPATH=/data/hsliu2/tmp/vllm-lt-m3-capture-b /home/hsliu2/tmp/venvs/vllm-lt-q1/bin/python /data/hsliu2/tmp/vllm-lt-m3-capture/artifacts/capture_cpu_cli.py report --run-dir /data/hsliu2/tmp/vllm-lt-m3-capture/artifacts/capture-run-20260911
```

## Completed CPU preflight

Before reservation, the CUDA-blocked full suite passed **908 tests with 16 skipped** in 210.46 seconds; Ruff passed and all 109 checked files were formatted. Production source was frozen for the resolved probe; the last changes during the full check concerned report decision wording, whose 58 focused report tests also passed. These checks do not establish CUDA capture correctness.

- `artifacts/capture-full-cpu.log`: SHA-256 `b920926b7b1caf43e53be1c63c225317a56c79e7fca898ee21756ee5d662dfa2`.
- `artifacts/capture-lint-final.log`: SHA-256 `82b3e6a6c090a57601d22943bd23fca9218d1031dbe5a7b754092f9a156b4f18`.
- `artifacts/capture-format-final.log`: SHA-256 `93f257ad0b7294206e6eb174e7f37cfc2821709aff9dc074632c0d4fdbc8104d`.
- `artifacts/capture-probe.log`: SHA-256 `8ba89ba37aaa08bd6c8a701d07e97ccee18706d4f4f6409f3d512ebe2017d3c4`.

The reviewed GPU command adds `--nonblock` to the generated reservation template; an unavailable selected GPU stops this attempt before device work. The original generator is retained unchanged.
