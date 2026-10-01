# Training contract with two A100 workers — 2026-10-01

Two A100-SXM4-40GB GPUs, driver 580.105.08, the existing `/venv/main` runtime,
PyTorch 2.13.0+cu130, Triton 3.7.1 and NIXL 1.4.1. Separate spawned P/D owners
register their GPU KV/hidden pools and exchange real NIXL WRITE transfers.
The public `LLM` interface stages physical policy chunks through CPU IPC and waits
for every worker to commit. No package, driver or service changes are made.

| Case | Active slots | Maximum logprob error vs local | Published/recovered version | D graph captures / replays | P bytes transferred |
|---|---:|---:|---:|---:|---:|
| Random FP32, eager, K=2/3/4 | 4 | 4.77e-7 | 5 | Disabled | 300,928 |
| Random FP32, graphs, K=2/3/4 | 4 | 4.77e-7 | 5 | 2 / 116 | 300,928 |
| Full BF16 Ouro-1.4B, graphs, K=2/3/4 | 1 | 0 | 5 | 1 / 206 | 229,699,584 |

Each case compares raw and processed scores at temperature 0.9, top-k 8 and
nucleus 0.8, with two prompts and six response tokens. Token IDs, physical exit
depths, effective parameters and policy versions agree. Prompt prefill is full
K=4; decode budgets change after each complete publication. Prefix reuse,
chunked updates, a rejected update, recovery into version five, explicit stop
and active abort are exercised. Both workers finish with zero used blocks and
version five. Graph captures remain valid through publication and recovery.

The full-model source is `ByteDance/Ouro-1.4B` at
`574fa66cb8bf5abdc979642d01cf2b79b16bfab1`, safetensors SHA256
`58872a72616c736595b8b7662079c5b12c5a162ec16eae94f21c348dfa9885af`.
See [full outputs/counters](raw/pd-full-graph-matched.json),
[tiny eager](raw/pd-tiny-eager-tcp.json), [tiny graphs](raw/pd-tiny-graph-tcp.json)
and [runtime](raw/runtime.json).

## Transport and numerical limits

Default UCX GPU IPC reports completion but leaves the destination zero in an
independent two-process, 128-byte probe. Explicit IPC and shared-memory-only
configurations also fail on this container. `UCX_TLS=tcp,cuda_copy` passes the
probe and every case above with the installed library. Transfers are host staged;
this report does not establish direct GPU IPC/RDMA performance. The workaround
is scoped to each command. [Probe source](../../../benchmarks/nixl_preflight.py),
[default failure](raw/nixl-preflight.log) and [passing probe](raw/nixl-preflight-tcp.log)
are retained. No processes are forcibly terminated; successful workers use the
normal stop/acknowledgment protocol.

A full BF16 comparison with four slots has a maximum observed selected-score
difference of 0.11199 when local and PD prefill batches differ. The matched
one-slot run above is exact; the FP32 four-slot run is within 4.77e-7.
Different BF16 batching is therefore not claimed to be bitwise equivalent. See
[the rejected strict comparison](raw/pd-full-graph-tcp.log).
Other jobs were already running on both GPUs. Timings in these parity probes
include startup/capture/prefix-state differences and contention, and are not a
controlled throughput comparison. The [Thor table](../training-contract-thor-20261001/README.md)
contains alternating-order warmed eager/graph timings.

## Reproduction

From the repository root, in an existing CUDA/NIXL environment:

```bash
UCX_TLS=tcp,cuda_copy python benchmarks/nixl_preflight.py
UCX_TLS=tcp,cuda_copy python benchmarks/pd_training_contract.py tiny.json
UCX_TLS=tcp,cuda_copy python benchmarks/pd_training_contract.py tiny-graph.json --cuda-graphs
UCX_TLS=tcp,cuda_copy python benchmarks/pd_training_contract.py full.json \
  --model /models/Ouro-1.4B --cuda-graphs --max-num-seqs 1
```

The VIME training/save/resume and general HF export checks are recorded in
[0z5a/vime#5](https://github.com/0z5a/vime/pull/5).

## Repository checks

The complete default CPU regression suite passes: **421 passed, 141 skipped**
in 19.41 s; GPU tests, including the upstream forced-peer-failure test, are skipped
by the repository's ordinary collection rules. See
[regression log](raw/cpu-regression-final.log) and
[focused contract tests](raw/contract-cpu-final.log).
Ruff lint and format checks pass for all 88 Python files.
The existing runtimes lack `pre_commit`, so its three `language: system` hooks
are run directly without installing it: Python-file lint, format check and the
full default CPU suite. See [lint log](raw/lint-final.log).
