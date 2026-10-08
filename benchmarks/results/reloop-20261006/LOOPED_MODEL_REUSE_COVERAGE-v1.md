# Looped model evidence reuse — full E2E incomplete

Existing source/data and raw evidence were reused without repeating the old experiments: all87 handoff payloads and all141 original inner raw files match their original hashes. Five source snapshots preserve34 source/data files. These archives support continued development, not an H100 execution grant.

| Model | Official CPU adapter cases | Formal high-C / multi-BS GPU E2E complete | HTTP E2E complete |
|---|---:|---|---|
| Ouro-1.4B | 128 | No | No |
| Ouro-2.6B | 128 | No | No |
| Huginn-0125 | 96 | No | No |
| Nanbeige4.2-3B | 64 | No | No |
| LoopFormer-3block-8iterations | 48 | No | No |
| Parcae-370M | 96 | No | No |
| HRM-Text-1B | 112 | No | No |

All seven model variants remain incomplete for formal high-concurrency E2E. Current Ouro-1.4B FP32/eager numerical checks on one H100 pass the full original12/108/96/21 protocol, independent242-file byte/scalar review and known-actor handoff. The separate [native initial-total C32 result](H100_C32_FP32_RESULT-v1.md) subsequently completed its full fixed-work audit. Original W5, separate C128, Graph, HTTP and all-model high-C/multi-BS remain incomplete.

| Existing Ouro quantization smoke | Token cap128 | Token cap512 |
|---|---:|---:|
| SH1 speed change versus BF16 | −8.00% | −6.59% |
| Peak allocated-memory saving | 1.0962GiB | 1.0763GiB |

These older smoke results used eight arms, R4/C32, prompt256/output32 and64 requests per arm in one fresh repetition with a fixed20GiB KV pool. Token caps128/512 are token budgets, not request batch sizes. They are neither H100 results nor paired steady/HTTP E2E, locked quality or SOTA evidence.

Original Huginn/Parcae/Ouro failures, LoopFormer checkpoint mirror403 and all unrun cases remain retained. Per-model source/runtime/numerical contracts must close before new CUDA execution; old resource wrappers and bindings cannot be copied into a new ready. No weights were included, newly downloaded or deleted in this reuse handoff.

[Scientific coverage summary](LOOPED_MODEL_REUSE_COVERAGE-v1.json).
