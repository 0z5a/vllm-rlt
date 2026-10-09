# Re:Loop seven-model G0 source preparation

All seven checkpoint source/input/model-manifest packets are present. Independent review verified the complete 42-file tar, all 41 payload hashes and local bytes, 12 unchanged V2 model files and the unchanged assigned-device helper. This is source review; no new CPU numerical qualification or GPU experiment ran.

| Model | Declared sequential children | Declared cache cases | Actual new GPU cases |
|---|---:|---:|---:|
| armenjeddi/LoopFormer-3block-8iterations | 1 | 64 | 0 |
| ByteDance/Ouro-2.6B | 6 | 384 | 0 |
| Nanbeige/Nanbeige4.2-3B | 6 | 384 | 0 |
| ByteDance/Ouro-1.4B | 6 | 384 | 0 |
| sapientinc/HRM-Text-1B | 1 | 192 | 0 |
| tomg-group-umd/huginn-0125 | 6 | 384 | 0 |
| SandyResearch/parcae-370m | 6 | 384 | 0 |

The new Huginn/Parcae probes require actual CPU Torch qualification at their unchanged BF16 0.02 absolute/relative budget before any target GPU admission. They pair initial states and contain state/logit tensor-export code; no such exports have been produced. Prior K/V tensor pairs are absent, and the other five model probes remain scalar-only. Source presence does not establish independent full-tensor replay.

The frozen template forbids execution. All 32 declared children are sequential, with 2,176 declared cache cases. Actual high-concurrency/multi-batch native and HTTP tests, quality, Graph/async and model cleanup remain separate required work; no whole-model pass is recorded. Existing failure and unfinished records are retained.

[Scientific source inventory and immutable hashes](G0_SEVEN_MODEL_SOURCE-v3.json). [Previously reviewed raw model coverage](LOOPED_MODEL_REUSE_COVERAGE-v1.md).
