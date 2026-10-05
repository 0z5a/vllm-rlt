# Model coverage — 2026-10-05

The execution plan names Ouro-1.4B, Ouro-2.6B and Huginn. The broader looped-model
coverage also includes Nanbeige, LoopFormer, Parcae and HRM-Text. This is the
execution queue: a pinned checkpoint or downloaded file is not a passed model
test. No row has completed quantized G0–G3 or formal paired serving measurements.

| Checkpoint | Immutable revision | Current evidence | Quantized E2E |
|---|---|---|---|
| [Ouro-1.4B](https://huggingface.co/ByteDance/Ouro-1.4B/tree/574fa66cb8bf5abdc979642d01cf2b79b16bfab1) | `574fa66cb8bf5abdc979642d01cf2b79b16bfab1` | All 128 official BF16 eager CPU comparisons exact; 128calibration and five128-window dev arms complete; native differences retained | Eight C32/B128/512 smoke arms measured; formal G3 pending |
| [Ouro-2.6B](https://huggingface.co/ByteDance/Ouro-2.6B/tree/1ed04250da1a9936042725d302e81c8fa2ab5abd) | `1ed04250da1a9936042725d302e81c8fa2ab5abd` | All 128 official BF16 eager CPU comparisons exact; verified node weights retained | Pending |
| [Huginn-0125](https://huggingface.co/tomg-group-umd/huginn-0125/tree/bb6621b65e90b6a4b9b29ef88dc83866d450470c) | `bb6621b65e90b6a4b9b29ef88dc83866d450470c` | All96 official BF16 SDPA CPU cases exact across R8/16/32; native GPU semantics pending | Pending |
| [Nanbeige4.2-3B](https://huggingface.co/Nanbeige/Nanbeige4.2-3B/tree/b82e54bd609793562a75cbf9337970a93369eab5) | `b82e54bd609793562a75cbf9337970a93369eab5` | All64 official BF16 eager CPU cases exact across R1/R2; 32 native tiny FP8 GPU cases passed | Pending |
| [LoopFormer 3-block/8-iteration](https://huggingface.co/armenjeddi/LoopFormer-3block-8iterations/tree/2b4fbaaf4e2510353ef7cfe07a07c671b5226739) | `2b4fbaaf4e2510353ef7cfe07a07c671b5226739` | Differentiable adapter: eight pinned-author tiny CPU cases bitwise exact; INT4 export/continuation CPU checks and four tiny INT4 Graph/KV GPU cases pass in draft #23 | Full checkpoint and quantized GPU qualification pending |
| [Parcae-370M](https://huggingface.co/SandyResearch/parcae-370m/tree/439284464ee4999bd1f762da7d044613a4828efe) | `439284464ee4999bd1f762da7d044613a4828efe` | Peer native CPU adapter available; author FP32 passes but BF16 P3/P7 logits fail original budget; no full checkpoint qualification | Full checkpoint and quantized GPU qualification pending |
| [HRM-Text-1B](https://huggingface.co/sapientinc/HRM-Text-1B/tree/22097cbcecdd1301afe30a19a3ee61b96a9863e5) | `22097cbcecdd1301afe30a19a3ee61b96a9863e5` | Peer native full32-arm run archived; own INT4 export/continuation CPU checks and four tiny INT4 Graph/KV GPU cases pass in draft #23 | Full checkpoint and quantized GPU qualification pending |

The additional family inventories do not replace the four active model jobs.
They share the same requirements for fixed source/data versions, an explicit
adapter manifest, real loop boundaries, original-model comparison, native
exported quality, and high-concurrency/multiple-batch serving. Runtime-specific
context limits and attention semantics must be registered before generating
each model's trace; a nominal loop count does not make different architectures
equivalent.

LoopFormer and Parcae also appear in the external LoopQ comparison contract.
Testing their native checkpoints does not constitute reproducing LoopQ's
quantization algorithm; its separate [reproduction audit](EXTERNAL_BASELINE.md)
remains `not_reproduced`.

Original weights remain while required by active or explicitly registered future
readers. Completed duplicate Ouro CPU weights have already been removed after
verifying the retained node copies. New full-weight downloads wait for space and
their upcoming execution slot; small immutable configs and inventories are
prepared independently.

All five additional family tokenizer splits are now frozen and integrity checked;
see [family data protocols and counts](FAMILY_DATA.md). This prepares evaluation
inputs and does not qualify model quality or performance.
