# Model coverage — 2026-10-06

The execution plan names Ouro-1.4B, Ouro-2.6B and Huginn. The broader looped-model
coverage also includes Nanbeige, LoopFormer, Parcae and HRM-Text. This is the
execution queue: a pinned checkpoint or downloaded file is not a passed model
test. No row has completed quantized G0–G3 or formal paired serving measurements.

| Checkpoint | Immutable revision | Current evidence | Quantized E2E |
|---|---|---|---|
| [Ouro-1.4B](https://huggingface.co/ByteDance/Ouro-1.4B/tree/574fa66cb8bf5abdc979642d01cf2b79b16bfab1) | `574fa66cb8bf5abdc979642d01cf2b79b16bfab1` | All 128 official BF16 eager CPU comparisons exact; 128calibration and five128-window dev arms complete; native differences retained | Eight C32/B128/512 smoke arms measured; formal G3 pending |
| [Ouro-2.6B](https://huggingface.co/ByteDance/Ouro-2.6B/tree/1ed04250da1a9936042725d302e81c8fa2ab5abd) | `1ed04250da1a9936042725d302e81c8fa2ab5abd` | All 128 official BF16 eager CPU comparisons exact; verified node weights retained | Pending |
| [Huginn-0125](https://huggingface.co/tomg-group-umd/huginn-0125/tree/bb6621b65e90b6a4b9b29ef88dc83866d450470c) | `bb6621b65e90b6a4b9b29ef88dc83866d450470c` | All96 official BF16 SDPA CPU cases exact across R8/16/32; seeded native NLL and INT4/FP8 export implemented in [draft #10](https://github.com/0z5a/vllm-rlt/pull/10), CPU490 passed; boundary-prefill repaired; H20 INT4 2/2 and DYN FP8 4/4 pass, first SH1 sequence check fails; 11 FP8 cases and two controls unrun | Pending |
| [Nanbeige4.2-3B](https://huggingface.co/Nanbeige/Nanbeige4.2-3B/tree/b82e54bd609793562a75cbf9337970a93369eab5) | `b82e54bd609793562a75cbf9337970a93369eab5` | All64 official BF16 eager CPU cases exact across R1/R2; 32 native tiny FP8 GPU cases passed | Pending |
| [LoopFormer 3-block/8-iteration](https://huggingface.co/armenjeddi/LoopFormer-3block-8iterations/tree/2b4fbaaf4e2510353ef7cfe07a07c671b5226739) | `2b4fbaaf4e2510353ef7cfe07a07c671b5226739` | All48 official BF16 dense SDPA cases exact at R8 across B1/B2, including384 states; four tiny INT4 Graph/KV GPU cases and16 H20 FP8 cases pass in [draft #23](https://github.com/0z5a/vllm-rlt/pull/23) | Full quantized quality and serving pending |
| [Parcae-370M](https://huggingface.co/SandyResearch/parcae-370m/tree/439284464ee4999bd1f762da7d044613a4828efe) | `439284464ee4999bd1f762da7d044613a4828efe` | All96 official BF16 dense SDPA cases exact across R1/4/8, including416 states; native NLL, INT4/FP8 export and matched Q0 initialization implemented in [draft #27](https://github.com/0z5a/vllm-rlt/pull/27), CPU677 passed with2 strict retained BF16 failures; boundary-prefill repaired; H20 FP8 16/16 and native eager INT4 pass, compact/static synchronous INT4 rejected | Remaining INT4 modes and full E2E pending; full-checkpoint native G0 unresolved |
| [HRM-Text-1B](https://huggingface.co/sapientinc/HRM-Text-1B/tree/22097cbcecdd1301afe30a19a3ee61b96a9863e5) | `22097cbcecdd1301afe30a19a3ee61b96a9863e5` | Own full-gradient/prefix-aware adapter and FP8 paths pass666 CPU tests; four tiny INT4 and16 H20 FP8 Graph/KV cases pass in draft #23; all112 official-weight CPU comparisons through explicit SDPA shim pass bitwise | Full native GPU, quantized quality and E2E pending |

All families share the same requirements for fixed source/data versions, an explicit
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
inputs and does not qualify model quality or performance. Their
[512-request serving banks](FAMILY_SERVING.md) also retain model-specific context
limits and supported depth modes. Arrival rates and SLO thresholds still require
independent BF16 pilot measurements before candidate comparisons.

The native INT4 dev runner started on its assigned GPU before the remote SSH
endpoint became unreachable. Its child completion status remains unknown;
the allocation stays reserved and the job has not been duplicated. Independent H20 results and retained failures are recorded in the
[architecture audit](ASSOCIATIVE_IO.md); old-node packets have not been
resubmitted as a substitute for the unknown process. An SSH failure is not a model
test failure or a completed GPU experiment.

Latest H20 boundary repair verification: Parcae FP8 16/16 and Huginn INT4 2/2 pass. Huginn FP8 DYN 4/4 pass but first SH1 sequence comparison fails; Parcae INT4 native eager passes but compact/static synchronous construction fails. See [mixed result and raw evidence](ASSOCIATIVE_IO.md#h20-boundary-repair-verification). Full-checkpoint E2E and memory savings remain unmeasured for these families.


The latest full LoopFormer CUDA G0 attempt stops at mirror configuration HTTP403 before all64 checks (NOT_RUN); all12 raw files are verified offbox and the window is handed back. A new read-only peer-stream G0v3 is ready. Ouro-2.6B/Nanbeige v2 each freeze384 planned cache cases including eager and SDPA controls; earlier unsubmitted v1 packets are superseded. Tiny Ouro eager argmax discrepancy is retained. [Exact attempts, inputs, packet hashes and diagnostics](NATIVE_G0_QUEUE.md).
