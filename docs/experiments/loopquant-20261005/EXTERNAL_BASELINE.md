# External LoopQ reproduction audit

Status on 2026-10-05: **not_reproduced**. No official code revision or exported artifact was identified in this audit. We checked the pinned paper, the author's publication page, the author's public repositories and GitHub repository-name search. The same-name `sumterhouse/LoopQ` is a small JavaScript project and is not treated as the paper's implementation. Search absence does not prove that code can never become available. [Author publications](https://rui-fang.github.io/publications/), [author repositories](https://github.com/Rui-Fang?tab=repositories).

| Protocol property | Published LoopQ | This strict-sharing experiment |
|---|---|---|
| Evaluated backbones | Ouro1.4B/2.6B, LoopFormer3×8, Parcae370M | Ouro1.4B/2.6B, Huginn, Nanbeige |
| Quantization | Symmetric RTN W4A4/W4A8, group32 | Separately labeled INT4 W4A16, FP8, NVFP4 |
| Calibration | 1024 Pile validation samples, maximum256 tokens | Fixed document-disjoint calibration/dev protocol |
| Selected groups | Both transformations and quantized weights vary across loops | One packed matrix per physical projection |
| Online additions | Transforms and cross-loop normalization/affine/low-rank correction | No new unmerged auxiliary operators in the strict track |

These published properties come from Sections4–5 and AppendixB of [LoopQ v1](https://arxiv.org/html/2605.16343v1). In particular, AppendixB.1 does not justify calling the selected groups a single-packed-weight deployment.

No accuracy, memory or throughput superiority over original LoopQ is claimed. A later paper-based reimplementation must have a distinct label, explicit departures, its own exported artifacts, original-model numerical checks and real E2E trials. An official reproduction additionally needs a verified upstream code/checkpoint revision and compatible existing dependencies; this audit does not authorize environment changes. Raw query/page hashes are recorded in `external-loopq-audit.json`; local page/query snapshots remain in `evidence/external-loopq/`.
