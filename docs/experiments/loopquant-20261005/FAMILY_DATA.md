# Frozen family tokenization

All five additional family datasets are prepared using existing pinned tokenizer
files and the same verified WikiText revision. No model was loaded and no locked
quality score was evaluated. Ouro1.4B and Ouro2.6B have byte-identical tokenizers
and retain the original registered v3 splits.

Each family uses seed17,512-token nonoverlapping windows,128calibration and128dev
windows, and all eligible locked-test windows. Tokenization changes both window
counts and selected documents; comparisons must remain paired within each model.
All file hashes, unique document/window identities and zero cross-split document
overlap were verified. [Validation receipt](family-data-v1/validation.json),
[preparation script](family-data-v1/prepare.py). Full token files remain in the
local `data-families-v1` directory and are identified by the manifest SHA256s.

Huginn and Nanbeige retain their official tokenizer postprocessor at the original
document start; later token windows do not insert additional special tokens.
Fast-JSON encoding matches all32 existing official CPU smoke inputs for each.
HRM, LoopFormer and Parcae use no added special tokens. Parcae's tokenizer has its
own official revision6247b5d, distinct from model revision4392844.

HRM uses a128-token bidirectional prefix followed by causal continuation. Its
minimum window length is256 and only continuation targets are scored. Other
families use causal next-token scoring with minimum128-token windows. This
protocol is fixed before quality evaluation, so HRM never sees future targets
inside the bidirectional prompt.

| Model | Calibration docs / targets | Dev docs / targets | Locked windows / docs / targets |
|---|---:|---:|---:|
| [huginn-0125](family-data-v1/huginn-0125-manifest.json) | 113 / 62,045 | 44 / 62,693 | 610 / 61 / 302,840 |
| [nanbeige-4.2-3b](family-data-v1/nanbeige-4.2-3b-manifest.json) | 112 / 63,777 | 46 / 64,219 | 594 / 61 / 294,915 |
| [hrm-text-1b](family-data-v1/hrm-text-1b-manifest.json) | 115 / 48,513 | 48 / 48,085 | 669 / 61 / 252,686 |
| [loopformer-3block-8iterations](family-data-v1/loopformer-3block-8iterations-manifest.json) | 106 / 64,247 | 49 / 63,148 | 569 / 61 / 281,633 |
| [parcae-370m](family-data-v1/parcae-370m-manifest.json) | 116 / 63,791 | 51 / 64,111 | 607 / 61 / 300,260 |

Targets here are a protocol-derived count, not a completed model evaluation.
Full-model BF16/native parity, quantized reload quality, locked G2 and serving
measurements remain required for every family.

[Performance-only serving inputs](FAMILY_SERVING.md) are also frozen separately from quality data.
