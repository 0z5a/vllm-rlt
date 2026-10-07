# Ouro1.4B native FP8 dev evaluation

Every arm evaluated the same128 dev windows from44 original documents, with
62,361 next-token targets. R4, source72b8634, BF16 state/KV and the native
Triton cache backend are fixed. Each quantized model was exported, reloaded
and compared byte-for-byte before evaluating NLL through its native core.

| Policy | Dev PPL | PPL ratio vs BF16 | PPL change | One-sided 95% upper ratio |
|---|---:|---:|---:|---:|
| BF16 | 11.559050 | 1.000000 | +0.000% | 1.000000 |
| DYN | 11.649150 | 1.007795 | +0.779% | 1.009332 |
| SH1 | 11.640786 | 1.007071 | +0.707% | 1.008543 |
| ST2 | 11.650949 | 1.007950 | +0.795% | 1.009565 |
| LOOP | 11.652965 | 1.008125 | +0.812% | 1.009776 |

Intervals use10,000 paired bootstrap draws of original documents, seed20261005;
they recompute token-weighted aggregate NLL, then exponentiate its difference.
Windows from one document are never treated as independent observations.
All arms drain their cache completely. The four FP8 dev upper ratios are
below1.01. This is initial-policy dev evidence, not a locked-test G2 pass.
No GSM8K, ARC, locked WikiText or long-context gate has run.

SH1, ST2 and LOOP use the same fixed amax initialization from the128-window
calibration and the prefill/decode maximum. ST2 boundary2 and all spatial
granularities were fixed before these results; no clipping/boundary search
was performed. DYN retains per-row dynamic scales. The observed SH1 point
estimate is slightly below DYN/LOOP, but this small difference is not evidence
that sharing itself improves quality.

[Summary and fixed input hashes](ouro-native-dev-v1-summary.json),
[BF16/DYN raw files](dynamic-dev-v1/ouro-dynamic-dev-v1-file-manifest.json),
[static policies and INT4 raw files](quant-quality-v1/quant-quality-v1-file-manifest.json).
All six child processes across these two reservations and their controllers/SSH
exited naturally0. The four full FP8 exports remain on node for subsequent
quality and deployment checks; temporary tiny-model payloads were removed.
