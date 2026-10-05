# Delayed Ouro1.4B quality: sync, resident and Graph paths

All six arms use the same frozen GSM8K100 IDs/prompts, official checkpoint,
strict extraction rule and verified byte-stop table as the original-policy
[quality cohort](quality.md). The policy here is delayed threshold0.5/min2/max4;
B16/C100, greedy seed17, up to1,024 outputs, natural EOS and text stops. The
checkpoint is loaded once, with a fresh engine/cache per arm. Source
`77ccbbb1bcd3df16bd9fa3a248be2467aab10ecf`. These runs are not timing evidence.

| Path | Strict correct /100 | Wilson95% | Tokens | Token / exit different requests vs matching native mode | Submitted recurrent rows | Extra submitted rows beyond committed decode |
|---|---:|---|---:|---:|---:|---:|
| native-sync | 63 | 53.2–71.8% | 10,706 | 0 / 0 | 42,110 | 0 |
| alias-sync | 63 | 53.2–71.8% | 10,706 | 0 / 0 | 42,110 | 0 |
| native-async | 63 | 53.2–71.8% | 10,706 | 0 / 0 | 42,110 | 0 |
| alias-async | 63 | 53.2–71.8% | 10,706 | 0 / 0 | 42,110 | 0 |
| native-graph | 63 | 53.2–71.8% | 11,531 | 0 / 0 | 45,292 | 0 |
| alias-graph | 62 | 52.2–70.9% | 11,489 | 15 / 43 | 45,072 | 13 |

The four non-Graph paths produce identical complete request objects, including
all10,706 tokens, exit depths and text-stop offsets. All100 requests text-stop.
Each Graph path has99 text stops and one output-limit termination. Alias Graph
changes15 token sequences and43 exit sequences relative to native Graph. It gains
one correct answer and loses two (source IDs595,1069,1303), a net loss of one.
The paired accuracy difference is−1 percentage point, with a95% paired-bootstrap
interval of -5 to 2 points
(10,000 resamples, seed1705). This small cohort does not establish quality equivalence
or a stable degradation. It is an observed negative result, retained without
changing the policy, scorer or acceptance thresholds.

The independent fixed-schedule B16/B32 diagnostic yields exact native/alias
request objects; the serialized oracle matches all5,493 hidden/gate/sample
stages bitwise. These results qualify storage under matched work. They do not
erase the natural-scheduling quality difference or prove its cause.

The CUDA text-stop gate passes12 cases at B1/32/128 C2B, two ID-reuse rounds,
native/alias resident and Graph modes. Stops cross token boundaries, retain the
exact token sequence/byte offset, and return all cache blocks/state slots.
[CUDA gate and matched-schedule evidence](evidence/resident-schedule-cuda-v2.json).

| Matched-policy E2E speed comparison | Native tokens/s | Alias tokens/s | Speedup |
|---|---:|---:|---:|
| Quality cohort | — | — | Not measured |

All processes naturally exited0. The complete archive has independently matched
local/remote SHA256`4c005745908ae4fd978bd284a975c94b929a59bd57b82e9fa4cd6173782189f6`.
[Per-arm provenance, work, scores and discordant answers](evidence/delayed-quality-v1.json).
