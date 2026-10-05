# Looped KV execution status

The campaign is in progress. The strongest completed fixed-work resident Graph
comparison regresses at all four registered points. Compact routing and replay
pass correctness checks, but natural HTTP Graph execution retains output/exit
differences. Complete all-model, quality, serving and context matrices remain
outstanding.

| GPU | B / C | Native Graph tokens/s | Alias + prefix Graph tokens/s | Paired speedup [95% CI] | Change |
|---|---:|---:|---:|---:|---:|
| RTX5090 #0 | 16 / 32 | 387.05 | 380.45 | 0.9825× [0.9790, 0.9837] | -1.75% |
| RTX5090 #0 | 32 / 64 | 452.56 | 432.99 | 0.9568× [0.9440, 0.9641] | -4.32% |
| RTX5090 #1 | 16 / 32 | 385.93 | 379.58 | 0.9797× [0.9787, 0.9892] | -2.03% |
| RTX5090 #1 | 32 / 64 | 453.33 | 435.25 | 0.9575× [0.9363, 0.9625] | -4.25% |

Seven balanced pairs per point/card, official Ouro-1.4B, BF16, P128/D128 and
complete fixed-schedule execution. All56 measured arms /2,688 requests /344,064
output tokens exactly match external references, including equal work and Graph
coverage. Both cards and IO remained quiet. These are within-session intervals,
not independent process-start repetitions or HTTP results. The comparison covers
the whole alias stack and does not isolate the prefix increment.
[Full performance report](resident-replay-performance.md).

| Exploration | Completed evidence | Remaining scope |
|---|---|---|
| Alias resolution and copy removal | Independent address/lifetime oracles, official-model parity and Nsight profile | Broader natural contexts and strong-backend comparison |
| Compact physical storage | Record accounting, GPU lifetime checks and official parity | Incremental paired speed and complete fixed-budget service matrix |
| Hybrid reader and staging | CPU/CUDA and official E2E/quality checks; five-shape pilot | Formal repeated attribution |
| Resident async/Graph | Rectangular and compact metadata ownership; all84 compact routing and24 replay arms exact | Natural Graph equivalence and broader contexts |
| Admission credits | Sudden full-depth completion; fixed6GiB residency32→38 | Same-budget goodput, work/quality attribution and broader stress |
| HTTP load | 24 real TCP arms /1,392 requests /72,384 tokens complete | Frozen SLO/open-loop curves and full formal cohorts |
| HRM prefix-LM | Native adapter,214 affected CPU checks,72 tiny CUDA arms;32 official sync arms exact through B512/C1024 (actual resident117) | Official async/Graph, quality and formal service performance |

The HTTP experiment reports zero non-Graph candidate token/exit differences.
Natural Graph totals are7 token /116 exit-different requests for alias and3 token
/105 exit-different requests for compact. Work and Graph coverage differ as well.
[Per-point diagnostic and measurement limits](http-diagnostic.md).

| Model | Official execution status |
|---|---|
| Ouro-1.4B | Original/delayed, fixed-work, quality and finite HTTP evidence; full matrix incomplete |
| Ouro-1.4B-Thinking | 21 official synchronous arms at B1–128/C2B exact; formal speed/quality pending |
| Ouro-2.6B | 24 official arms at B1–128/C2B; alias/compact exact, credits exit/work differences retained |
| Ouro-2.6B-Thinking | Release pinned; execution pending |
| Nanbeige4.2-3B | 32 official sync arms /8,104 requests /1,037,312 tokens exact through actual B512 residency; async/Graph, quality and formal service pending |
| Huginn-0125 | Tiny CUDA qualified; official asset readers and full execution pending |
| HRM-Text-1B | 32 official synchronous arms /8,104 requests /1,037,312 tokens exact; B512/C1024 caps at117 residents; broader matrix pending |
| LoopFormer | Native adapter, actual author-forward check and 72 tiny CUDA arms; official weights pending |
| Parcae | Public pinned config/source/card audit only; adapter and weights pending |

[Model evidence](models.md), [Ouro-2.6B](ouro-2.6b.md), [Nanbeige](nanbeige.md), [HRM contract](hrm-text.md),
[quality](quality.md), [delayed quality](delayed-quality.md),
[profiling](profile.md), [compact CUDA](compact-resident-cuda.md).
All negative results remain part of the evidence. Model files are retained while
required tests or registered readers remain; no family is declared complete from
a tiny-model check or one finite diagnostic.
