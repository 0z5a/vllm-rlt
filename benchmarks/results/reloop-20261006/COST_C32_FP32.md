# Re:Loop — complete FP32 C32 fixed-work cost

Two fresh processes per arm, A–B–C / C–B–A; 5 warmups and 5 measured trials per process. Native eager FP32/Triton, P512/T128, B16/C32, observed A=S=16, R4/common64 GiB pool. True-off baseline and matched off both have LoopCD disabled.

| Arm | TPS mean | TTFT ms | TPOT ms | Peak allocated GiB | Paired speed versus D4 off |
|---|---:|---:|---:|---:|---:|
| P4D4 off | 144.177 | 11945.17 | 74.59 | 69.385046 | 1.000000× |
| P4D3 off | 167.121 | 10934.28 | 59.38 | 69.385046 | 1.159342× |
| P4D3 guided | 164.701 | 11076.65 | 60.52 | 69.385168 | 1.142596× |

Guided / full-depth matched elapsed ratio: mean 1.142596×, range 1.123242–1.183086×. Guided / matched-off3 ratio: 0.985644×. Descriptive means and ranges; no confidence or production-serving claim. Guidance speed change versus D3off is -1.435607%; mean elapsed overhead is separately 1.475311%.

The predeclared 1.15× reference is not met by the observed mean. The plan calls it a continuation reference rather than a hard acceptance gate; the reproducible ~14% cost improvement can justify independent confirmation. This does not establish non-inferiority. Peak allocated memory savings are -131072 bytes: guidance adds 128 KiB; the common R4 pool is unchanged.

All 6 processes / 60 raw trials / 1,920 requests / 245,760 tokens passed the frozen auditor. All 99 manifest payload hashes and outer archive SHA `06d960ebb83e97eb1dee8af3a98b5c3cc956815eb558c7cdabccc37077ec5cb4` match. Archive is offbox; actor absence is verified. LoopCD owner explicitly returned the whole window after independent absence/offbox checks; the coordinator controls subsequent grants.
