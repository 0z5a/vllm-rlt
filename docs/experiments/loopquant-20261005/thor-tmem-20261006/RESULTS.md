# Thor SM110 TMEM projection compile results

Existing Python 3.12, Triton 3.7.1 and CUDA 13.2; BF16 inputs/output and FP32 accumulation. All six synthetic Ouro-dimension projection shapes compile offline for sm_110a with allocation, MMA, load and deallocation instructions.

| M | N | K | Offline compilation | GPU correctness | Speedup | Memory saved |
|---:|---:|---:|---|---|---|---|
| 1 | 2048 | 2048 | Pass | Unmeasured | Unmeasured | Unmeasured |
| 32 | 2048 | 2048 | Pass | Unmeasured | Unmeasured | Unmeasured |
| 64 | 2048 | 2048 | Pass | Unmeasured | Unmeasured | Unmeasured |
| 512 | 2048 | 2048 | Pass | Unmeasured | Unmeasured | Unmeasured |
| 32 | 5632 | 2048 | Pass | Unmeasured | Unmeasured | Unmeasured |
| 32 | 2048 | 5632 | Pass | Unmeasured | Unmeasured | Unmeasured |

The existing GPU attempt has no retrieved terminal or numerical result. Its last successful observation was a resource wait; current status is unverified. No additional GPU attempt was launched.

The planned microbenchmark compares the same BF16 inputs against Torch GEMM, with five warmups and five groups of 30 calls. Four repeated calls share weights but do not establish cross-loop TMEM residency. This is not full-model E2E or a FlashLoop comparison.

Published PTX copies replace only the compiler debug source-file path. The original PTX and execution evidence remain in private storage. Public-copy hashes are listed in manifest.json; original hashes in the compile summary identify the privately retained originals. PTX instruction bytes apart from the debug .file directive are unchanged. Earlier four-shape FP32-output compilation is separate from this BF16-output evidence.
