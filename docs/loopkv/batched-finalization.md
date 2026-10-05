# Batched finalized-depth publication

The synchronous engine now groups requests exiting the same recurrent batch.
Alias and compact caches validate their completed versions and update host
validity/admission state, then publish exit-depth metadata in one staged tensor
and one Triton store kernel. Previously each token used a separate scalar fill.
Rectangular full-depth tokens still need no metadata write. Payload layout,
attention math, gate thresholds and executed versions are unchanged.

The runner publishes on its boundary stream after the contributing requests'
last core events, and records a completion event for following consumers. This
does not enable async or Graph for these cache modes; those paths remain
rejected until routing descriptors are adapted and tested. Native cache
finalization keeps its existing behavior.

| Validation | Result |
|---|---|
| Combined CPU cache / model / engine / async regression | 167 passed, 32 CUDA/official-input skips, 36 address subtests |
| Mixed request positions and R1/R2/R3/R4 histories | Correct canonical values at every query depth; finalized writes rejected |
| Huginn boundary lifetime and request reuse | CPU passed |
| CUDA mixed publication at N1/127/128/129/257 | Both alias and compact passed; exact canonical versions and unchanged payload |
| CUDA synchronous B1–128/C2B engine gate | Both storage paths pass exact tokens/exits and return all credits |
| Official-weight equivalence and timing | Pending |

| E2E speed comparison | Previous candidate tokens/s | Batched publication tokens/s | Speedup |
|---|---:|---:|---:|
| Same source and fixed workload | — | — | Not measured |

The earlier [Nsight profile](profile.md) observed many scalar metadata fills;
that is motivation for this experiment, not a measured benefit of the new
kernel. A quiet matched comparison against the parent and Native-S is required.

The CUDA gates naturally completed0 on RTX5090/Torch2.12.1+cu130 at frozen
source `c0180b7a7b2bf41c3ef4eae147cde939762e3088`. Per-storage payload gates
pass64 exact-version checks,46 bitwise attention checks and18 skipped-record
poison checks; alias also passes8 mixed-prefix checks. These are correctness
results using tiny inputs, with no timing claim.
[Results, immutable source and raw hashes](evidence/publication-cuda-v1.json).
