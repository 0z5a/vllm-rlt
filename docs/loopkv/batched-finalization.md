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
| CUDA publication and official-weight equivalence | Queued |

| E2E speed comparison | Previous candidate tokens/s | Batched publication tokens/s | Speedup |
|---|---:|---:|---:|
| Same source and fixed workload | — | — | Not measured |

The earlier [Nsight profile](profile.md) observed many scalar metadata fills;
that is motivation for this experiment, not a measured benefit of the new
kernel. A quiet matched comparison against the parent and Native-S is required.
