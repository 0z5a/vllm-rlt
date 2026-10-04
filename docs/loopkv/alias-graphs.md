# Rectangular alias CUDA Graph path

Status: implemented, CUDA qualification pending. The change builds on resident
async alias routing and enables recurrent-core graphs for that path. Compact
graphs and synchronous alias graphs remain unsupported.

Each graph entry owns stable tables for all loop depths, query-depth storage and
zero first-alias offsets. Before replay, the runner copies the current bank's
tables and depths into that entry. The captured attention kernel resolves the
live exit map using the canonical-depth reader. Replays retain the native
last-event dependency and clone outputs before another entry can reuse the graph
pool. The host records both physical writes and alias-readable positions after
submission, and rejects writes to finalized alias positions.

The finite gate compares synchronous native, asynchronous native/alias and
native/alias Graph modes on the same tiny weights and mixed-depth trace. It
covers B1–128/C2B, two request-lifetime rounds, cancellation/reuse and both
mapped-host and H2D routing. Graph arms must report actual captures and repeated
replays; the configuration flag alone does not count as coverage.

| Validation | Result |
|---|---|
| Affected CPU alias/async/counter tests | 29 passed, 12 CUDA skips |
| CUDA graph gate | Pending |
| Official model/policy parity | Pending |

| Fixed-policy E2E comparison | Native Graph tokens/s | Alias Graph tokens/s | Speedup |
|---|---:|---:|---:|
| Official checkpoint | — | — | Not measured |
