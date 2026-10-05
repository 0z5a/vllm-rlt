# Rectangular alias CUDA Graph path

Status: tiny-model CUDA qualification passed; official-weight diagnostic complete with exit-depth differences. The change builds on resident
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
| CUDA graph gate | 60 execution arms pass; 24 Graph arms execute 116 captures and 1,116 replays with zero fallbacks |
| Official Ouro1.4B B1–128/C2B | All 36 arms token-exact; 143 alias Graph request exit sequences differ from native Graph at B16/B32/B64; [full diagnostic](delayed-official.md) |

| Fixed-policy E2E comparison | Native Graph tokens/s | Alias Graph tokens/s | Speedup |
|---|---:|---:|---:|
| Official checkpoint | — | — | Not measured |

Frozen source `4adada84b66e31a71697a662893d007872f1e5b5` naturally completed0
on RTX5090/Torch2.12.1+cu130. All60 arms match native synchronous output/exit
sequences over two rounds and return all block credits/state slots. Cross-stream
allocation-reset and final-reader probes pass in both descriptor modes.
[All cases, Graph counters and immutable raw hash](evidence/alias-graph-cuda-v1.json).
