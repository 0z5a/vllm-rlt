# Roadmap: faster Ouro inference first

Prioritize single-GPU Ouro latency and throughput while preserving request
isolation, sampling state and LAST-EXITED KV semantics. New comparisons use the
[precision policy](precision-policy.md); completed experiment details and
measurements live in the [results index](benchmarks.md).

## Milestones and dependencies

| Milestone | Deliverable | Completion criterion |
| --- | --- | --- |
| M0 — preview | Current synchronous engine and validation record | Implemented; historical validation limitations remain recorded. |
| M1 — baseline | BF16 benchmark, bounded profiler captures, and bottleneck report | Explicit accumulation policy, correct timing/replay accounting, raw results, variability, and a ranked next optimization. Retain the historical FP32 baseline. |
| Q1 — numerical validation, parallel with M1 | BF16 kernel/model comparisons and independent adaptive-history oracle; optional FP32 diagnostics | Justified BF16 fidelity/decision criteria, exact state invariants, and a separately reported investigation of historical failures. |
| M2 — reduce host overhead | Reuse batch metadata across layers; batch result transfers where profiling supports it | Preserve outputs, gates, cache isolation, and sampling state; demonstrate the declared end-to-end benefit. |
| M3 — recurrent CUDA graphs | Safe inactive rows, persistent buffers, then capture one recurrent traversal | Eager/graph equivalence, safe slot reuse, bounded graph memory, and measured benefit including routing overhead. |
| M4 — attention and KV efficiency | One measured attention or KV bottleneck per PR | Improve the chosen workload within declared latency, correctness, and memory guardrails. |
| Q2 — quality and comparison | Bounded task-quality screen and equivalent cached external baseline | State which dtype/depth policy is qualified and which speed/quality comparisons are supported. |
| M5 — asynchronous routing, later | Evaluate the paper's lookahead method only after the synchronous baseline is fast | Available or separately trained/calibrated gate, held-out quality evidence, and benefit over the optimized synchronous engine. |

Profiling determines the order of metadata, CUDA-graph, attention and KV work.
Numerical fidelity, task quality and performance are separate results; an old
experiment's fixed budget or acceptance threshold is not a new feature's API.
Asynchronous routing requires an available, calibrated lookahead gate and must
not reinterpret the released Ouro gate.
