# Fixed INT4 weight baselines

The F0 execution packets are prepared. Full checkpoint results are **not run**;
the CPU check below qualifies only their small-model integration.

| Method | Calibration rows per physical input | Loop policy | Fixed packing |
|---|---:|---|---|
| RTN | None | Same packed W at every depth | Signed INT4, group128 |
| GPTQ-first | 62,993 | First native loop | Group128, block128, damping0.01 |
| GPTQ-all | 251,972 | All four native loops, row-weighted | Same |
| GPTQ-matched | 62,993 | Seed17 SHA selection of one loop per document/window/position | Same |

All methods retain one physical packed matrix per core projection and protect
embedding/head/gate/norm weights. Native execution rounds group scales to BF16
and uses the installed Torch tinygemm kernel. Q/K/V and gate/up share their input
Gram matrices; their distinct weight matrices are each packed once. There is no
loop-wise replication of weight storage and no online error-compensation module.

Calibration uses the previously frozen128 windows/112 documents, SHA256
`b33e74f657049f725ca1fb025213110789594d2118d148f85fb2bf192e620d68`.
The trace traverses the actual native BF16 core with distinct document/window IDs,
all four recurrence steps, real KV row metadata and no padding in the statistics.
It records per-loop row counts and energy and asserts the matched/first budget
identity before packing. This calibration is teacher forced; it does not replace
the separate closed-loop decode diagnostics.

The three policies together allocate12,759,072,768 bytes of FP32 Gram storage for
Ouro1.4B (24 layers; three2048-wide input groups and one5632-wide group per layer).
This is a configuration-derived allocation estimate, not a measured peak.
The runner records actual peak CUDA allocation and preserves factorization
failures. Ouro2.6B is not admitted with this same allocation strategy.

The independent frozen dev set remains128 windows/44 documents/62,361 targets,
SHA256 `325f23166ecec45f22a8a8254dc5edc7bf465b1d456beff801b58c7a07770c1c`.
BF16/RTN and each GPTQ export will be evaluated at R1/2/3/4. Native reload must
reproduce every encoded state tensor byte before the actual KV-path NLL runs.
Each arm retains window losses, document totals, exported-manifest hash and KV
drain status. No locked test or parameter search occurs in these packets.

| Packet | Work | State |
|---|---|---|
| [int4-dev-curves-v1](f0-fixed-diagnostics/int4-dev-curves-v1/manifest.json) | Eight BF16/RTN dev arms | Frozen; awaiting admission |
| [gptq-export-v1](f0-fixed-diagnostics/gptq-export-v1/manifest.json) | Native collection plus three GPTQ exports | Frozen locally; not uploaded/run |
| [gptq-dev-curves-v1](f0-fixed-diagnostics/gptq-dev-curves-v1/manifest.json) | Twelve GPTQ dev arms | Frozen locally; requires successful parent exports |

The tiny CPU integration passes all three policy exports,14 physical matrices
per policy,24 Gram summaries, first9/all36/matched9 row counts, finite decoded
weights and unchanged original/protected parameters. Temporary payloads were
removed. [Receipt](f0-fixed-diagnostics/gptq-export-pipeline-cpu-attempt1.json).
Existing unit coverage separately verifies exact Gram aggregation, selection and
padding exclusion. The GPU packet still must qualify the full exported artifacts.

| Method | Full checkpoint dev PPL | Native output tok/s | Speedup vs BF16 |
|---|---:|---:|---:|
| RTN | Not run | Not run | Not measured |
| GPTQ-first | Not run | Not run | Not measured |
| GPTQ-all | Not run | Not run | Not measured |
| GPTQ-matched | Not run | Not run | Not measured |

These fixed weight diagnostics do not establish an activation-scaling speedup.
G0 retains its earlier numerical failures; full QAT search and formal G2/G3 remain
subject to their registered gates.
