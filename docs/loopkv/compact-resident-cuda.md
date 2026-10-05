# Compact resident CUDA qualification

Compact resident routing and complete logical-schedule replay pass on RTX 5090.
The frozen source is `2b05bae6fa78d620ca6383fb16dd6d6f00c79727`, Torch2.12.1+cu130,
Triton3.7.1. The real checkpoint is Ouro-1.4B at
`574fa66cb8bf5abdc979642d01cf2b79b16bfab1`, BF16, delayed threshold0.5/min2/max4.

All84 tiny routing arms pass across B1/4/16/32/64/128, C2B, seven execution
paths and mapped-host/H2D descriptors. Both allocation-reset ordering and bank
retirement probes pass, as does cancellation while a compact-map reader is
outstanding. Each arm drains two request-ID lifetimes. P33/P34 crosses page
and attention-tile boundaries; padded graph rows do not reserve records.

The complete-schedule tests contain18 tiny and6 official arms: native capture,
native self-replay and compact replay at every point. An independent offbox
audit compares the raw complete request objects, work counters and Graph
coverage rather than trusting only the process return code. All comparisons
are exact and every arm returns its full block budget.

| B / C | Requests per arm | Output tokens per arm | Graph captures / replays / fallback | Exact full requests / work |
|---|---:|---:|---:|---|
| 16 / 32 | 32 | 4096 | 16 / 1391 / 0 | Yes / yes |
| 32 / 64 | 64 | 8192 | 16 / 383 / 784 | Yes / yes |

The official matrix contains288 requests /36,864 output tokens across six arms.
Its complete captured admission, recurrent batching and coda-delivery schedule
is replayed while real gate kernels still run. This does not prove equivalence
under a naturally changing schedule. The subsequent [HTTP diagnostic](http-load.md)
retains actual Graph token/exit differences.

| Matched resident E2E | Native tokens/s | Compact tokens/s | Speedup |
|---|---:|---:|---:|
| Official B16/C32 and B32/C64 | — | — | Not measured |

This was a correctness window sharing the other GPU with an independent
calibration job. Model reads and artifact writes used the original shared IO
lock. No timing, SLO or quality conclusion is drawn. All three qualification
children, the later HTTP child, both controllers and SSH completed naturally0.
All raw files were archived and independently hash-verified offbox.

[Detailed counters and raw hashes](evidence/compact-resident-cuda-v2.json).
Archive SHA256: `c5c77ea7e64e0dd7e7e4afc25d45ab4e15a14a27bc34a80525b6a368663fddb2`.
