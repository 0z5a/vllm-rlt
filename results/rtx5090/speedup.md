# Ouro-1.4B speculative PD E2E

NVIDIA GeForce RTX 5090; two replicas versus 1P1D. 4 requests × 128 prompt / 32 output tokens; BF16, Triton, d=2/D=4, K=4. Startup and warmup excluded; prefill, NIXL transfer, speculative decode and KV commit included. 5 alternating paired trials; all output tokens and exit depths matched.

| GPUs | Replicas median s | PD median s | PD speedup | Output |
|---:|---:|---:|---:|---|
| 2 | 1.788 | 2.732 | 0.654× | exact match |
