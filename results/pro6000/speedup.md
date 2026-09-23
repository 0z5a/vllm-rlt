# Ouro-1.4B equal-card E2E

Device: NVIDIA RTX PRO 6000 Blackwell Server Edition; 2 replicas versus 1P1D. 4 requests × 256 prompt / 32 output tokens; BF16, Triton, LAST_EXITED, fixed four loops. Model startup and warmup excluded; prefill, NIXL transfer, and decode included. 3 alternating trials with strict token and exit-depth equality.

| GPUs | Replicas median s | PD median s | PD speedup | Status |
|---:|---:|---:|---:|---|
| 2 | 2.067 | 2.249 | 0.919× | PASS |
