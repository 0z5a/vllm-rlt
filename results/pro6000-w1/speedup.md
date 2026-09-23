# Ouro-1.4B equal-card E2E

Device: NVIDIA RTX PRO 6000 Blackwell Server Edition; 2 replicas versus 1P1D. 4 requests × 256 prompt / 128 output tokens; BF16, Triton, LAST_EXITED, fixed four loops. Model startup and warmup excluded; prefill, NIXL transfer, and decode included. 3 alternating trials with strict token and exit-depth equality. A separate DiffusionWorker occupied GPU 1 during a later trial; timings may include host contention.

| GPUs | Replicas median s | PD median s | PD speedup | Status |
|---:|---:|---:|---:|---|
| 2 | 6.773 | 7.506 | 0.902× | PASS |
