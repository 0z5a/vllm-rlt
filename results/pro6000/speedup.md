# Ouro-1.4B fixed-depth greedy E2E

Device: NVIDIA RTX PRO 6000 Blackwell Server Edition (GPU 1). Model remains resident. Timings include prefill, draft, verification, and KV commit. Each arm has one warmup and paired inputs with alternating order; medians use three trials. A separate benchmark used GPU 0 during part of this run, so host-level contention may affect timings.

| Workload | γ | Baseline E2E s | Spec E2E s | Speedup | Accepted / drafted |
|---|---:|---:|---:|---:|---:|
| repeated | 1 | 3.285 | 1.646 | 1.996× | 93 / 93 |
| repeated | 2 | 3.285 | 1.202 | 2.734× | 120 / 126 |
| repeated | 4 | 3.285 | 0.737 | 4.457× | 147 / 147 |
| prose | 1 | 2.860 | 3.145 | 0.909× | 18 / 84 |
| prose | 2 | 2.860 | 3.089 | 0.926× | 27 / 156 |
| prose | 4 | 2.860 | 2.696 | 1.061× | 30 / 294 |
