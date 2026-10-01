# Training contract on Thor — 2026-10-01

These timings use full BF16 `ByteDance/Ouro-1.4B` at revision
`574fa66cb8bf5abdc979642d01cf2b79b16bfab1`, with the public `LLM` rollout/update
API on upstream `ecb1f8b505b7e831815b40aec3b4598619cca23a`.

| Rollout | Eager (s) | CUDA graphs (s) | Speedup | Time saved |
|---|---:|---:|---:|---:|
| Native K=2 decode | 1.0689 | 0.9939 | 1.076× | 7.02% |
| Native K=3 decode | 1.5190 | 1.4207 | 1.069× | 6.47% |
| Native K=4 decode | 1.9414 | 1.8234 | 1.065× | 6.08% |
| Self-speculative target K=4, draft K=2, 3 candidates | 1.4952 | 1.3741 | 1.088× | 8.11% |

Five alternating-order pairs after per-arm warmup and full-weight publication;
two prompts, 32 response tokens each, temperature=0.9 and fixed per-request seeds.
Returned selected-token raw logprobs, tokens, exit depths and weight versions
match exactly between eager and graph arms in every pair. Each native prompt
prefill uses full K=4. Measurements include logprob delivery and exclude model
loading and initial capture. Each speculative comparison uses the same speculative
algorithm in both arms. See [raw timing pairs](raw/shared-full-rollout-benchmark.json).

CPU tests pass all five checks; CUDA/Triton tests pass all four applicable checks.
The independent FP32 oracle covers processed probabilities with temperature and
top-k/top-p, speculative target scores after publication, early exit, async
stop/abort, prefix invalidation, stable parameter addresses and failed publication
recovery. The PD control test confirms a newer complete epoch recovers from a
partial peer commit; delivery is in-process, so IPC and NIXL are not covered.

Full-model VIME training and fresh-process checkpoint/HF export parity are
reported in [0z5a/vime#5](https://github.com/0z5a/vime/pull/5).
Real multi-GPU PD publication and rollout checks are recorded in the
[A100 report](../training-contract-a100-20261001/README.md); Thor has one GPU and no NIXL.
No packages, CUDA drivers or environments were updated. GPU runs use the existing
performance lock and finish naturally. Completed experiment weights/checkpoints
are removed after preserving logs and tensor hashes.
