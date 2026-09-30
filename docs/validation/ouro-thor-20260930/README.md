# Ouro RL CUDA graphs on Thor — 2026-09-30

Recurrent CUDA graph replay remains active after in-place full policy publication.
This continues [the synchronous rollout draft](https://github.com/0z5a/vllm-rlt/pull/1)
for [VIME integration in issue 70](https://github.com/ThinkFlowLab/vllm-rlt/issues/70#issuecomment-5887476683).

## Full model generation timing

| K | Eager (s) | CUDA graphs (s) | Speedup | Time saved |
|---|---:|---:|---:|---:|
| 2 | 1.0880 | 0.9597 | 1.134× | 11.79% |
| 3 | 1.5567 | 1.4105 | 1.104× | 9.40% |
| 4 | 1.9689 | 1.8269 | 1.078× | 7.21% |

Ouro-1.4B, BF16, one NVIDIA Thor (SM110), Triton attention, two identical prompts,
32-token cap per request, five measured repeats per K. Each arm is warmed at each
K before timing; repeat order alternates. Sampling seeds are identical across
arms. Wall time is bounded by CUDA synchronization and includes generation,
bookkeeping and sampling. Capture and model loading are outside these medians.
All 15 pairs have identical token IDs and block-token counts; logprobs satisfy
`torch.testing.assert_close`. Generation after a full policy update also matches.

The CUDA unit test exercises K=4/2/3/4 before and after publication, unchanged
parameter addresses, no recapture, cache release, versions, logprobs and work
counts. It passed on Thor (`Ran 1 test in 2.742s`, `OK`).

The companion VIME run completed real GRPO updates and a fresh-process resume.
Its final checkpoint matches continuous eager training across all 272 tensors,
including FP32 master weights and both Adam moments, with matching optimizer
step metadata. Detailed training timings are in the companion report.

## Reproduce

```bash
PYTHONPATH=. python tests/test_rl_cuda_graphs.py
PYTHONPATH=. python benchmarks/rl_cuda_graphs.py /models/Ouro-1.4B rollout-benchmark.json
```

Model: `ByteDance/Ouro-1.4B`, revision
`574fa66cb8bf5abdc979642d01cf2b79b16bfab1`; `model.safetensors` SHA256
`58872a72616c736595b8b7662079c5b12c5a162ec16eae94f21c348dfa9885af`.
Runtime: Python 3.12.3, PyTorch 2.13.0+cu130, Triton 3.7.1, Transformers 5.17.0,
NumPy 2.3.5, driver 595.78, aarch64. See [raw measurements](rollout-benchmark.json)
and [package versions](runtime.json). Existing packages and driver were used.
This validates fixed-depth synchronous RL with recurrent graphs; early exit,
PD and speculative RL remain outside this change.
