# Huginn conditional policy replay

`SamplingParams(latent_seed=...)` opts into `like-init-cpu-f32-v1`. Each logical
input position receives a separate truncated Gaussian draw on CPU in FP32,
using the checkpoint's like-init standard deviation and three-sigma bounds.
The draw is scaled by `sqrt(n_embd)` once and cast to the execution dtype.
Request seed and input position own the draw; packing, prefill chunk size,
arrival order and CUDA device do not. Requests without this flag retain the
existing global/device-RNG initialization.

Persist the latent seed, profile, model revision, execution dtype and PyTorch
version with the rollout. For PPO, replay the noise for every prompt and
response input position and recompute the prelude with the current weights.
Only the initial recurrent state is fixed. The likelihood is conditional on
that saved noise; the like-init prior is independent of the trainable weights.

This profile is explicit because CPU FP32 draws need not equal the legacy
device/dtype RNG stream. Selected-token logprob consistency must be checked
against teacher forcing before optimizing and again after fresh resume.
