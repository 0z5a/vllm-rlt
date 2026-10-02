# SPDX-License-Identifier: Apache-2.0
"""Opt-in, packing-independent Huginn noise for conditional policy replay."""

import hashlib
import math
from collections.abc import Sequence

import torch

HUGINN_LATENT_PROFILE = "like-init-cpu-f32-v1"


def replay_huginn_latents(
    width: int,
    seeds: Sequence[int],
    positions: Sequence[int],
    *,
    dtype: torch.dtype,
    device: torch.device | str,
) -> torch.Tensor:
    """Draw the like-init prior on CPU, scale once, then cast for execution.

    The persisted request seed and logical input position own each independent
    draw. CPU FP32 fixes the noise across accelerator types and batch layouts;
    this explicit profile leaves the legacy device/global-RNG policy unchanged.
    """
    std = math.sqrt(2 / (5 * width))
    states = []
    for seed, position in zip(seeds, positions, strict=True):
        identity = seed.to_bytes(8, "little") + position.to_bytes(8, "little")
        noise_seed = (
            int.from_bytes(hashlib.blake2b(identity, digest_size=8).digest(), "little") % 2**63
        )
        generator = torch.Generator(device="cpu").manual_seed(noise_seed)
        state = torch.empty(width, dtype=torch.float32, device="cpu")
        torch.nn.init.trunc_normal_(state, std=std, a=-3 * std, b=3 * std, generator=generator)
        states.append(state * math.sqrt(width))
    return torch.stack(states).to(device=device, dtype=dtype)
