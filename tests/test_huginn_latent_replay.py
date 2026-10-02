# SPDX-License-Identifier: Apache-2.0
"""Replay the conditional policy across batching, chunks and a fresh engine."""

import copy

import pytest
import torch
import torch.nn.functional as F

from tests.reference.huginn import dense_huginn_reference
from tests.test_huginn import tiny_huginn_config
from vllm_rlt import LLM, CacheConfig, SamplingParams, SchedulerConfig
from vllm_rlt.models.huginn import HuginnForCausalLM
from vllm_rlt.models.huginn_latents import replay_huginn_latents


def test_seeded_latents_do_not_depend_on_packing_or_global_rng():
    kwargs = dict(dtype=torch.bfloat16, device="cpu")
    packed = replay_huginn_latents(32, [42, 7, 42], [0, 0, 1], **kwargs)
    torch.manual_seed(999)
    reordered = replay_huginn_latents(32, [42, 42, 7], [1, 0, 0], **kwargs)
    assert torch.equal(packed, reordered[[1, 2, 0]])
    assert not torch.equal(packed[0], packed[2])


def test_native_rollout_logprobs_match_replayed_dense_and_fresh_publication():
    torch.manual_seed(42)
    model = HuginnForCausalLM(tiny_huginn_config(eos_token_id=None)).eval()
    prompts = [[4, 7, 9], [3, 12]]
    params = [
        SamplingParams(
            max_tokens=4,
            min_loops=3,
            max_loops=3,
            seed=seed,
            latent_seed=seed,
            temperature=0.8,
            logprobs=0,
            logprobs_mode="processed",
        )
        for seed in (11, 22)
    ]
    baseline = LLM(copy.deepcopy(model), cache_config=CacheConfig(num_blocks=32))
    expected = baseline.generate(prompts, params)
    for prompt, output, sampling in zip(prompts, expected, params, strict=True):
        tokens = torch.tensor(prompt + output.token_ids[:-1])
        state = replay_huginn_latents(
            model.config.n_embd,
            [sampling.latent_seed] * len(tokens),
            range(len(tokens)),
            dtype=torch.float32,
            device="cpu",
        )
        _, logits = dense_huginn_reference(model, tokens, state)
        response_logits = logits[len(prompt) - 1 :] / sampling.temperature
        scores = F.log_softmax(response_logits, -1).gather(
            1, torch.tensor(output.token_ids)[:, None]
        )[:, 0]
        torch.testing.assert_close(scores, torch.tensor(output.log_probs), atol=3e-6, rtol=3e-5)
    baseline.close()

    torch.manual_seed(2026)
    fresh = LLM(
        copy.deepcopy(model),
        cache_config=CacheConfig(num_blocks=32),
        scheduler_config=SchedulerConfig(
            prefill_chunk_size=1,
            max_num_batched_tokens=1,
            max_num_seqs=1,
        ),
    )
    fresh.start_weight_update(1)
    fresh.update_weights(model.named_parameters())
    fresh.finish_weight_update()
    actual = fresh.generate(prompts[::-1], params[::-1])[::-1]
    for before, after in zip(expected, actual, strict=True):
        assert after.weight_version == 1
        assert before.token_ids == after.token_ids
        assert before.exit_depths == after.exit_depths
        torch.testing.assert_close(
            torch.tensor(before.log_probs), torch.tensor(after.log_probs), atol=3e-6, rtol=3e-5
        )
    fresh.close()


@pytest.mark.parametrize("seed", [-1, True, 2**63])
def test_invalid_latent_seed_is_rejected(seed):
    with pytest.raises(ValueError, match="latent_seed"):
        SamplingParams(latent_seed=seed)
