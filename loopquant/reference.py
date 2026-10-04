# SPDX-License-Identifier: Apache-2.0
"""Ouro oracle following pinned official FP32 operator order; no model forward calls."""

import torch
from torch.nn import functional as F


@torch.inference_mode()
def ouro_reference(model, token_ids, loops, *, reference_dtype=torch.float32):
    """Functional causal oracle: no model modules, paged cache, or recurrent API.

    Matches official modeling_ouro.py@574fa66 FP32 rsqrt and multiply-scale order.
    Includes split-half
    RoPE, GQA, all four sandwich norms, and norm at each loop boundary.
    Returns every depth so intermediate hidden states and gate logits are checked.
    """
    config = model.config
    if reference_dtype not in (torch.bfloat16, torch.float32, torch.float64):
        raise ValueError("reference arithmetic must be BF16, FP32 or FP64")
    reduction_dtype = torch.float64 if reference_dtype == torch.float64 else torch.float32
    weights = {key: value.to(reference_dtype) for key, value in model.state_dict().items()}
    hidden = weights["model.embed_tokens.weight"][token_ids]
    length = len(token_ids)
    positions = torch.arange(length, dtype=reduction_dtype, device=hidden.device)
    frequencies = 1.0 / config.rope_theta ** (
        torch.arange(0, config.head_dim, 2, dtype=reduction_dtype, device=hidden.device)
        / config.head_dim
    )
    angles = torch.outer(positions, frequencies)
    cos = torch.cat([angles.cos(), angles.cos()], dim=-1)[:, None, :].to(hidden.dtype)
    sin = torch.cat([angles.sin(), angles.sin()], dim=-1)[:, None, :].to(hidden.dtype)
    future = torch.ones(length, length, dtype=torch.bool, device=hidden.device).triu(diagonal=1)

    def norm(value, name):
        normalized = value.to(reduction_dtype) * torch.rsqrt(
            value.to(reduction_dtype).square().mean(dim=-1, keepdim=True) + config.rms_norm_eps
        )
        return normalized.to(value.dtype) * weights[name + ".weight"]

    def linear(value, name):
        return value @ weights[name + ".weight"].T

    def rotary(value):
        split = config.head_dim // 2
        rotated = torch.cat([-value[..., split:], value[..., :split]], dim=-1)
        return value * cos + rotated * sin

    outputs = []
    for _ in range(loops):
        for layer in range(config.num_hidden_layers):
            prefix = f"model.layers.{layer}"
            attention_input = norm(hidden, prefix + ".input_layernorm")
            q = rotary(
                linear(attention_input, prefix + ".self_attn.q_proj").reshape(
                    length, config.num_attention_heads, config.head_dim
                )
            )
            k = rotary(
                linear(attention_input, prefix + ".self_attn.k_proj").reshape(
                    length, config.num_key_value_heads, config.head_dim
                )
            )
            v = linear(attention_input, prefix + ".self_attn.v_proj").reshape(
                length, config.num_key_value_heads, config.head_dim
            )
            groups = config.num_attention_heads // config.num_key_value_heads
            k = k.repeat_interleave(groups, dim=1)
            v = v.repeat_interleave(groups, dim=1)
            scores = torch.einsum("thd,shd->hts", q, k) * config.head_dim**-0.5
            scores.masked_fill_(future, -torch.inf)
            probabilities = scores.to(reduction_dtype).softmax(dim=-1).to(hidden.dtype)
            attention = torch.einsum("hts,shd->thd", probabilities, v).reshape(length, -1)
            attention = linear(attention, prefix + ".self_attn.o_proj")
            hidden = hidden + norm(attention, prefix + ".input_layernorm_2")
            mlp_input = norm(hidden, prefix + ".post_attention_layernorm")
            mlp = F.silu(linear(mlp_input, prefix + ".mlp.gate_proj")) * linear(
                mlp_input, prefix + ".mlp.up_proj"
            )
            hidden = hidden + norm(
                linear(mlp, prefix + ".mlp.down_proj"), prefix + ".post_attention_layernorm_2"
            )
        hidden = norm(hidden, "model.norm")
        gate = (
            linear(hidden, "model.early_exit_gate") + weights["model.early_exit_gate.bias"]
        ).squeeze(-1)
        outputs.append((hidden.clone(), gate, linear(hidden, "lm_head")))
    return outputs
