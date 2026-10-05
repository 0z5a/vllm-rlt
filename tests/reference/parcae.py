"""Dense Parcae equations independent of model forwards and paged storage."""

import math

import torch
from torch.nn import functional as F


@torch.inference_mode()
def dense_parcae_reference(model, tokens, initial_state):
    config, weights = model.config, model.state_dict()
    length, heads, width = len(tokens), config.num_attention_heads, config.head_dim
    phase = torch.outer(
        torch.arange(length, device=tokens.device).float(),
        config.rope_base ** (-torch.arange(0, width, 2, device=tokens.device).float() / width),
    )[:, None]
    future = torch.ones(length, length, device=tokens.device, dtype=torch.bool).triu(1)
    saved = {}

    def norm(x, name):
        normalized = x.float() * torch.rsqrt(
            x.float().square().mean(-1, keepdim=True) + config.norm_eps
        )
        return normalized.to(x.dtype) * weights[name + ".weight"]

    def linear(x, name):
        return F.linear(x, weights[name + ".weight"])

    def rotate_normalize(x):
        pairs = x.float().reshape(length, heads, -1, 2)
        real = pairs[..., 0] * phase.cos() - pairs[..., 1] * phase.sin()
        imag = pairs[..., 1] * phase.cos() + pairs[..., 0] * phase.sin()
        rotated = torch.stack((real, imag), -1).flatten(-2).to(x.dtype)
        normalized = rotated.float() * torch.rsqrt(
            rotated.float().square().mean(-1, keepdim=True) + torch.finfo(x.dtype).eps
        )
        return normalized.to(x.dtype)

    def block(x, name, layer, depth):
        source = norm(x, name + ".norm_1")
        q = rotate_normalize(linear(source, name + ".attn.c_q"))
        k = rotate_normalize(linear(source, name + ".attn.c_k"))
        v = linear(source, name + ".attn.c_v").reshape(length, heads, width)
        value_key, gate_key = f"value_embeds.{layer}.weight", name + ".attn.ve_gate.weight"
        if value_key in weights and gate_key in weights:
            values = weights[value_key][tokens].reshape(length, heads, width)
            gates = 2 * torch.sigmoid(F.linear(source[:, :32], weights[gate_key]))
            v = v + gates.unsqueeze(-1) * values
        saved[layer, depth] = (k.clone(), v.clone())
        scores = torch.einsum("thd,shd->hts", q, k) / math.sqrt(width)
        scores.masked_fill_(future, -torch.inf)
        probabilities = scores.float().softmax(-1).to(v.dtype)
        attended = torch.einsum("hts,shd->thd", probabilities, v)
        x = x + linear(attended.reshape(length, -1), name + ".attn.c_proj")
        mlp = 0.5 * linear(norm(x, name + ".norm_2"), name + ".mlp.fc").clamp_min(0).square()
        return x + linear(mlp, name + ".mlp.proj")

    p, c = config.n_layers_in_prelude, config.n_layers_in_recurrent_block
    injection = weights["transformer.wte.weight"][tokens]
    for i in range(p):
        injection = block(injection, f"transformer.prelude.{i}", i, 0)
    injection = norm(injection, "transformer.ln_prelude")
    states, state = [], initial_state
    dt = F.softplus(weights["transformer.adapter.dt_bias"])
    decay = torch.exp(-dt * weights["transformer.adapter.A_log"].exp())
    for depth in range(config.mean_recurrence):
        state = state * decay + dt * (injection @ weights["transformer.adapter.B"].T)
        for i in range(c):
            state = block(state, f"transformer.core_block.{i}", p + i, depth)
        states.append(state.clone())
    state = linear(state, "transformer.C")
    for i in range(config.n_layers_in_coda):
        state = block(state, f"transformer.coda.{i}", p + c + i, 0)
    return states, linear(norm(state, "transformer.ln_f"), "lm_head").float(), saved
