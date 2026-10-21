"""Dense functional HRM oracle, independent of paged cache and model forward."""

import torch
from torch.nn import functional as F


@torch.no_grad()
def dense_hrm_reference(model, tokens, prompt_length):
    config, weights = model.config, model.state_dict()
    count, width = len(tokens), config.n_embd
    rows = torch.arange(count, device=tokens.device)
    allowed = rows[None, :] <= rows[:, None]
    if config.prefix_lm:
        allowed |= rows[None, :] < prompt_length
    inverse = 1 / (
        config.rope_theta
        ** (torch.arange(0, config.head_dim, 2, device=tokens.device).float() / config.head_dim)
    )
    angle = rows.float()[:, None] * inverse
    cosine, sine = (
        torch.cat((angle, angle), -1).cos()[:, None],
        torch.cat((angle, angle), -1).sin()[:, None],
    )
    cache = {}

    def rms(value):
        return value * torch.rsqrt(value.square().mean(-1, keepdim=True) + config.rms_norm_eps)

    def linear(value, name):
        return F.linear(value, weights[name])

    def module(value, name, outer, offset):
        for layer in range(config.module_layers):
            base = f"model.{name}.layers.{layer}"
            gate, query, key, val = linear(rms(value), base + ".attn.gqkv_proj.weight").split(
                width, -1
            )
            gate, query, key, val = [
                x.reshape(count, config.num_attention_heads, config.head_dim)
                for x in (gate, query, key, val)
            ]
            rotated = []
            for item in (query, key):
                first, second = item.float().split(config.head_dim // 2, -1)
                rotated.append(
                    (item.float() * cosine + torch.cat((-second, first), -1) * sine).to(item.dtype)
                )
            query, key = rotated
            cache[outer, offset + layer] = key.clone(), val.clone()
            scores = torch.einsum("qhd,khd->hqk", query.float(), key.float()) / config.head_dim**0.5
            probabilities = scores.masked_fill(~allowed[None], -torch.inf).softmax(-1)
            attended = torch.einsum("hqk,khd->qhd", probabilities, val.float()).to(value.dtype)
            value = value + linear(
                (attended * gate.sigmoid()).reshape(count, width), base + ".attn.o_proj.weight"
            )
            g, u = linear(rms(value), base + ".mlp.gate_up_proj.weight").chunk(2, -1)
            value = value + linear(F.silu(g) * u, base + ".mlp.down_proj.weight")
        return rms(value)

    high = F.embedding(tokens, weights["model.embed_tokens.weight"]) * config.embedding_scale
    low = weights["model.z_L_init"].expand_as(high)
    states = []
    for outer in range(config.H_cycles):
        for inner in range(config.L_cycles):
            low = module(low + high, "L_module", outer, inner * config.module_layers)
        high = module(high + low, "H_module", outer, config.L_cycles * config.module_layers)
        states.append(torch.cat((high, low), -1))
    return states, linear(high, "lm_head.weight"), cache
