"""Dense functional LoopFormer oracle; no paged cache or native forward calls."""

import math

import torch
from torch.nn import functional as F


@torch.no_grad()
def dense_loopformer_reference(model, tokens):
    cfg, weights = model.config, model.state_dict()
    positions = torch.arange(len(tokens), device=tokens.device)
    hidden = F.embedding(tokens, weights["gpt.transformer.wte.weight"])
    hidden = hidden + F.embedding(positions, weights["gpt.transformer.wpe.weight"])
    mask = positions[None] <= positions[:, None]
    frequency = torch.exp(
        -math.log(10000) * torch.arange(128, dtype=torch.float32, device=tokens.device) / 128
    )

    def linear(value, path):
        return F.linear(value, weights[path + ".weight"], weights.get(path + ".bias"))

    def timestep(value, name):
        phase = value.float()[:, None] * frequency
        features = torch.cat((phase.cos(), phase.sin()), -1).to(hidden.dtype)
        return linear(F.silu(linear(features, name + ".mlp.0")), name + ".mlp.2")

    states, kv = [], {}
    time = hidden.new_zeros(1)
    delta = time.new_full((1,), 1 / 8)
    for depth in range(8):
        condition = timestep(time, "gpt.time_embedder") + timestep(delta, "gpt.dt_embedder")
        for layer in range(cfg.n_layer):
            base = f"gpt.transformer.h.blocks.{layer}"
            ga, gm, sa, sm = linear(F.silu(condition), base + ".adaLN_modulation.1").chunk(4, -1)
            projected = linear(F.rms_norm(hidden, (cfg.n_embd,)) * (1 + sa), base + ".attn.c_attn")
            query, key, value = (
                p.reshape(len(tokens), cfg.n_head, cfg.head_dim) for p in projected.chunk(3, -1)
            )
            kv[depth, layer] = key.clone(), value.clone()
            scores = torch.einsum("qhd,khd->hqk", query.float(), key.float()) / cfg.head_dim**0.5
            probabilities = scores.masked_fill(~mask[None], -torch.inf).softmax(-1)
            attended = torch.einsum("hqk,khd->qhd", probabilities, value.float()).to(hidden.dtype)
            hidden = hidden + ga * linear(attended.flatten(1), base + ".attn.c_proj")
            intermediate = linear(F.rms_norm(hidden, (cfg.n_embd,)) * (1 + sm), base + ".mlp.c_fc")
            hidden = hidden + gm * linear(F.gelu(intermediate), base + ".mlp.c_proj")
        time = time + delta
        states.append(torch.cat((hidden, time.expand(len(tokens), 1)), -1))
    normalized = F.rms_norm(hidden, (cfg.n_embd,), weights["gpt.transformer.norm_f.weight"])
    return states, F.linear(normalized, weights["gpt.transformer.wte.weight"]), kv
