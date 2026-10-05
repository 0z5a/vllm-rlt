# SPDX-License-Identifier: Apache-2.0
"""LoopFormer with cached causal attention and a per-row recurrent clock."""

import json
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class LoopFormerConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 3
    n_head: int = 32
    n_embd: int = 2048
    intermediate_dim: int = 5120
    bias: bool = False
    dropout: float = 0.0
    bos_token_id: int = 50256
    eos_token_id: int = 50256
    pad_token_id: int = 50256

    def __post_init__(self):
        values = asdict(self)
        for name in ("block_size", "vocab_size", "n_layer", "n_head", "n_embd", "intermediate_dim"):
            if type(values[name]) is not int or values[name] <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.n_embd % self.n_head:
            raise ValueError("LoopFormer requires n_embd divisible by n_head")
        if self.dropout != 0 or type(self.bias) is not bool:
            raise ValueError("LoopFormer inference requires dropout=0 and boolean bias")

    @property
    def hidden_size(self):
        return self.n_embd + 1

    @property
    def num_hidden_layers(self):
        return self.n_layer

    @property
    def num_attention_heads(self):
        return self.n_head

    @property
    def num_key_value_heads(self):
        return self.n_head

    @property
    def head_dim(self):
        return self.n_embd // self.n_head

    @property
    def max_position_embeddings(self):
        return self.block_size

    @property
    def total_ut_steps(self):
        return 8

    @classmethod
    def from_dict(cls, values: dict):
        if values.get("model_type") != "loopformer":
            raise ValueError("Expected model_type=loopformer")
        if values.get("architectures") != ["LoopFormerGPTForCausalLM"]:
            raise ValueError("Expected LoopFormerGPTForCausalLM architecture")
        names = {f.name for f in fields(cls)}
        return cls(**{key: value for key, value in values.items() if key in names})

    def to_dict(self):
        return dict(
            model_type="loopformer", architectures=["LoopFormerGPTForCausalLM"], **asdict(self)
        )


class TimestepEmbedder(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(256, width), nn.SiLU(), nn.Linear(width, width))

    def forward(self, time):
        frequencies = torch.exp(
            -math.log(10000) * torch.arange(128, dtype=torch.float32, device=time.device) / 128
        )
        angle = time.float()[:, None] * frequencies
        embedding = torch.cat((angle.cos(), angle.sin()), -1).to(self.mlp[0].weight.dtype)
        return self.mlp(embedding)


class LoopFormerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.norm_1 = nn.RMSNorm(config.n_embd, elementwise_affine=False)
        self.norm_2 = nn.RMSNorm(config.n_embd, elementwise_affine=False)
        self.attn = nn.Module()
        self.attn.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.attn.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.mlp = nn.Module()
        self.mlp.c_fc = nn.Linear(config.n_embd, config.intermediate_dim, bias=config.bias)
        self.mlp.c_proj = nn.Linear(config.intermediate_dim, config.n_embd, bias=config.bias)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(config.n_embd, 4 * config.n_embd)
        )

    def forward(self, hidden, condition, layer, batch, cache):
        gate_attn, gate_mlp, scale_attn, scale_mlp = self.adaLN_modulation(condition).chunk(4, -1)
        projected = self.attn.c_attn(self.norm_1(hidden) * (1 + scale_attn))
        query, key, value = (
            part.reshape(-1, self.config.n_head, self.config.head_dim)
            for part in projected.chunk(3, -1)
        )
        cache._write_prepared(layer, batch, key, value)
        attended = cache._attend_prepared(layer, batch, query).flatten(1)
        hidden = hidden + gate_attn * self.attn.c_proj(attended)
        intermediate = self.mlp.c_fc(self.norm_2(hidden) * (1 + scale_mlp))
        return hidden + gate_mlp * self.mlp.c_proj(F.gelu(intermediate))


class LoopFormerForCausalLM(nn.Module):
    def __init__(self, config: LoopFormerConfig):
        super().__init__()
        self.config = config
        self.gpt = nn.Module()
        self.gpt.transformer = nn.Module()
        self.gpt.transformer.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.gpt.transformer.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.gpt.transformer.h = nn.Module()
        self.gpt.transformer.h.blocks = nn.ModuleList(
            LoopFormerBlock(config) for _ in range(config.n_layer)
        )
        self.gpt.transformer.norm_f = nn.RMSNorm(config.n_embd)
        self.gpt.time_embedder = TimestepEmbedder(config.n_embd)
        self.gpt.dt_embedder = TimestepEmbedder(config.n_embd)

    def prelude(self, token_ids):
        hidden = self.gpt.transformer.wte(token_ids)
        return torch.cat((hidden, hidden.new_zeros((len(hidden), 1))), -1)

    def recurrent_prepared(self, state, batch, cache, *, compute_gate=True):
        hidden, time = state[:, :-1], state[:, -1]
        # Carry the clock with resident hidden rows, including Graph replay and
        # padding. Learned positions are added only on the first recurrent step.
        position = self.gpt.transformer.wpe(batch.position_ids)
        hidden = hidden + torch.where(time[:, None] == 0, position, 0)
        delta = torch.full_like(time, 1 / 8)
        condition = self.gpt.time_embedder(time) + self.gpt.dt_embedder(delta)
        for layer, block in enumerate(self.gpt.transformer.h.blocks):
            hidden = block(hidden, condition, layer, batch, cache)
        gate = state.new_full((len(state),), -1e4) if compute_gate else None
        return torch.cat((hidden, (time + delta)[:, None]), -1), gate

    def recurrent(self, hidden, request_ids, depths, positions, cache, *, compute_gate=True):
        return self.recurrent_prepared(
            hidden,
            cache._prepare_batch(request_ids, depths, positions),
            cache,
            compute_gate=compute_gate,
        )

    def coda(self, hidden):
        hidden = self.gpt.transformer.norm_f(hidden[:, :-1])
        return F.linear(hidden, self.gpt.transformer.wte.weight)

    @classmethod
    def from_pretrained(cls, path_or_repo, *, revision=None, device="cpu", dtype=torch.bfloat16):
        from safetensors.torch import load_file

        folder = Path(path_or_repo).expanduser()
        if not folder.is_dir():
            from vllm_rlt.models import resolve_model_config

            source, _, _ = resolve_model_config(path_or_repo, revision=revision)
            folder = Path(source)
        config = LoopFormerConfig.from_dict(json.loads((folder / "config.json").read_text()))
        with torch.device("meta"):
            model = cls(config)
        model.load_state_dict(load_file(folder / "model.safetensors"), strict=True, assign=True)
        return model.requires_grad_(False).to(device=device, dtype=dtype).eval()
