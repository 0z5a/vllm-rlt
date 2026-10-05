# SPDX-License-Identifier: Apache-2.0
"""Parcae fixed-depth inference with boundary KV and token-dependent values."""

import json
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from vllm_rlt.layers import RMSNorm


@dataclass(frozen=True)
class ParcaeConfig:
    n_embd: int = 1024
    intermediate_size: int = 4096
    num_attention_heads: int = 8
    num_key_value_heads: int = 8
    n_layers_in_prelude: int = 4
    n_layers_in_recurrent_block: int = 4
    n_layers_in_coda: int = 4
    mean_recurrence: int = 8
    mean_backprop_depth: int = 4
    block_size: int = 2048
    vocab_size: int = 32768
    norm_eps: float = 1e-5
    rope_base: float = 50000.0
    eos_token_id: int | None = None

    def __post_init__(self):
        for name, value in asdict(self).items():
            if name not in ("norm_eps", "rope_base", "eos_token_id"):
                if type(value) is not int or value <= 0:
                    raise ValueError(f"{name} must be a positive integer")
        if (
            self.n_embd < 32
            or self.n_embd % self.num_attention_heads
            or self.head_dim % 2
            or self.num_key_value_heads != self.num_attention_heads
            or self.vocab_size > 65536
        ):
            raise ValueError("Parcae requires H>=32, equal Q/KV heads, even head size and V<=65536")
        if not all(math.isfinite(x) and x > 0 for x in (self.norm_eps, self.rope_base)):
            raise ValueError("norm_eps and rope_base must be finite and positive")

    @property
    def hidden_size(self):
        # Two exact uint8-valued channels retain token identity even in BF16.
        return 2 * self.n_embd + 2

    @property
    def head_dim(self):
        return self.n_embd // self.num_attention_heads

    @property
    def num_hidden_layers(self):
        return self.n_layers_in_prelude + self.n_layers_in_recurrent_block + self.n_layers_in_coda

    @property
    def total_ut_steps(self):
        return self.mean_recurrence

    @property
    def max_position_embeddings(self):
        return self.block_size

    @property
    def initializer_range(self):
        return math.sqrt(2 / (5 * self.n_embd))

    @classmethod
    def from_dict(cls, values: dict) -> "ParcaeConfig":
        if values.get("_class_name") != "ParcaeConfig":
            raise ValueError("Expected the published ParcaeConfig architecture")
        for name, expected in {
            "block_class_name": "TransformerPreNormBlock",
            "norm_class_name": "RMSNorm",
            "mlp_class_name": "BaseMLP",
            "nonlin_name": "ReLU2",
            "injection_type": "diagonal",
            "state_init": "like-init",
            "init_strategy": "scaled-zero",
            "mup_model_scaling_factor": 1,
            "prelude_norm": True,
            "qk_norm": True,
            "bias": False,
            "qk_bias": False,
            "tie_embeddings": True,
            "clip_qkv": None,
            "logit_softcap": None,
        }.items():
            if values.get(name, expected) != expected:
                raise ValueError(f"Unsupported Parcae {name}")
        for name, expected in {
            "padded_vocab_size": values["vocab_size"],
            "recurrent_embedding_dimension": values["n_embd"],
            "recurrent_intermediation_embedding_dimension": values["intermediate_size"],
        }.items():
            if values.get(name, expected) != expected:
                raise ValueError(f"Unsupported Parcae {name}")
        if values.get("recurrent_num_attention_heads") not in (None, values["num_attention_heads"]):
            raise ValueError("Recurrent and boundary head counts must match")
        rope = values.get("rope_settings", {"use_rope": True, "rope_condense_ratio": 1})
        if not rope["use_rope"] or rope["rope_condense_ratio"] != 1:
            raise ValueError("Parcae requires uncondensed RoPE")
        names = {field.name for field in fields(cls)}
        config = {key: value for key, value in values.items() if key in names}
        config["rope_base"] = rope.get("rope_base", values.get("rope_base", cls.rope_base))
        return cls(**config)

    def to_dict(self):
        return {"model_type": "parcae", "_class_name": "ParcaeConfig", **asdict(self)}


class DiagonalInjection(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.A_log = nn.Parameter(torch.zeros(width))
        self.dt_bias = nn.Parameter(torch.zeros(width))
        self.B = nn.Parameter(torch.eye(width))

    def forward(self, state, injection):
        dt = F.softplus(self.dt_bias)
        return state * torch.exp(-dt * self.A_log.exp()) + dt * F.linear(injection, self.B)


class ParcaeAttention(nn.Module):
    def __init__(self, config, cache_layer, author_layer):
        super().__init__()
        self.layer_idx = cache_layer
        self.heads, self.head_dim = config.num_attention_heads, config.head_dim
        self.c_q = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.c_k = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.c_v = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.ve_gate = (
            nn.Linear(32, self.heads, bias=False)
            if author_layer % 2
            == (config.n_layers_in_recurrent_block * config.mean_backprop_depth - 1) % 2
            else None
        )

    def forward(self, hidden, values, freqs, batch, cache):
        shape = (len(hidden), self.heads, self.head_dim)
        q, k, v = (
            self.c_q(hidden).view(shape),
            self.c_k(hidden).view(shape),
            self.c_v(hidden).view(shape),
        )
        if values is not None and self.ve_gate is not None:
            gate = 2 * torch.sigmoid(self.ve_gate(hidden[:, :32]))
            v = v + gate.unsqueeze(-1) * values.view(shape)
        pairs = torch.cat((q, k), dim=1).float().unflatten(-1, (-1, 2))
        real = pairs[..., 0] * freqs[..., 0] - pairs[..., 1] * freqs[..., 1]
        imag = pairs[..., 1] * freqs[..., 0] + pairs[..., 0] * freqs[..., 1]
        q, k = torch.stack((real, imag), dim=-1).flatten(-2).to(hidden.dtype).chunk(2, dim=1)
        q, k = F.rms_norm(q, (self.head_dim,)), F.rms_norm(k, (self.head_dim,))
        cache._write_prepared(self.layer_idx, batch, k, v)
        attended = cache._attend_prepared(self.layer_idx, batch, q)
        return self.c_proj(attended.reshape(len(hidden), -1))


class ParcaeMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc = nn.Linear(config.n_embd, config.intermediate_size, bias=False)
        self.proj = nn.Linear(config.intermediate_size, config.n_embd, bias=False)

    def forward(self, hidden):
        return self.proj(F.relu(self.fc(hidden)).square() * 0.5)


class ParcaeBlock(nn.Module):
    def __init__(self, config, cache_layer, author_layer):
        super().__init__()
        self.norm_1 = RMSNorm(config.n_embd, config.norm_eps)
        self.attn = ParcaeAttention(config, cache_layer, author_layer)
        self.norm_2 = RMSNorm(config.n_embd, config.norm_eps)
        self.mlp = ParcaeMLP(config)

    def forward(self, hidden, values, freqs, batch, cache):
        hidden = hidden + self.attn(self.norm_1(hidden), values, freqs, batch, cache)
        return hidden + self.mlp(self.norm_2(hidden))


class ParcaeForCausalLM(nn.Module):
    requires_boundary_kv = True

    def __init__(self, config: ParcaeConfig):
        super().__init__()
        self.config = config
        p, c = config.n_layers_in_prelude, config.n_layers_in_recurrent_block
        self.recurrent_kv_layers = tuple(range(p, p + c))
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                prelude=nn.ModuleList(ParcaeBlock(config, i, i) for i in range(p)),
                adapter=DiagonalInjection(config.n_embd),
                core_block=nn.ModuleList(ParcaeBlock(config, p + i, p + i) for i in range(c)),
                C=nn.Linear(config.n_embd, config.n_embd, bias=False),
                coda=nn.ModuleList(
                    ParcaeBlock(config, p + c + i, p + c * config.mean_recurrence + i)
                    for i in range(config.n_layers_in_coda)
                ),
                ln_f=RMSNorm(config.n_embd, config.norm_eps),
                ln_prelude=RMSNorm(config.n_embd, config.norm_eps),
            )
        )
        self.value_embeds = nn.ModuleDict(
            {
                str(i): nn.Embedding(config.vocab_size, config.n_embd)
                for i in range(config.num_hidden_layers)
                if i % 2 == (config.num_hidden_layers - 1) % 2
            }
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.transformer.wte.weight
        inv = config.rope_base ** (-torch.arange(0, config.head_dim, 2).float() / config.head_dim)
        angles = torch.outer(torch.arange(config.block_size).float(), inv)
        self.register_buffer(
            "freqs_cis", torch.stack((angles.cos(), angles.sin()), dim=-1)[None, :, None]
        )

    def _apply(self, fn, recurse=True):
        freqs = self.freqs_cis
        super()._apply(fn, recurse=recurse)
        self.freqs_cis = freqs.to(device=self.transformer.wte.weight.device, dtype=torch.float32)
        return self

    def _blocks(self, blocks, hidden, tokens, batch, cache):
        freqs = self.freqs_cis[0].index_select(0, batch.position_ids)
        for block in blocks:
            key = str(block.attn.layer_idx)
            values = self.value_embeds[key](tokens) if key in self.value_embeds else None
            hidden = block(hidden, values, freqs, batch, cache)
        return hidden

    def prelude_prepared(self, tokens, batch, cache):
        injection = self._blocks(
            self.transformer.prelude, self.transformer.wte(tokens), tokens, batch, cache
        )
        injection = self.transformer.ln_prelude(injection)
        state = torch.randn_like(injection)
        std = self.config.initializer_range
        nn.init.trunc_normal_(state, std=std, a=-3 * std, b=3 * std)
        token_bytes = torch.stack((tokens % 256, tokens // 256), dim=-1).to(injection.dtype)
        return torch.cat((state, injection, token_bytes), dim=-1)

    def recurrent_prepared(self, hidden, batch, cache, *, compute_gate=True):
        state, injection, token_bytes = hidden.split(
            (self.config.n_embd, self.config.n_embd, 2), dim=-1
        )
        tokens = token_bytes[:, 0].long() + 256 * token_bytes[:, 1].long()
        state = self.transformer.adapter(state, injection)
        state = self._blocks(self.transformer.core_block, state, tokens, batch, cache)
        gate = (
            torch.full((len(hidden),), -1e4, device=hidden.device, dtype=hidden.dtype)
            if compute_gate
            else None
        )
        return torch.cat((state, injection, token_bytes), dim=-1), gate

    def recurrent(self, hidden, request_ids, depths, positions, cache, *, compute_gate=True):
        batch = cache._prepare_batch(request_ids, depths, positions)
        return self.recurrent_prepared(hidden, batch, cache, compute_gate=compute_gate)

    def coda_prepared(self, hidden, batch, cache):
        state, _, token_bytes = hidden.split((self.config.n_embd, self.config.n_embd, 2), dim=-1)
        tokens = token_bytes[:, 0].long() + 256 * token_bytes[:, 1].long()
        state = self._blocks(self.transformer.coda, self.transformer.C(state), tokens, batch, cache)
        return self.lm_head(self.transformer.ln_f(state)).float()

    @classmethod
    def from_pretrained(cls, path_or_repo, *, revision=None, device="cpu", dtype=torch.bfloat16):
        folder = Path(path_or_repo).expanduser()
        if not folder.is_dir():
            from vllm_rlt.models import resolve_model_config

            source, _, _ = resolve_model_config(path_or_repo, revision=revision)
            folder = Path(source)
        config = ParcaeConfig.from_dict(json.loads((folder / "config.json").read_text()))
        checkpoint = torch.load(
            folder / "pytorch_model.bin", map_location="cpu", weights_only=True, mmap=True
        )
        if not torch.equal(checkpoint["lm_head.weight"], checkpoint["transformer.wte.weight"]):
            raise ValueError("Parcae tied checkpoint embeddings differ")
        with torch.device("meta"):
            model = cls(config)
        model.load_state_dict(checkpoint, strict=True, assign=True)
        model.to(device=device, dtype=dtype)
        model.lm_head.weight = model.transformer.wte.weight
        return model.requires_grad_(False).eval()
