# SPDX-License-Identifier: Apache-2.0
"""HRM-Text: one scheduler step is L_cycles low updates then one high update.

Each module application has distinct KV planes. The runner retains [z_H, z_L];
prefix-LM prompts must complete together before any decode token is admitted.
"""

import json
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from vllm_rlt.core.kv_cache_manager import KVCacheManager, _PreparedKVBatch
from vllm_rlt.layers import RotaryEmbedding


@dataclass(frozen=True)
class HrmTextConfig:
    vocab_size: int = 65536
    n_embd: int = 1536
    intermediate_size: int = 4096
    module_layers: int = 16
    num_attention_heads: int = 12
    num_key_value_heads: int = 12
    head_dim: int = 128
    H_cycles: int = 2
    L_cycles: int = 3
    max_position_embeddings: int = 4096
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    initializer_range: float = 0.025515518153991442
    embedding_scale: float = 39.191835884530846
    prefix_lm: bool = True
    bos_token_id: int = 6
    eos_token_id: int = 11
    pad_token_id: int = 5

    def __post_init__(self):
        values = asdict(self)
        for name in (
            "vocab_size",
            "n_embd",
            "intermediate_size",
            "module_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "H_cycles",
            "L_cycles",
            "max_position_embeddings",
        ):
            if type(values[name]) is not int or values[name] <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.n_embd != self.num_attention_heads * self.head_dim or self.head_dim % 2:
            raise ValueError("HRM requires hidden_size = heads * even head_dim")
        if self.num_key_value_heads != self.num_attention_heads:
            raise ValueError("HRM-Text requires equal query and KV heads")
        for name in ("rms_norm_eps", "rope_theta", "initializer_range", "embedding_scale"):
            if not math.isfinite(values[name]) or values[name] <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if type(self.prefix_lm) is not bool:
            raise ValueError("prefix_lm must be boolean")

    @property
    def hidden_size(self) -> int:
        return 2 * self.n_embd

    @property
    def num_hidden_layers(self) -> int:
        return (self.L_cycles + 1) * self.module_layers

    @property
    def total_ut_steps(self) -> int:
        return self.H_cycles

    @classmethod
    def from_dict(cls, values: dict) -> "HrmTextConfig":
        if values.get("model_type") != "hrm_text":
            raise ValueError("Expected model_type=hrm_text")
        if values.get("architectures") != ["HrmTextForCausalLM"]:
            raise ValueError("Expected HrmTextForCausalLM architecture")
        if values.get("tie_word_embeddings", False):
            raise ValueError("HRM-Text uses an untied LM head")
        names = {f.name for f in fields(cls)}
        config = {key: value for key, value in values.items() if key in names}
        config["n_embd"] = values["hidden_size"]
        config["module_layers"] = values["num_hidden_layers"]
        return cls(**config)

    def to_dict(self) -> dict:
        values = asdict(self)
        values["hidden_size"] = values.pop("n_embd")
        values["num_hidden_layers"] = values.pop("module_layers")
        return dict(
            model_type="hrm_text",
            architectures=["HrmTextForCausalLM"],
            tie_word_embeddings=False,
            **values,
        )


class HrmAttention(nn.Module):
    def __init__(self, config: HrmTextConfig):
        super().__init__()
        self.config = config
        self.gqkv_proj = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.o_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

    def forward(self, hidden, cos, sin, layer, batch, cache):
        cfg = self.config
        gate, q, k, v = (
            self.gqkv_proj(hidden)
            .reshape(-1, 4 * cfg.num_attention_heads, cfg.head_dim)
            .chunk(4, dim=1)
        )

        def rotate(x):
            left, right = x.float().chunk(2, dim=-1)
            return (x.float() * cos + torch.cat((-right, left), -1) * sin).to(x.dtype)

        q, k = rotate(q), rotate(k)
        cache._write_prepared(layer, batch, k, v)
        attention = cache._attend_prepared(layer, batch, q)
        return self.o_proj((torch.sigmoid(gate) * attention).flatten(1))


class HrmMLP(nn.Module):
    def __init__(self, config: HrmTextConfig):
        super().__init__()
        self.gate_up_proj = nn.Linear(config.n_embd, 2 * config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.n_embd, bias=False)

    def forward(self, hidden):
        gate, up = self.gate_up_proj(hidden).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)


class HrmBlock(nn.Module):
    def __init__(self, config: HrmTextConfig):
        super().__init__()
        self.eps = config.rms_norm_eps
        self.attn = HrmAttention(config)
        self.mlp = HrmMLP(config)

    def forward(self, hidden, cos, sin, layer, batch, cache):
        hidden = hidden + self.attn(
            F.rms_norm(hidden, (hidden.shape[-1],), eps=self.eps),
            cos,
            sin,
            layer,
            batch,
            cache,
        )
        return hidden + self.mlp(F.rms_norm(hidden, (hidden.shape[-1],), eps=self.eps))


class HrmModule(nn.Module):
    def __init__(self, config: HrmTextConfig):
        super().__init__()
        self.layers = nn.ModuleList(HrmBlock(config) for _ in range(config.module_layers))
        self.eps = config.rms_norm_eps

    def forward(self, hidden, cos, sin, first_layer, batch, cache):
        for offset, block in enumerate(self.layers):
            hidden = block(hidden, cos, sin, first_layer + offset, batch, cache)
        return F.rms_norm(hidden, (hidden.shape[-1],), eps=self.eps)


class HrmTextForCausalLM(nn.Module):
    def __init__(self, config: HrmTextConfig):
        super().__init__()
        self.config = config
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(config.vocab_size, config.n_embd)
        self.model.z_L_init = nn.Parameter(torch.zeros(config.n_embd), requires_grad=False)
        self.model.H_module = HrmModule(config)
        self.model.L_module = HrmModule(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.model.rotary_emb = RotaryEmbedding(config.head_dim, config.rope_theta)

    def prelude(self, token_ids: torch.Tensor) -> torch.Tensor:
        z_h = self.model.embed_tokens(token_ids) * self.config.embedding_scale
        z_l = self.model.z_L_init.expand_as(z_h)
        return torch.cat((z_h, z_l), dim=-1)

    def recurrent_prepared(
        self,
        hidden: torch.Tensor,
        batch: _PreparedKVBatch,
        cache: KVCacheManager,
        *,
        compute_gate: bool = True,
    ):
        z_h, z_l = hidden.chunk(2, dim=-1)
        freqs = batch.position_ids.float().unsqueeze(1) * self.model.rotary_emb.inv_freq.float()
        angles = torch.cat((freqs, freqs), dim=-1).unsqueeze(1)
        cos, sin = angles.cos(), angles.sin()
        for cycle in range(self.config.L_cycles):
            z_l = self.model.L_module(
                z_l + z_h, cos, sin, cycle * self.config.module_layers, batch, cache
            )
        z_h = self.model.H_module(
            z_h + z_l, cos, sin, self.config.L_cycles * self.config.module_layers, batch, cache
        )
        gate = hidden.new_full((len(hidden),), -1e4) if compute_gate else None
        return torch.cat((z_h, z_l), dim=-1), gate

    def recurrent(self, hidden, request_ids, depths, positions, cache, *, compute_gate=True):
        batch = cache._prepare_batch(request_ids, depths, positions)
        return self.recurrent_prepared(hidden, batch, cache, compute_gate=compute_gate)

    def coda(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden[:, : self.config.n_embd])

    @classmethod
    def from_pretrained(cls, path_or_repo, *, revision=None, device="cpu", dtype=torch.bfloat16):
        from safetensors import safe_open

        if dtype not in (torch.float32, torch.float16, torch.bfloat16):
            raise ValueError("dtype must be float32, float16, or bfloat16")
        folder = Path(path_or_repo).expanduser()
        if not folder.is_dir():
            from vllm_rlt.models import resolve_model_config

            source, _, _ = resolve_model_config(path_or_repo, revision=revision)
            folder = Path(source)
        config = HrmTextConfig.from_dict(json.loads((folder / "config.json").read_text()))
        with torch.device("meta"):
            model = cls(config)
        with safe_open(folder / "model.safetensors", framework="pt", device="cpu") as checkpoint:
            expected = model.state_dict()
            if set(checkpoint.keys()) != set(expected):
                raise ValueError("HRM checkpoint tensor keys differ")
            for name, target in expected.items():
                if tuple(checkpoint.get_slice(name).get_shape()) != tuple(target.shape):
                    raise ValueError(f"HRM checkpoint shape mismatch: {name}")
            for name in checkpoint.keys():
                parent, field = name.rsplit(".", 1)
                tensor = checkpoint.get_tensor(name)
                if not tensor.is_floating_point():
                    raise ValueError(f"HRM checkpoint tensor must be floating point: {name}")
                model.get_submodule(parent).register_parameter(
                    field, nn.Parameter(tensor.to(device=device, dtype=dtype), requires_grad=False)
                )
        model.model.rotary_emb.inv_freq = model.model.rotary_emb.frequencies(device=device)
        return model.eval()
