"""Dense, differentiable Ouro path with explicit loop IDs and BF16 state boundaries."""

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F

from vllm_rlt.layers import apply_rotary_pos_emb
from vllm_rlt.models.ouro import OuroForCausalLM

from ..quantizers import FP8FakeLinear
from ..stats import ActivationStats


@dataclass
class LoopOutput:
    logits: torch.Tensor
    states: list[torch.Tensor]


class OuroAdapter(nn.Module):
    """Training/diagnostics adapter; native serving owns its separate paged cache.

    Each loop has independent causal K/V in full-sequence evaluation. This path
    intentionally avoids the inference cache's no-grad writes during QAT.
    """

    def __init__(
        self,
        model: OuroForCausalLM,
        *,
        attention_backend: Literal["eager", "sdpa"] = "eager",
    ) -> None:
        super().__init__()
        if attention_backend not in ("eager", "sdpa"):
            raise ValueError("reference attention must be explicitly eager or sdpa")
        self.model = model
        self.attention_backend = attention_backend
        self.quantized = nn.ModuleDict()
        self.statistics: dict[tuple[str, int], ActivationStats] = {}
        self.collect = False

    def _linear(
        self, name: str, module: nn.Linear, values: torch.Tensor, loop: int, valid: torch.Tensor
    ) -> torch.Tensor:
        if self.collect:
            self.statistics.setdefault((name, loop), ActivationStats()).update(
                values.flatten(0, 1), valid.flatten()
            )
        key = name.replace(".", "__")
        if key in self.quantized:
            loops = torch.full(values.shape[:-1], loop, device=values.device, dtype=torch.long)
            return self.quantized[key](values, loops)
        return module(values)

    def attach(self, name: str, quantized: FP8FakeLinear) -> None:
        self.quantized[name.replace(".", "__")] = quantized

    def forward(self, tokens: torch.Tensor, valid: torch.Tensor, loops: int) -> LoopOutput:
        if loops < 1 or loops > self.model.config.total_ut_steps:
            raise ValueError("loop count outside the registered model range")
        if tokens.shape != valid.shape or not valid.any(dim=1).all():
            raise ValueError("each sequence must have valid tokens and an aligned mask")
        config = self.model.config
        hidden = self.model.model.embed_tokens(tokens)
        batch, length = tokens.shape
        positions = torch.arange(length, device=tokens.device).repeat(batch)
        cos, sin = self.model.model.rotary_emb(hidden.flatten(0, 1), positions)
        causal = torch.ones(length, length, device=tokens.device, dtype=torch.bool).tril()
        mask = causal[None, None] & valid[:, None, None, :]
        states = []
        for loop in range(loops):
            for index, layer in enumerate(self.model.model.layers):
                prefix = f"model.layers.{index}"
                normalized = layer.input_layernorm(hidden)
                attn = layer.self_attn
                q = self._linear(prefix + ".self_attn.q_proj", attn.q_proj, normalized, loop, valid)
                k = self._linear(prefix + ".self_attn.k_proj", attn.k_proj, normalized, loop, valid)
                v = self._linear(prefix + ".self_attn.v_proj", attn.v_proj, normalized, loop, valid)
                q, k = apply_rotary_pos_emb(
                    q.reshape(batch * length, -1, config.head_dim),
                    k.reshape(batch * length, -1, config.head_dim),
                    cos,
                    sin,
                )
                q = q.reshape(batch, length, -1, config.head_dim).transpose(1, 2)
                k = k.reshape(batch, length, -1, config.head_dim).transpose(1, 2)
                v = v.reshape(batch, length, -1, config.head_dim).transpose(1, 2)
                if self.attention_backend == "sdpa":
                    attention = F.scaled_dot_product_attention(
                        q, k, v, attn_mask=mask, enable_gqa=True
                    )
                else:
                    # Follow the pinned official eager path and its BF16 boundaries.
                    groups = config.num_attention_heads // config.num_key_value_heads
                    k = k.repeat_interleave(groups, dim=1)
                    v = v.repeat_interleave(groups, dim=1)
                    scores = (q @ k.transpose(-2, -1)) * config.head_dim**-0.5
                    scores = scores.masked_fill(~mask, torch.finfo(q.dtype).min)
                    probabilities = scores.softmax(dim=-1, dtype=torch.float32).to(q.dtype)
                    attention = probabilities @ v
                attention = attention.transpose(1, 2).reshape(batch, length, -1)
                projected = self._linear(
                    prefix + ".self_attn.o_proj", attn.o_proj, attention, loop, valid
                )
                hidden = hidden + layer.input_layernorm_2(projected)
                normalized = layer.post_attention_layernorm(hidden)
                gate = self._linear(
                    prefix + ".mlp.gate_proj", layer.mlp.gate_proj, normalized, loop, valid
                )
                up = self._linear(
                    prefix + ".mlp.up_proj", layer.mlp.up_proj, normalized, loop, valid
                )
                down = self._linear(
                    prefix + ".mlp.down_proj", layer.mlp.down_proj, F.silu(gate) * up, loop, valid
                )
                hidden = hidden + layer.post_attention_layernorm_2(down)
            hidden = self.model.model.norm(hidden)
            states.append(hidden)
        return LoopOutput(self.model.lm_head(hidden), states)
