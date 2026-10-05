"""Differentiable fixed-loop Nanbeige with its original residual and norm order."""

import math
from typing import Literal

import torch
from torch.nn import functional as F

from vllm_rlt.layers import apply_rotary_pos_emb
from vllm_rlt.models.nanbeige import NanbeigeForCausalLM

from .base import LoopOutput, QuantizedProjections


class NanbeigeAdapter(QuantizedProjections):
    def __init__(
        self,
        model: NanbeigeForCausalLM,
        *,
        attention_backend: Literal["eager", "sdpa"] = "eager",
    ) -> None:
        super().__init__(attention_backend)
        self.model = model

    def reference(self) -> "NanbeigeAdapter":
        return NanbeigeAdapter(self.model, attention_backend=self.attention_backend)

    def forward(self, tokens: torch.Tensor, valid: torch.Tensor, loops: int) -> LoopOutput:
        config = self.model.config
        if not 1 <= loops <= config.total_ut_steps:
            raise ValueError("loop count outside the registered model range")
        if tokens.shape != valid.shape or not valid.any(dim=1).all():
            raise ValueError("each sequence must have valid tokens and an aligned mask")
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
                    groups = config.num_attention_heads // config.num_key_value_heads
                    k, v = k.repeat_interleave(groups, 1), v.repeat_interleave(groups, 1)
                    scores = (q @ k.transpose(-2, -1)) / math.sqrt(config.head_dim)
                    scores = scores.masked_fill(~mask, torch.finfo(q.dtype).min)
                    attention = scores.softmax(-1, dtype=torch.float32).to(q.dtype) @ v
                attention = attention.transpose(1, 2).reshape(batch, length, -1)
                hidden = hidden + self._linear(
                    prefix + ".self_attn.o_proj", attn.o_proj, attention, loop, valid
                )
                normalized = layer.post_attention_layernorm(hidden)
                gate = self._linear(
                    prefix + ".mlp.gate_proj", layer.mlp.gate_proj, normalized, loop, valid
                )
                up = self._linear(
                    prefix + ".mlp.up_proj", layer.mlp.up_proj, normalized, loop, valid
                )
                hidden = hidden + self._linear(
                    prefix + ".mlp.down_proj", layer.mlp.down_proj, F.silu(gate) * up, loop, valid
                )
            if not config.skip_loop_final_norm:
                hidden = self.model.model.norm(hidden)
            states.append(hidden)
        # With skip_loop_final_norm, the official model still normalizes readout once.
        readout = self.model.model.norm(hidden) if config.skip_loop_final_norm else hidden
        return LoopOutput(self.model.lm_head(readout).float(), states)
