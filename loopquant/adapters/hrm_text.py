"""Differentiable HRM H/L recurrence with an explicit bidirectional prefix boundary."""

from typing import Literal

import torch
from torch.nn import functional as F

from vllm_rlt.models.hrm_text import HrmModule, HrmTextForCausalLM

from .base import LoopOutput, QuantizedProjections


class HrmTextAdapter(QuantizedProjections):
    def __init__(
        self, model: HrmTextForCausalLM, *, attention_backend: Literal["eager", "sdpa"] = "sdpa"
    ) -> None:
        super().__init__(attention_backend)
        self.model = model

    def reference(self) -> "HrmTextAdapter":
        return HrmTextAdapter(self.model, attention_backend=self.attention_backend)

    def forward(
        self,
        tokens: torch.Tensor,
        valid: torch.Tensor,
        loops: int,
        *,
        prefix_lengths: torch.Tensor | None = None,
    ) -> LoopOutput:
        config, model = self.model.config, self.model.model
        if loops != config.H_cycles:
            raise ValueError("HRM requires its complete H_cycles recurrence")
        if tokens.shape != valid.shape or tokens.ndim != 2:
            raise ValueError("tokens and valid rows must align")
        batch, length = tokens.shape
        lengths = valid.sum(-1)
        positions = torch.arange(length, device=tokens.device)
        if length > config.max_position_embeddings or not torch.equal(
            valid, positions < lengths[:, None]
        ):
            raise ValueError("HRM expects right-padded inputs within its context")
        if (
            prefix_lengths is None
            or prefix_lengths.shape != (batch,)
            or prefix_lengths.dtype != torch.long
            or not ((prefix_lengths >= 1) & (prefix_lengths <= lengths)).all()
        ):
            raise ValueError("each HRM row needs an explicit valid prefix length")
        allowed = (positions[None, :] <= positions[:, None])[None]
        if config.prefix_lm:
            allowed = allowed | (positions[None, None, :] < prefix_lengths[:, None, None])
        allowed = (allowed & valid[:, None, :])[:, None]
        angles = positions.float()[:, None] * model.rotary_emb.inv_freq.float()
        angles = torch.cat((angles, angles), -1)[None, :, None]
        cosine, sine = angles.cos(), angles.sin()

        def rotate(value: torch.Tensor) -> torch.Tensor:
            left, right = value.float().chunk(2, -1)
            return (value.float() * cosine + torch.cat((-right, left), -1) * sine).to(value.dtype)

        def module(hidden: torch.Tensor, layers: HrmModule, prefix: str, loop: int) -> torch.Tensor:
            for index, block in enumerate(layers.layers):
                name = f"model.{prefix}.layers.{index}"
                normalized = F.rms_norm(hidden, (config.n_embd,), eps=config.rms_norm_eps)
                projection = self._linear(
                    name + ".attn.gqkv_proj", block.attn.gqkv_proj, normalized, loop, valid
                )
                gate, q, k, v = projection.reshape(
                    batch, length, 4 * config.num_attention_heads, config.head_dim
                ).chunk(4, 2)
                q, k, v = rotate(q).transpose(1, 2), rotate(k).transpose(1, 2), v.transpose(1, 2)
                if self.attention_backend == "sdpa":
                    attended = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed)
                else:
                    scores = (q.float() @ k.float().transpose(-2, -1)) / config.head_dim**0.5
                    probabilities = scores.masked_fill(~allowed, -torch.inf).softmax(-1)
                    attended = (probabilities @ v.float()).to(hidden.dtype)
                gated = (gate.sigmoid() * attended.transpose(1, 2)).reshape(
                    batch, length, config.n_embd
                )
                hidden = hidden + self._linear(
                    name + ".attn.o_proj", block.attn.o_proj, gated, loop, valid
                )
                normalized = F.rms_norm(hidden, (config.n_embd,), eps=config.rms_norm_eps)
                gate, up = self._linear(
                    name + ".mlp.gate_up_proj", block.mlp.gate_up_proj, normalized, loop, valid
                ).chunk(2, -1)
                hidden = hidden + self._linear(
                    name + ".mlp.down_proj", block.mlp.down_proj, F.silu(gate) * up, loop, valid
                )
            return F.rms_norm(hidden, (config.n_embd,), eps=config.rms_norm_eps)

        high = model.embed_tokens(tokens) * config.embedding_scale
        low = model.z_L_init.expand_as(high)
        states = []
        for loop in range(loops):
            for _ in range(config.L_cycles):
                low = module(low + high, model.L_module, "L_module", loop)
            high = module(high + low, model.H_module, "H_module", loop)
            states.append(torch.cat((high, low), -1))
        return LoopOutput(self.model.lm_head(high).float(), states)
