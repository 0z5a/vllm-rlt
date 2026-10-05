"""Differentiable fixed-step LoopFormer with protected time conditioning."""

from typing import Literal

import torch
from torch.nn import functional as F

from vllm_rlt.models.loopformer import LoopFormerForCausalLM

from .base import LoopOutput, QuantizedProjections


class LoopFormerAdapter(QuantizedProjections):
    def __init__(
        self,
        model: LoopFormerForCausalLM,
        *,
        attention_backend: Literal["eager", "sdpa"] = "sdpa",
    ) -> None:
        super().__init__(attention_backend)
        self.model = model

    def reference(self) -> "LoopFormerAdapter":
        return LoopFormerAdapter(self.model, attention_backend=self.attention_backend)

    def forward(self, tokens: torch.Tensor, valid: torch.Tensor, loops: int) -> LoopOutput:
        config, gpt = self.model.config, self.model.gpt
        if loops != 8:
            raise ValueError("this checkpoint requires eight steps of size 1/8")
        if tokens.shape != valid.shape or not valid.any(dim=1).all():
            raise ValueError("each sequence must have valid tokens and an aligned mask")
        batch, length = tokens.shape
        if length > config.block_size:
            raise ValueError("sequence exceeds the learned position table")
        positions = torch.arange(length, device=tokens.device)
        hidden = gpt.transformer.wte(tokens) + gpt.transformer.wpe(positions)
        mask = (positions[None, :] <= positions[:, None])[None, None]
        mask = mask & valid[:, None, None, :]
        time = hidden.new_zeros(batch)
        delta = time.new_full((batch,), 1 / 8)
        states = []
        for loop in range(loops):
            condition = gpt.time_embedder(time) + gpt.dt_embedder(delta)
            for index, block in enumerate(gpt.transformer.h.blocks):
                name = f"gpt.transformer.h.blocks.{index}"
                ga, gm, sa, sm = block.adaLN_modulation(condition).unsqueeze(1).chunk(4, -1)
                projected = self._linear(
                    name + ".attn.c_attn",
                    block.attn.c_attn,
                    block.norm_1(hidden) * (1 + sa),
                    loop,
                    valid,
                )
                q, k, v = (
                    part.reshape(batch, length, config.n_head, config.head_dim).transpose(1, 2)
                    for part in projected.chunk(3, -1)
                )
                if self.attention_backend == "sdpa":
                    attended = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
                else:
                    scores = (q.float() @ k.float().transpose(-2, -1)) / config.head_dim**0.5
                    probabilities = scores.masked_fill(~mask, -torch.inf).softmax(-1)
                    attended = (probabilities @ v.float()).to(hidden.dtype)
                attended = attended.transpose(1, 2).reshape(batch, length, config.n_embd)
                hidden = hidden + ga * self._linear(
                    name + ".attn.c_proj", block.attn.c_proj, attended, loop, valid
                )
                intermediate = self._linear(
                    name + ".mlp.c_fc",
                    block.mlp.c_fc,
                    block.norm_2(hidden) * (1 + sm),
                    loop,
                    valid,
                )
                hidden = hidden + gm * self._linear(
                    name + ".mlp.c_proj", block.mlp.c_proj, F.gelu(intermediate), loop, valid
                )
            time = time + delta
            states.append(torch.cat((hidden, time[:, None, None].expand(batch, length, 1)), -1))
        readout = gpt.transformer.norm_f(hidden)
        return LoopOutput(F.linear(readout, gpt.transformer.wte.weight).float(), states)
