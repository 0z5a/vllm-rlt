"""Differentiable Huginn recurrence with explicit initial state and fixed injection."""

import math
from typing import Literal

import torch
from torch.nn import functional as F

from vllm_rlt.models.huginn import HuginnBlock, HuginnForCausalLM

from .base import LoopOutput, QuantizedProjections


class HuginnAdapter(QuantizedProjections):
    def __init__(
        self,
        model: HuginnForCausalLM,
        *,
        attention_backend: Literal["eager", "sdpa"] = "sdpa",
    ) -> None:
        super().__init__(attention_backend)
        self.model = model

    def reference(self) -> "HuginnAdapter":
        return HuginnAdapter(self.model, attention_backend=self.attention_backend)

    def initialize_state(self, tokens: torch.Tensor) -> torch.Tensor:
        embedding = self.model.transformer.wte(tokens)
        state = torch.randn_like(embedding)
        std = self.model.config.initializer_range
        torch.nn.init.trunc_normal_(state, std=std, a=-3 * std, b=3 * std)
        return state * math.sqrt(self.model.config.n_embd)

    def _block(
        self,
        hidden: torch.Tensor,
        block: HuginnBlock,
        name: str,
        loop: int,
        valid: torch.Tensor,
        mask: torch.Tensor,
        freqs: torch.Tensor,
    ) -> torch.Tensor:
        config = self.model.config
        batch, length, _ = hidden.shape

        def project(suffix: str, module: torch.nn.Linear, values: torch.Tensor) -> torch.Tensor:
            if loop >= 0:
                return self._linear(name + suffix, module, values, loop, valid)
            return module(values)

        q, k, v = project(".attn.Wqkv", block.attn.Wqkv, block.norm_1(hidden)).chunk(3, -1)
        shape = (batch, length, config.n_heads, config.head_dim)
        q, k, v = q.reshape(shape), k.reshape(shape), v.reshape(shape)
        q, k = q + block.attn.qk_bias[0], k + block.attn.qk_bias[1]
        pairs = torch.stack((q, k)).float().reshape(2, batch, length, config.n_heads, -1, 2)
        cos, sin = freqs[..., 0], freqs[..., 1]
        real = pairs[..., 0] * cos - pairs[..., 1] * sin
        imag = pairs[..., 1] * cos + pairs[..., 0] * sin
        q, k = torch.stack((real, imag), -1).flatten(-2).to(hidden.dtype).unbind(0)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        if self.attention_backend == "sdpa":
            attended = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        else:
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(config.head_dim)
            scores = scores.masked_fill(~mask, -torch.inf)
            attended = scores.softmax(-1, dtype=torch.float32).to(q.dtype) @ v
        attended = attended.transpose(1, 2).reshape(batch, length, config.n_embd)
        hidden = block.norm_2(project(".attn.proj", block.attn.proj, attended) + hidden)
        gate, up = project(".mlp.fc", block.mlp.fc, block.norm_3(hidden)).chunk(2, -1)
        return block.norm_4(project(".mlp.proj", block.mlp.proj, F.silu(gate) * up) + hidden)

    def forward(
        self,
        tokens: torch.Tensor,
        valid: torch.Tensor,
        loops: int,
        initial_state: torch.Tensor | None = None,
    ) -> LoopOutput:
        config = self.model.config
        if not 1 <= loops <= config.mean_recurrence:
            raise ValueError("loop count outside the registered model range")
        if tokens.shape != valid.shape or not valid.any(dim=1).all():
            raise ValueError("each sequence must have valid tokens and an aligned mask")
        length = tokens.shape[1]
        mask = torch.ones(length, length, device=tokens.device, dtype=torch.bool).tril()
        mask = mask[None, None] & valid[:, None, None, :]
        freqs = self.model.freqs_cis[:, :length]
        injection = self.model.transformer.wte(tokens) * math.sqrt(config.n_embd)
        for index, block in enumerate(self.model.transformer.prelude):
            injection = self._block(
                injection, block, f"transformer.prelude.{index}", -1, valid, mask, freqs
            )
        state = self.initialize_state(tokens) if initial_state is None else initial_state
        if state.shape != injection.shape or state.dtype != injection.dtype:
            raise ValueError("initial state must match the embedded sequence shape and dtype")
        states = []
        for loop in range(loops):
            state = self._linear(
                "transformer.adapter",
                self.model.transformer.adapter,
                torch.cat((state, injection), -1),
                loop,
                valid,
            )
            for index, block in enumerate(self.model.transformer.core_block):
                state = self._block(
                    state, block, f"transformer.core_block.{index}", loop, valid, mask, freqs
                )
            states.append(state)
        readout = self.model.transformer.ln_f(state)
        for index, block in enumerate(self.model.transformer.coda):
            readout = self._block(
                readout, block, f"transformer.coda.{index}", -1, valid, mask, freqs
            )
        return LoopOutput(self.model.lm_head(self.model.transformer.ln_f(readout)).float(), states)
