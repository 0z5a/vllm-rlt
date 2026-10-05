"""Differentiable Parcae with fixed injection, token values and explicit initial state."""

from typing import Literal

import torch
from torch.nn import functional as F

from vllm_rlt.models.parcae import ParcaeBlock, ParcaeForCausalLM

from .base import LoopOutput, QuantizedProjections


class ParcaeAdapter(QuantizedProjections):
    def __init__(
        self, model: ParcaeForCausalLM, *, attention_backend: Literal["eager", "sdpa"] = "sdpa"
    ) -> None:
        super().__init__(attention_backend)
        self.model = model

    def reference(self) -> "ParcaeAdapter":
        return ParcaeAdapter(self.model, attention_backend=self.attention_backend)

    def initialize_state(self, tokens: torch.Tensor) -> torch.Tensor:
        state = torch.randn_like(self.model.transformer.wte(tokens))
        std = self.model.config.initializer_range
        return torch.nn.init.trunc_normal_(state, std=std, a=-3 * std, b=3 * std)

    def _block(
        self,
        hidden: torch.Tensor,
        block: ParcaeBlock,
        name: str,
        tokens: torch.Tensor,
        valid: torch.Tensor,
        mask: torch.Tensor,
        freqs: torch.Tensor,
        loop: int,
    ) -> torch.Tensor:
        config = self.model.config
        batch, length, _ = hidden.shape

        def project(suffix: str, layer: torch.nn.Linear, values: torch.Tensor) -> torch.Tensor:
            return (
                self._linear(name + suffix, layer, values, loop, valid)
                if loop >= 0
                else layer(values)
            )

        source = block.norm_1(hidden)
        shape = (batch, length, config.num_attention_heads, config.head_dim)
        q = project(".attn.c_q", block.attn.c_q, source).reshape(shape)
        k = project(".attn.c_k", block.attn.c_k, source).reshape(shape)
        v = project(".attn.c_v", block.attn.c_v, source).reshape(shape)
        key = str(block.attn.layer_idx)
        if key in self.model.value_embeds and block.attn.ve_gate is not None:
            values = self.model.value_embeds[key](tokens).reshape(shape)
            gate = 2 * torch.sigmoid(block.attn.ve_gate(source[..., :32]))
            v = v + gate.unsqueeze(-1) * values
        pairs = torch.cat((q, k), dim=2).float().unflatten(-1, (-1, 2))
        real = pairs[..., 0] * freqs[..., 0] - pairs[..., 1] * freqs[..., 1]
        imag = pairs[..., 1] * freqs[..., 0] + pairs[..., 0] * freqs[..., 1]
        q, k = torch.stack((real, imag), -1).flatten(-2).to(hidden.dtype).chunk(2, dim=2)
        q = F.rms_norm(q, (config.head_dim,)).transpose(1, 2)
        k = F.rms_norm(k, (config.head_dim,)).transpose(1, 2)
        v = v.transpose(1, 2)
        if self.attention_backend == "sdpa":
            attended = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        else:
            scores = (q @ k.transpose(-2, -1)) / config.head_dim**0.5
            probabilities = scores.masked_fill(~mask, -torch.inf).float().softmax(-1).to(v.dtype)
            attended = probabilities @ v
        attended = attended.transpose(1, 2).reshape(batch, length, config.n_embd)
        hidden = hidden + project(".attn.c_proj", block.attn.c_proj, attended)
        intermediate = project(".mlp.fc", block.mlp.fc, block.norm_2(hidden))
        return hidden + project(".mlp.proj", block.mlp.proj, F.relu(intermediate).square() * 0.5)

    def forward(
        self,
        tokens: torch.Tensor,
        valid: torch.Tensor,
        loops: int,
        initial_state: torch.Tensor | None = None,
    ) -> LoopOutput:
        config, transformer = self.model.config, self.model.transformer
        if (
            not 1 <= loops <= config.mean_recurrence
            or tokens.shape != valid.shape
            or not valid.any(dim=1).all()
        ):
            raise ValueError("expected aligned nonempty token windows and registered recurrence")
        length = tokens.shape[1]
        if length > config.block_size:
            raise ValueError("sequence exceeds the registered context")
        mask = torch.ones(length, length, device=tokens.device, dtype=torch.bool).tril()[None, None]
        mask = mask & valid[:, None, None, :]
        freqs = self.model.freqs_cis[:, :length]
        injection = transformer.wte(tokens)
        for index, block in enumerate(transformer.prelude):
            injection = self._block(
                injection, block, f"transformer.prelude.{index}", tokens, valid, mask, freqs, -1
            )
        injection = transformer.ln_prelude(injection)
        state = self.initialize_state(tokens) if initial_state is None else initial_state
        if state.shape != injection.shape or state.dtype != injection.dtype:
            raise ValueError("initial state must match the embedded sequence")
        states = []
        for loop in range(loops):
            state = transformer.adapter(state, injection)
            for index, block in enumerate(transformer.core_block):
                state = self._block(
                    state,
                    block,
                    f"transformer.core_block.{index}",
                    tokens,
                    valid,
                    mask,
                    freqs,
                    loop,
                )
            states.append(state)
        state = transformer.C(state)
        for index, block in enumerate(transformer.coda):
            state = self._block(
                state, block, f"transformer.coda.{index}", tokens, valid, mask, freqs, -1
            )
        return LoopOutput(self.model.lm_head(transformer.ln_f(state)).float(), states)
