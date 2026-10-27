# SPDX-License-Identifier: Apache-2.0
"""AsymLoop execution on Ouro weights, for measuring its serving cost.

AsymLoop (arXiv 2610.28489) runs the first ``encoder_layers`` layers once per
token. Each decoder layer projects the encoder output once to a fixed global
KV, and the decoder layers loop: at every depth a decoder layer attends to the
global KV and its own depth's KV under one causal softmax. The first loop's
input is the encoder output; Ouro's in-loop norm, exit gate and LM head stay.

KV lives in decoder-width cache slots: the encoder's KV and the global KV in
two auxiliary planes, each loop depth's KV in its own plane. Ouro checkpoints
have no global projections, so each decoder layer's starts as a copy of its
own K/V projections and is applied after its own input norm. The outputs are
therefore not a trained AsymLoop model's; the costs are.
"""

from dataclasses import replace
from typing import TYPE_CHECKING

import torch
from torch import nn

from vllm_rlt.layers import apply_rotary_pos_emb, rotate_half
from vllm_rlt.models.ouro import OuroForCausalLM

if TYPE_CHECKING:
    from vllm_rlt.core.kv_cache_manager import KVCacheManager

# Auxiliary KV planes of the cache manager.
ENCODER, GLOBAL = 0, 1


def _residual(layer, hidden, attention):
    """The rest of an Ouro decoder layer after its attention output projection."""
    hidden = hidden + layer.input_layernorm_2(attention)
    return hidden + layer.post_attention_layernorm_2(
        layer.mlp(layer.post_attention_layernorm(hidden))
    )


class AsymLoopOuroForCausalLM(nn.Module):
    """Ouro split into a one-pass encoder and a looped decoder with global KV."""

    # The encoder runs in the prelude and writes KV, as Huginn's prelude does.
    requires_boundary_kv = True
    packed_boundary_prefill = True
    aux_kv_planes = 2
    coda_kv_layers = ()

    def __init__(self, base: OuroForCausalLM, *, encoder_layers: int, loops: int) -> None:
        super().__init__()
        layers = base.config.num_hidden_layers
        if type(encoder_layers) is not int or not 0 < encoder_layers < layers:
            raise ValueError(f"encoder_layers must be in [1, {layers})")
        if type(loops) is not int or loops <= 0:
            raise ValueError("loops must be a positive integer")
        self.base = base
        self.config = replace(base.config, total_ut_steps=loops)
        self.encoder_layers = encoder_layers
        decoder_layers = layers - encoder_layers
        self.kv_cache_layers = max(encoder_layers, decoder_layers)
        self.recurrent_kv_layers = tuple(range(decoder_layers))
        weight = base.model.embed_tokens.weight
        self.global_kv = nn.ModuleList()
        for layer in base.model.layers[encoder_layers:]:
            attention = layer.self_attn
            projection = nn.Linear(
                weight.shape[1],
                2 * attention.k_proj.out_features,
                bias=False,
                device=weight.device,
                dtype=weight.dtype,
            )
            with torch.no_grad():
                projection.weight.copy_(
                    torch.cat((attention.k_proj.weight, attention.v_proj.weight))
                )
            self.global_kv.append(projection)
        self.requires_grad_(False)
        self.eval()

    @classmethod
    def from_pretrained(cls, path_or_repo, *, encoder_layers, loops, **kwargs):
        return cls(
            OuroForCausalLM.from_pretrained(path_or_repo, **kwargs),
            encoder_layers=encoder_layers,
            loops=loops,
        )

    @property
    def _layers(self):
        return self.base.model.layers

    def _qkv(self, layer, hidden, position_embeddings):
        attention = layer.self_attn
        shape = (hidden.shape[0], -1, self.config.head_dim)
        normed = layer.input_layernorm(hidden)
        q = attention.q_proj(normed).view(shape)
        k = attention.k_proj(normed).view(shape)
        v = attention.v_proj(normed).view(shape)
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        return q, k, v

    def prelude(self, token_ids):
        raise ValueError("AsymLoop's prelude writes encoder KV; use prelude_prepared")

    def prelude_prepared(self, tokens, batch, cache: "KVCacheManager", *, generators=None):
        """Embed, run the encoder once, and write every decoder layer's global KV."""
        hidden = self.base.model.embed_tokens(tokens)
        position_embeddings = self.base.model.rotary_emb(hidden, batch.position_ids)
        count = hidden.shape[0]
        for slot, layer in enumerate(self._layers[: self.encoder_layers]):
            q, k, v = self._qkv(layer, hidden, position_embeddings)
            cache._write_aux_prepared(ENCODER, slot, batch, k, v)
            output = cache._attend_aux_prepared(ENCODER, slot, batch, q)
            hidden = _residual(layer, hidden, layer.self_attn.o_proj(output.reshape(count, -1)))
        cos, sin = position_embeddings
        for slot, (layer, projection) in enumerate(
            zip(self._layers[self.encoder_layers :], self.global_kv)
        ):
            k, v = (
                projection(layer.input_layernorm(hidden))
                .view(count, 2, -1, self.config.head_dim)
                .unbind(1)
            )
            cache._write_aux_prepared(GLOBAL, slot, batch, k * cos + rotate_half(k) * sin, v)
        return hidden

    def recurrent_prepared(self, hidden, batch, cache, *, compute_gate=True):
        """One decoder loop at each row's own depth, reading the fixed global KV."""
        position_embeddings = self.base.model.rotary_emb(hidden, batch.position_ids)
        count = hidden.shape[0]
        for slot, layer in enumerate(self._layers[self.encoder_layers :]):
            q, k, v = self._qkv(layer, hidden, position_embeddings)
            cache._write_prepared(slot, batch, k, v)
            output = cache._attend_with_aux_prepared(GLOBAL, slot, batch, q)
            hidden = _residual(layer, hidden, layer.self_attn.o_proj(output.reshape(count, -1)))
        hidden = self.base.model.norm(hidden)
        gate = self.base.model.early_exit_gate(hidden).squeeze(-1) if compute_gate else None
        return hidden, gate

    def recurrent(self, hidden, request_ids, depths, positions, cache, *, compute_gate=True):
        batch = cache._prepare_batch(request_ids, depths, positions)
        return self.recurrent_prepared(hidden, batch, cache, compute_gate=compute_gate)

    def coda(self, hidden):
        return self.base.lm_head(hidden)

    def coda_prepared(self, hidden, batch, cache, *, compute_logits=True):
        return self.base.lm_head(hidden) if compute_logits else None
