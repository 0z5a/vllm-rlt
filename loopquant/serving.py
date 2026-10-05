"""Native Ouro/Nanbeige FP8 cores with shared producers and explicit loop IDs."""

from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F

from vllm_rlt.core.kv_cache_manager import KVCacheManager, _PreparedKVBatch
from vllm_rlt.layers import apply_rotary_pos_emb
from vllm_rlt.models.nanbeige import NanbeigeDecoderLayer, NanbeigeForCausalLM
from vllm_rlt.models.ouro import OuroDecoderLayer, OuroForCausalLM

from .fused_fp8 import FP8InputQuantizer, FP8Workspace, RowwiseFP8Linear
from .quantizers import ScaleLayout


class FP8Scratch:
    """One engine's serialized recurrent stream owns these reusable row buckets."""

    def __init__(self, model: OuroForCausalLM | NanbeigeForCausalLM, max_rows: int) -> None:
        device = model.lm_head.weight.device
        padded = (max_rows + 15) // 16 * 16
        widths = {
            module.in_features
            for name, module in model.named_modules()
            if name.startswith("model.layers.") and isinstance(module, nn.Linear)
        }
        self.storage = {
            columns: FP8Workspace.allocate(padded, columns, device) for columns in widths
        }
        self.buffers = {
            (rows, columns): FP8Workspace(buffer.packed[:rows], buffer.row_scales[:rows])
            for rows in range(16, padded + 1, 16)
            for columns, buffer in self.storage.items()
        }

    def for_values(self, values: torch.Tensor) -> FP8Workspace:
        return self.buffers[(values.shape[0] + 15) // 16 * 16, values.shape[1]]

    @property
    def nbytes(self) -> int:
        return sum(
            tensor.numel() * tensor.element_size()
            for buffer in self.storage.values()
            for tensor in (buffer.packed, buffer.row_scales)
        )


class FP8DecoderLayer(nn.Module):
    def __init__(
        self,
        layer: OuroDecoderLayer | NanbeigeDecoderLayer,
        prefix: str,
        scales: dict[str, torch.Tensor],
        layout: ScaleLayout,
        mode: Literal["static", "dynamic"],
        scratch: FP8Scratch,
        fuse_norm: bool,
        packed: dict[str, tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> None:
        super().__init__()
        self.config, self.layer_idx = layer.self_attn.config, layer.self_attn.layer_idx
        self.scratch, self.fuse_norm = scratch, fuse_norm
        self.input_layernorm = layer.input_layernorm
        self.input_layernorm_2 = (
            layer.input_layernorm_2 if isinstance(layer, OuroDecoderLayer) else nn.Identity()
        )
        self.post_attention_layernorm = layer.post_attention_layernorm
        self.post_attention_layernorm_2 = (
            layer.post_attention_layernorm_2
            if isinstance(layer, OuroDecoderLayer)
            else nn.Identity()
        )
        self.self_attn = nn.ModuleDict(
            {
                name: RowwiseFP8Linear.from_packed(*packed[prefix + ".self_attn." + name])
                if packed is not None
                else RowwiseFP8Linear(module)
                for name, module in layer.self_attn.named_children()
                if name in ("q_proj", "k_proj", "v_proj", "o_proj")
            }
        )
        self.mlp = nn.ModuleDict(
            {
                name: RowwiseFP8Linear.from_packed(*packed[prefix + ".mlp." + name])
                if packed is not None
                else RowwiseFP8Linear(module)
                for name, module in layer.mlp.named_children()
                if name in ("gate_proj", "up_proj", "down_proj")
            }
        )
        groups = {
            "qkv": ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"),
            "o": ("self_attn.o_proj",),
            "gate_up": ("mlp.gate_proj", "mlp.up_proj"),
            "down": ("mlp.down_proj",),
        }
        producers = {}
        device = self.input_layernorm.weight.device
        for name, members in groups.items():
            value = torch.ones(layout.stages, device=device)
            if mode == "static":
                value = scales[prefix + "." + members[0]].to(device)
                if any(
                    not torch.equal(value, scales[prefix + "." + m].to(device)) for m in members
                ):
                    raise ValueError(
                        "a shared producer requires identical scales for all consumers"
                    )
            producers[name] = FP8InputQuantizer(
                value, layout, mode, norm_epsilon=self.config.rms_norm_eps
            )
        self.producers = nn.ModuleDict(producers)

    def forward(
        self,
        hidden: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        batch: _PreparedKVBatch,
        cache: KVCacheManager,
    ) -> torch.Tensor:
        rows, dtype = hidden.shape[0], hidden.dtype
        normalized = hidden if self.fuse_norm else self.input_layernorm(hidden)
        activation = self.producers["qkv"](
            normalized,
            self.scratch.for_values(normalized),
            norm_weight=self.input_layernorm.weight if self.fuse_norm else None,
            loop_ids=batch.loop_ids,
        )
        shape = (rows, -1, self.config.head_dim)
        q, k, v = (
            self.self_attn[name](activation, rows, dtype=dtype).view(shape)
            for name in ("q_proj", "k_proj", "v_proj")
        )
        q, k = apply_rotary_pos_emb(q, k, *position_embeddings)
        cache._write_prepared(self.layer_idx, batch, k, v)
        attended = cache._attend_prepared(self.layer_idx, batch, q).reshape(rows, -1)
        activation = self.producers["o"](
            attended, self.scratch.for_values(attended), loop_ids=batch.loop_ids
        )
        hidden = hidden + self.input_layernorm_2(
            self.self_attn["o_proj"](activation, rows, dtype=dtype)
        )
        normalized = hidden if self.fuse_norm else self.post_attention_layernorm(hidden)
        activation = self.producers["gate_up"](
            normalized,
            self.scratch.for_values(normalized),
            norm_weight=self.post_attention_layernorm.weight if self.fuse_norm else None,
            loop_ids=batch.loop_ids,
        )
        gate = self.mlp["gate_proj"](activation, rows, dtype=dtype)
        up = self.mlp["up_proj"](activation, rows, dtype=dtype)
        gated = F.silu(gate) * up
        activation = self.producers["down"](
            gated, self.scratch.for_values(gated), loop_ids=batch.loop_ids
        )
        return hidden + self.post_attention_layernorm_2(
            self.mlp["down_proj"](activation, rows, dtype=dtype)
        )


def quantize_native_core(
    model: OuroForCausalLM | NanbeigeForCausalLM,
    scales: dict[str, torch.Tensor],
    layout: ScaleLayout,
    mode: Literal["static", "dynamic"],
    *,
    max_rows: int,
    fuse_norm: bool = True,
    packed: dict[str, tuple[torch.Tensor, torch.Tensor]] | None = None,
) -> FP8Scratch:
    """Convert once before engine creation; preserve BF16 state, KV, gate and head.

    One converted model belongs to one engine. Prefill and recurrent work share
    that engine's serialized core stream; boundary work never touches scratch.
    """
    if layout.max_loops < model.config.total_ut_steps:
        raise ValueError("scale layout must cover the model's registered recurrence")
    if packed is not None:
        projections = {
            name: module
            for name, module in model.named_modules()
            if name.startswith("model.layers.") and isinstance(module, nn.Linear)
        }
        if packed.keys() != projections.keys() or any(
            packed[name][0].shape != (module.in_features, module.out_features)
            for name, module in projections.items()
        ):
            raise ValueError("packed matrices must match every physical core projection")
    scratch = FP8Scratch(model, max_rows)
    for index, layer in enumerate(model.model.layers):
        model.model.layers[index] = FP8DecoderLayer(
            layer, f"model.layers.{index}", scales, layout, mode, scratch, fuse_norm, packed
        )
    return scratch


# Preserve the interface used by already frozen Ouro experiment runners.
quantize_ouro_native = quantize_native_core
