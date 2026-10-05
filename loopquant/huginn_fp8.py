"""Huginn FP8 core with explicit row depths and protected boundary arithmetic."""

from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F

from vllm_rlt.core.kv_cache_manager import KVCacheManager, _PreparedKVBatch
from vllm_rlt.models.huginn import HuginnBlock, HuginnForCausalLM

from .fused_fp8 import FP8InputQuantizer, FP8Workspace, RowwiseFP8Linear
from .huginn_int4 import core_projections
from .quantizers import ScaleLayout


class HuginnFP8Scratch:
    """Row buckets alias one allocation per width on one engine's core stream."""

    def __init__(self, model: HuginnForCausalLM, max_rows: int) -> None:
        padded = (max_rows + 15) // 16 * 16
        widths = {layer.in_features for layer in core_projections(model).values()}
        self.storage = {
            width: FP8Workspace.allocate(padded, width, model.transformer.wte.weight.device)
            for width in widths
        }
        self.buffers = {
            (rows, width): FP8Workspace(buffer.packed[:rows], buffer.row_scales[:rows])
            for rows in range(16, padded + 1, 16)
            for width, buffer in self.storage.items()
        }

    def for_values(self, values: torch.Tensor) -> FP8Workspace:
        return self.buffers[(values.shape[0] + 15) // 16 * 16, values.shape[1]]


class FP8HuginnBlock(nn.Module):
    def __init__(
        self,
        block: HuginnBlock,
        prefix: str,
        scales: dict[str, torch.Tensor],
        layout: ScaleLayout,
        mode: Literal["static", "dynamic"],
        scratch: HuginnFP8Scratch,
        fuse_norm: bool,
        packed: dict[str, tuple[torch.Tensor, torch.Tensor]] | None,
    ) -> None:
        super().__init__()
        self.scratch, self.fuse_norm = scratch, fuse_norm
        self.norm_1, self.norm_2 = block.norm_1, block.norm_2
        self.norm_3, self.norm_4 = block.norm_3, block.norm_4
        self.attn, self.mlp = block.attn, block.mlp
        producers = {}
        for name in ("attn.Wqkv", "attn.proj", "mlp.fc", "mlp.proj"):
            layer = block.get_submodule(name)
            parent, field = name.rsplit(".", 1)
            block.get_submodule(parent).add_module(
                field,
                RowwiseFP8Linear(layer)
                if packed is None
                else RowwiseFP8Linear.from_packed(*packed[prefix + "." + name]),
            )
            values = (
                scales[prefix + "." + name].to(self.norm_1.weight.device)
                if mode == "static"
                else torch.ones(layout.stages, device=self.norm_1.weight.device)
            )
            producers[name.replace(".", "_")] = FP8InputQuantizer(
                values, layout, mode, norm_epsilon=self.norm_1.variance_epsilon
            )
        self.producers = nn.ModuleDict(producers)

    def forward(
        self,
        hidden: torch.Tensor,
        freqs: torch.Tensor,
        batch: _PreparedKVBatch,
        cache: KVCacheManager,
    ) -> torch.Tensor:
        rows, dtype = hidden.shape[0], hidden.dtype
        normalized = hidden if self.fuse_norm else self.norm_1(hidden)
        activation = self.producers["attn_Wqkv"](
            normalized,
            self.scratch.for_values(normalized),
            norm_weight=self.norm_1.weight if self.fuse_norm else None,
            loop_ids=batch.loop_ids,
        )
        q, k, v = self.attn.Wqkv(activation, rows, dtype=dtype).chunk(3, -1)
        shape = (rows, self.attn.n_heads, self.attn.head_dim)
        q = q.reshape(shape) + self.attn.qk_bias[0]
        k = k.reshape(shape) + self.attn.qk_bias[1]
        v = v.reshape(shape)
        pair = torch.stack((q, k)).float().reshape(2, rows, self.attn.n_heads, -1, 2)
        cos, sin = freqs[..., 0].unsqueeze(0), freqs[..., 1].unsqueeze(0)
        real = pair[..., 0] * cos - pair[..., 1] * sin
        imag = pair[..., 1] * cos + pair[..., 0] * sin
        q, k = torch.stack((real, imag), -1).flatten(-2).to(dtype).unbind(0)
        cache._write_prepared(self.attn.layer_idx, batch, k, v)
        attended = cache._attend_prepared(self.attn.layer_idx, batch, q).reshape(rows, -1)
        activation = self.producers["attn_proj"](
            attended, self.scratch.for_values(attended), loop_ids=batch.loop_ids
        )
        hidden = self.norm_2(self.attn.proj(activation, rows, dtype=dtype) + hidden)
        normalized = hidden if self.fuse_norm else self.norm_3(hidden)
        activation = self.producers["mlp_fc"](
            normalized,
            self.scratch.for_values(normalized),
            norm_weight=self.norm_3.weight if self.fuse_norm else None,
            loop_ids=batch.loop_ids,
        )
        gate, up = self.mlp.fc(activation, rows, dtype=dtype).chunk(2, -1)
        gated = F.silu(gate) * up
        activation = self.producers["mlp_proj"](
            gated, self.scratch.for_values(gated), loop_ids=batch.loop_ids
        )
        return self.norm_4(self.mlp.proj(activation, rows, dtype=dtype) + hidden)


def quantize_huginn_fp8(
    model: HuginnForCausalLM,
    scales: dict[str, torch.Tensor],
    layout: ScaleLayout,
    mode: Literal["static", "dynamic"],
    *,
    max_rows: int,
    fuse_norm: bool = True,
    packed: dict[str, tuple[torch.Tensor, torch.Tensor]] | None = None,
) -> HuginnFP8Scratch:
    """Convert only recurrent blocks before the owning engine is constructed."""
    if max_rows < 1 or layout.max_loops < model.config.mean_recurrence:
        raise ValueError("scratch and scale layout must cover the registered execution")
    projections = core_projections(model)
    if mode == "static" and scales.keys() != projections.keys():
        raise ValueError("static scales must cover exactly the core projections")
    if packed is not None and (
        packed.keys() != projections.keys()
        or any(
            packed[name][0].shape != (layer.in_features, layer.out_features)
            for name, layer in projections.items()
        )
    ):
        raise ValueError("packed matrices must match every physical core projection")
    scratch = HuginnFP8Scratch(model, max_rows)
    for index, block in enumerate(model.transformer.core_block):
        model.transformer.core_block[index] = FP8HuginnBlock(
            block,
            f"transformer.core_block.{index}",
            scales,
            layout,
            mode,
            scratch,
            fuse_norm,
            packed,
        )
    return scratch
