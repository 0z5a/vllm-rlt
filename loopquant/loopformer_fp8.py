"""Native LoopFormer FP8 projections with protected conditioning and clock arithmetic."""

from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F

from vllm_rlt.core.kv_cache_manager import KVCacheManager, _PreparedKVBatch
from vllm_rlt.models.loopformer import LoopFormerBlock, LoopFormerForCausalLM

from .fused_fp8 import FP8InputQuantizer, FP8Workspace, RowwiseFP8Linear
from .int4_export import core_projections
from .quantizers import ScaleLayout

CORE_NAMES = ("attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj")


class LoopFormerFP8Scratch:
    """Row buckets share one allocation per width on one engine's core stream."""

    def __init__(self, model: LoopFormerForCausalLM, max_rows: int) -> None:
        padded = (max_rows + 15) // 16 * 16
        widths = {layer.in_features for layer in core_projections(model).values()}
        self.storage = {
            width: FP8Workspace.allocate(padded, width, model.gpt.transformer.wte.weight.device)
            for width in widths
        }
        self.buffers = {
            (rows, width): FP8Workspace(buffer.packed[:rows], buffer.row_scales[:rows])
            for rows in range(16, padded + 1, 16)
            for width, buffer in self.storage.items()
        }

    def for_values(self, values: torch.Tensor) -> FP8Workspace:
        return self.buffers[(values.shape[0] + 15) // 16 * 16, values.shape[1]]


class FP8LoopFormerBlock(nn.Module):
    def __init__(
        self,
        block: LoopFormerBlock,
        prefix: str,
        scales: dict[str, torch.Tensor],
        layout: ScaleLayout,
        mode: Literal["static", "dynamic"],
        scratch: LoopFormerFP8Scratch,
        packed: dict[str, tuple[torch.Tensor, torch.Tensor]] | None,
    ) -> None:
        super().__init__()
        self.config, self.scratch = block.config, scratch
        self.norm_1, self.norm_2 = block.norm_1, block.norm_2
        self.adaLN_modulation = block.adaLN_modulation
        self.attn, self.mlp = block.attn, block.mlp
        device = block.adaLN_modulation[1].weight.device
        producers = {}
        for name in CORE_NAMES:
            projection = block.get_submodule(name)
            parent, field = name.rsplit(".", 1)
            block.get_submodule(parent).add_module(
                field,
                RowwiseFP8Linear(projection)
                if packed is None
                else RowwiseFP8Linear.from_packed(*packed[prefix + "." + name]),
            )
            values = (
                scales[prefix + "." + name].to(device)
                if mode == "static"
                else torch.ones(layout.stages, device=device)
            )
            producers[name.replace(".", "_")] = FP8InputQuantizer(values, layout, mode)
        self.producers = nn.ModuleDict(producers)

    def forward(
        self,
        hidden: torch.Tensor,
        condition: torch.Tensor,
        layer: int,
        batch: _PreparedKVBatch,
        cache: KVCacheManager,
    ) -> torch.Tensor:
        rows, dtype = hidden.shape[0], hidden.dtype
        gate_attn, gate_mlp, scale_attn, scale_mlp = self.adaLN_modulation(condition).chunk(4, -1)
        normalized = self.norm_1(hidden) * (1 + scale_attn)
        activation = self.producers["attn_c_attn"](
            normalized, self.scratch.for_values(normalized), loop_ids=batch.loop_ids
        )
        projected = self.attn.c_attn(activation, rows, dtype=dtype)
        query, key, value = (
            part.reshape(rows, self.config.n_head, self.config.head_dim)
            for part in projected.chunk(3, -1)
        )
        cache._write_prepared(layer, batch, key, value)
        attended = cache._attend_prepared(layer, batch, query).flatten(1)
        activation = self.producers["attn_c_proj"](
            attended, self.scratch.for_values(attended), loop_ids=batch.loop_ids
        )
        hidden = hidden + gate_attn * self.attn.c_proj(activation, rows, dtype=dtype)
        normalized = self.norm_2(hidden) * (1 + scale_mlp)
        activation = self.producers["mlp_c_fc"](
            normalized, self.scratch.for_values(normalized), loop_ids=batch.loop_ids
        )
        intermediate = F.gelu(self.mlp.c_fc(activation, rows, dtype=dtype))
        activation = self.producers["mlp_c_proj"](
            intermediate, self.scratch.for_values(intermediate), loop_ids=batch.loop_ids
        )
        return hidden + gate_mlp * self.mlp.c_proj(activation, rows, dtype=dtype)


def quantize_loopformer_fp8(
    model: LoopFormerForCausalLM,
    scales: dict[str, torch.Tensor],
    layout: ScaleLayout,
    mode: Literal["static", "dynamic"],
    *,
    max_rows: int,
    packed: dict[str, tuple[torch.Tensor, torch.Tensor]] | None = None,
) -> LoopFormerFP8Scratch:
    """Convert core projections once; conditioning and normalization remain BF16."""
    if max_rows < 1 or layout.max_loops < model.config.total_ut_steps:
        raise ValueError("scratch and scale layout must cover all eight steps")
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
    scratch = LoopFormerFP8Scratch(model, max_rows)
    for index, block in enumerate(model.gpt.transformer.h.blocks):
        model.gpt.transformer.h.blocks[index] = FP8LoopFormerBlock(
            block, f"gpt.transformer.h.blocks.{index}", scales, layout, mode, scratch, packed
        )
    return scratch
