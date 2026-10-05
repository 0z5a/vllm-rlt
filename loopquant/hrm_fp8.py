"""Shared HRM FP8 projections with explicit H-cycle scales and original H/L arithmetic."""

from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F

from vllm_rlt.core.kv_cache_manager import KVCacheManager, _PreparedKVBatch
from vllm_rlt.models.hrm_text import HrmBlock, HrmTextForCausalLM

from .fused_fp8 import FP8InputQuantizer, FP8Workspace, RowwiseFP8Linear
from .int4_export import core_projections
from .quantizers import ScaleLayout

CORE_NAMES = ("attn.gqkv_proj", "attn.o_proj", "mlp.gate_up_proj", "mlp.down_proj")


class HrmFP8Scratch:
    """One allocation per input width, reused on the owning engine's core stream."""

    def __init__(self, model: HrmTextForCausalLM, max_rows: int) -> None:
        self.device = model.lm_head.weight.device
        padded = (max_rows + 15) // 16 * 16
        widths = {layer.in_features for layer in core_projections(model).values()}
        self.storage = {
            width: FP8Workspace.allocate(padded, width, self.device) for width in widths
        }
        self.buffers = {
            (rows, width): FP8Workspace(buffer.packed[:rows], buffer.row_scales[:rows])
            for rows in range(16, padded + 1, 16)
            for width, buffer in self.storage.items()
        }

    def for_values(self, values: torch.Tensor) -> FP8Workspace:
        return self.buffers[(values.shape[0] + 15) // 16 * 16, values.shape[1]]


class FP8HrmBlock(nn.Module):
    def __init__(
        self,
        block: HrmBlock,
        prefix: str,
        scales: dict[str, torch.Tensor],
        layout: ScaleLayout,
        mode: Literal["static", "dynamic"],
        scratch: HrmFP8Scratch,
        packed: dict[str, tuple[torch.Tensor, torch.Tensor]] | None,
    ) -> None:
        super().__init__()
        self.eps, self.scratch = block.eps, scratch
        self.attn, self.mlp = block.attn, block.mlp
        device = scratch.device
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
        cosine: torch.Tensor,
        sine: torch.Tensor,
        layer: int,
        batch: _PreparedKVBatch,
        cache: KVCacheManager,
    ) -> torch.Tensor:
        rows, dtype, config = hidden.shape[0], hidden.dtype, self.attn.config
        normalized = F.rms_norm(hidden, (config.n_embd,), eps=self.eps)
        activation = self.producers["attn_gqkv_proj"](
            normalized, self.scratch.for_values(normalized), loop_ids=batch.loop_ids
        )
        gate, query, key, value = (
            self.attn.gqkv_proj(activation, rows, dtype=dtype)
            .reshape(rows, 4 * config.num_attention_heads, config.head_dim)
            .chunk(4, 1)
        )

        def rotate(values: torch.Tensor) -> torch.Tensor:
            left, right = values.float().chunk(2, -1)
            return (values.float() * cosine + torch.cat((-right, left), -1) * sine).to(dtype)

        query, key = rotate(query), rotate(key)
        cache._write_prepared(layer, batch, key, value)
        attended = (gate.sigmoid() * cache._attend_prepared(layer, batch, query)).flatten(1)
        activation = self.producers["attn_o_proj"](
            attended, self.scratch.for_values(attended), loop_ids=batch.loop_ids
        )
        hidden = hidden + self.attn.o_proj(activation, rows, dtype=dtype)
        normalized = F.rms_norm(hidden, (config.n_embd,), eps=self.eps)
        activation = self.producers["mlp_gate_up_proj"](
            normalized, self.scratch.for_values(normalized), loop_ids=batch.loop_ids
        )
        gate, up = self.mlp.gate_up_proj(activation, rows, dtype=dtype).chunk(2, -1)
        gated = F.silu(gate) * up
        activation = self.producers["mlp_down_proj"](
            gated, self.scratch.for_values(gated), loop_ids=batch.loop_ids
        )
        return hidden + self.mlp.down_proj(activation, rows, dtype=dtype)


def quantize_hrm_fp8(
    model: HrmTextForCausalLM,
    scales: dict[str, torch.Tensor],
    layout: ScaleLayout,
    mode: Literal["static", "dynamic"],
    *,
    max_rows: int,
    packed: dict[str, tuple[torch.Tensor, torch.Tensor]] | None = None,
) -> HrmFP8Scratch:
    """Convert physical H/L projections once; the native recurrence and KV planes stay intact."""
    if max_rows < 1 or layout.max_loops < model.config.H_cycles:
        raise ValueError("scratch and scale layout must cover the complete H cycles")
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
    scratch = HrmFP8Scratch(model, max_rows)
    for name in ("H_module", "L_module"):
        module = model.model.get_submodule(name)
        for index, block in enumerate(module.layers):
            module.layers[index] = FP8HrmBlock(
                block, f"model.{name}.layers.{index}", scales, layout, mode, scratch, packed
            )
    return scratch
