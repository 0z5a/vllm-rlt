"""Parcae FP8 core preserving token gates, diagonal injection and boundary arithmetic."""

from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F

from vllm_rlt.core.kv_cache_manager import KVCacheManager, _PreparedKVBatch
from vllm_rlt.models.parcae import ParcaeBlock, ParcaeForCausalLM

from .fused_fp8 import FP8InputQuantizer, FP8Workspace, RowwiseFP8Linear
from .int4_export import core_projections
from .quantizers import ScaleLayout

PROJECTION_GROUPS = {
    "qkv": ("attn.c_q", "attn.c_k", "attn.c_v"),
    "attn_proj": ("attn.c_proj",),
    "mlp_fc": ("mlp.fc",),
    "mlp_proj": ("mlp.proj",),
}


class ParcaeFP8Scratch:
    """One allocation per width, aliased by row buckets on the engine's core stream."""

    def __init__(self, model: ParcaeForCausalLM, max_rows: int) -> None:
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


class FP8ParcaeBlock(nn.Module):
    def __init__(
        self,
        block: ParcaeBlock,
        prefix: str,
        scales: dict[str, torch.Tensor],
        layout: ScaleLayout,
        mode: Literal["static", "dynamic"],
        scratch: ParcaeFP8Scratch,
        fuse_norm: bool,
        packed: dict[str, tuple[torch.Tensor, torch.Tensor]] | None,
    ) -> None:
        super().__init__()
        self.scratch, self.fuse_norm = scratch, fuse_norm
        self.norm_1, self.norm_2 = block.norm_1, block.norm_2
        self.attn, self.mlp = block.attn, block.mlp
        device = self.norm_1.weight.device
        producers = {}
        for group, members in PROJECTION_GROUPS.items():
            values = torch.ones(layout.stages, device=device)
            if mode == "static":
                values = scales[prefix + "." + members[0]].to(device)
                if any(
                    not torch.equal(values, scales[prefix + "." + m].to(device)) for m in members
                ):
                    raise ValueError("shared Q/K/V producer requires identical input scales")
            producers[group] = FP8InputQuantizer(
                values, layout, mode, norm_epsilon=self.norm_1.variance_epsilon
            )
            for name in members:
                layer = block.get_submodule(name)
                parent, field = name.rsplit(".", 1)
                block.get_submodule(parent).add_module(
                    field,
                    RowwiseFP8Linear(layer)
                    if packed is None
                    else RowwiseFP8Linear.from_packed(*packed[prefix + "." + name]),
                )
        self.producers = nn.ModuleDict(producers)

    def forward(
        self,
        hidden: torch.Tensor,
        values: torch.Tensor | None,
        freqs: torch.Tensor,
        batch: _PreparedKVBatch,
        cache: KVCacheManager,
    ) -> torch.Tensor:
        rows, dtype = hidden.shape[0], hidden.dtype
        normalized = hidden if self.fuse_norm else self.norm_1(hidden)
        activation = self.producers["qkv"](
            normalized,
            self.scratch.for_values(normalized),
            norm_weight=self.norm_1.weight if self.fuse_norm else None,
            loop_ids=batch.loop_ids,
        )
        shape = (rows, self.attn.heads, self.attn.head_dim)
        q, k, v = (
            projection(activation, rows, dtype=dtype).view(shape)
            for projection in (self.attn.c_q, self.attn.c_k, self.attn.c_v)
        )
        if values is not None and self.attn.ve_gate is not None:
            gate_input = self.norm_1(hidden) if self.fuse_norm else normalized
            gate = 2 * torch.sigmoid(self.attn.ve_gate(gate_input[:, :32]))
            v = v + gate.unsqueeze(-1) * values.view(shape)
        pairs = torch.cat((q, k), dim=1).float().unflatten(-1, (-1, 2))
        real = pairs[..., 0] * freqs[..., 0] - pairs[..., 1] * freqs[..., 1]
        imag = pairs[..., 1] * freqs[..., 0] + pairs[..., 0] * freqs[..., 1]
        q, k = torch.stack((real, imag), dim=-1).flatten(-2).to(dtype).chunk(2, dim=1)
        q, k = F.rms_norm(q, (self.attn.head_dim,)), F.rms_norm(k, (self.attn.head_dim,))
        cache._write_prepared(self.attn.layer_idx, batch, k, v)
        attended = cache._attend_prepared(self.attn.layer_idx, batch, q).reshape(rows, -1)
        activation = self.producers["attn_proj"](
            attended, self.scratch.for_values(attended), loop_ids=batch.loop_ids
        )
        hidden = hidden + self.attn.c_proj(activation, rows, dtype=dtype)
        normalized = hidden if self.fuse_norm else self.norm_2(hidden)
        activation = self.producers["mlp_fc"](
            normalized,
            self.scratch.for_values(normalized),
            norm_weight=self.norm_2.weight if self.fuse_norm else None,
            loop_ids=batch.loop_ids,
        )
        intermediate = F.relu(self.mlp.fc(activation, rows, dtype=dtype)).square() * 0.5
        activation = self.producers["mlp_proj"](
            intermediate, self.scratch.for_values(intermediate), loop_ids=batch.loop_ids
        )
        return hidden + self.mlp.proj(activation, rows, dtype=dtype)


def quantize_parcae_fp8(
    model: ParcaeForCausalLM,
    scales: dict[str, torch.Tensor],
    layout: ScaleLayout,
    mode: Literal["static", "dynamic"],
    *,
    max_rows: int,
    fuse_norm: bool = True,
    packed: dict[str, tuple[torch.Tensor, torch.Tensor]] | None = None,
) -> ParcaeFP8Scratch:
    """Convert only recurrent projections before constructing their owning engine."""
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
    scratch = ParcaeFP8Scratch(model, max_rows)
    for index, block in enumerate(model.transformer.core_block):
        model.transformer.core_block[index] = FP8ParcaeBlock(
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
