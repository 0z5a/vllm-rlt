"""Native Parcae FP8 artifacts preserve encoded weights and tied protected readout."""

import json
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn

from vllm_rlt.models.parcae import ParcaeConfig, ParcaeForCausalLM

from .fused_fp8 import RowwiseFP8Linear
from .int4_export import core_projections
from .native_export import _digest
from .parcae_fp8 import PROJECTION_GROUPS, FP8ParcaeBlock, quantize_parcae_fp8
from .quantizers import ScaleLayout


def export_parcae_fp8(model: ParcaeForCausalLM, folder: Path, *, model_revision: str) -> None:
    blocks = list(model.transformer.core_block)
    if not blocks or not all(isinstance(block, FP8ParcaeBlock) for block in blocks):
        raise ValueError("export requires every Parcae core block converted")
    first = blocks[0]
    producer = first.producers["qkv"]
    if any(
        block.fuse_norm != first.fuse_norm
        or any(
            p.layout != producer.layout or p.mode != producer.mode for p in block.producers.values()
        )
        for block in blocks
    ):
        raise ValueError("one registered scale layout and mode must cover all core blocks")
    if model.lm_head.weight is not model.transformer.wte.weight:
        raise ValueError("Parcae requires a tied embedding/head")
    weights = [
        layer.packed_weight for layer in model.modules() if isinstance(layer, RowwiseFP8Linear)
    ]
    if len(weights) != len(blocks) * 6 or len({w.data_ptr() for w in weights}) != len(weights):
        raise ValueError("each physical core projection must own one packed weight")
    folder.mkdir(parents=True, exist_ok=False)
    payload = folder / "tensors.pt"
    torch.save(
        {
            name: value.detach().cpu()
            for name, value in model.state_dict().items()
            if name != "lm_head.weight"
        },
        payload,
    )
    manifest = dict(
        schema_version=1,
        format="parcae_fused_rowwise_fp8_e4m3fn",
        model_revision=model_revision,
        model_config=model.config.to_dict(),
        layout=asdict(producer.layout),
        mode=producer.mode,
        fused_norm=first.fuse_norm,
        packed_weight_matrices=len(weights),
        packed_weight_copies=1,
        packed_weight_bytes=sum(w.numel() for w in weights),
        protected_aliases={"lm_head.weight": "transformer.wte.weight"},
        state_dtype="bfloat16",
        kv_dtype="bfloat16",
        native_quality="not_run",
        tensor_sha256=_digest(payload),
    )
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def load_parcae_fp8(folder: Path, device: torch.device, *, max_rows: int) -> ParcaeForCausalLM:
    manifest = json.loads((folder / "manifest.json").read_text())
    if manifest["schema_version"] != 1 or manifest["format"] != "parcae_fused_rowwise_fp8_e4m3fn":
        raise ValueError("unsupported native Parcae FP8 artifact")
    if device.type != "cuda":
        raise ValueError("native FP8 execution requires CUDA")
    payload = folder / "tensors.pt"
    if _digest(payload) != manifest["tensor_sha256"]:
        raise ValueError("exported tensor hash mismatch")
    if manifest["protected_aliases"] != {"lm_head.weight": "transformer.wte.weight"}:
        raise ValueError("artifact changed the tied Parcae readout")
    tensors = torch.load(payload, map_location="cpu", weights_only=True)
    with torch.device("meta"):
        model = ParcaeForCausalLM(ParcaeConfig.from_dict(manifest["model_config"]))
    projections = core_projections(model)
    expected = model.state_dict()
    protected = expected.keys() - {name + ".weight" for name in projections} - {"lm_head.weight"}
    scale_names = {
        f"transformer.core_block.{index}.{member}": (
            f"transformer.core_block.{index}.producers.{group}.scales"
        )
        for index in range(model.config.n_layers_in_recurrent_block)
        for group, members in PROJECTION_GROUPS.items()
        for member in members
    }
    all_keys = (
        protected
        | set(scale_names.values())
        | {name + suffix for name in projections for suffix in (".packed_weight", ".weight_scales")}
    )
    if tensors.keys() != all_keys:
        raise ValueError("artifact changed the core or protected tensor set")
    for name in protected:
        value = tensors[name]
        dtype = torch.float32 if name == "freqs_cis" else torch.bfloat16
        if value.dtype != dtype or value.shape != expected[name].shape:
            raise ValueError(f"protected tensor shape or dtype mismatch: {name}")
        if name == "freqs_cis":
            model.register_buffer(name, value.to(device), persistent=True)
        else:
            parent, field = name.rsplit(".", 1)
            model.get_submodule(parent).register_parameter(
                field, nn.Parameter(value.to(device), requires_grad=False)
            )
    model.lm_head.weight = model.transformer.wte.weight
    scales = {name: tensors[key].to(device) for name, key in scale_names.items()}
    packed = {
        name: (
            tensors[name + ".packed_weight"].to(device),
            tensors[name + ".weight_scales"].to(device),
        )
        for name in projections
    }
    layout = ScaleLayout(manifest["layout"]["max_loops"], tuple(manifest["layout"]["boundaries"]))
    quantize_parcae_fp8(
        model,
        scales,
        layout,
        manifest["mode"],
        max_rows=max_rows,
        fuse_norm=manifest["fused_norm"],
        packed=packed,
    )
    return model.eval()
