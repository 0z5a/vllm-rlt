"""Persist the qualified fused FP8 representation without decoding its weights."""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn

from vllm_rlt.models.nanbeige import NanbeigeConfig, NanbeigeForCausalLM
from vllm_rlt.models.ouro import OuroConfig, OuroForCausalLM

from .fused_fp8 import RowwiseFP8Linear
from .quantizers import ScaleLayout
from .serving import FP8DecoderLayer, quantize_native_core


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_native_model(
    model: OuroForCausalLM | NanbeigeForCausalLM, folder: Path, *, model_revision: str
) -> None:
    layers = list(model.model.layers)
    if not layers or not all(isinstance(layer, FP8DecoderLayer) for layer in layers):
        raise ValueError("export requires a converted fused FP8 core")
    first = layers[0]
    producer = first.producers["qkv"]
    layout, mode = producer.layout, producer.mode
    if any(
        layer.fuse_norm != first.fuse_norm
        or any(p.layout != layout or p.mode != mode for p in layer.producers.values())
        for layer in layers
    ):
        raise ValueError("export requires one registered format and layout across layers")
    weights = [
        module.packed_weight for module in model.modules() if isinstance(module, RowwiseFP8Linear)
    ]
    if len(weights) != len(layers) * 7 or len({weight.data_ptr() for weight in weights}) != len(
        weights
    ):
        raise ValueError("each physical projection must own exactly one packed matrix")
    folder.mkdir(parents=True, exist_ok=False)
    payload = folder / "tensors.pt"
    torch.save({name: value.detach().cpu() for name, value in model.state_dict().items()}, payload)
    manifest = dict(
        schema_version=1,
        format="fused_rowwise_fp8_e4m3fn",
        model_revision=model_revision,
        model_config=model.config.to_dict(),
        layout=asdict(layout),
        mode=mode,
        fused_norm=first.fuse_norm,
        packed_weight_matrices=len(weights),
        packed_weight_copies=1,
        packed_weight_bytes=sum(weight.numel() for weight in weights),
        state_dtype="bfloat16",
        kv_dtype="bfloat16",
        native_quality="not_run",
        tensor_sha256=_digest(payload),
    )
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def load_native_model(
    folder: Path, device: torch.device, *, max_rows: int
) -> OuroForCausalLM | NanbeigeForCausalLM:
    manifest = json.loads((folder / "manifest.json").read_text())
    if manifest["schema_version"] != 1 or manifest["format"] != "fused_rowwise_fp8_e4m3fn":
        raise ValueError("unsupported native export format")
    if device.type != "cuda":
        raise ValueError("fused native execution requires CUDA")
    payload = folder / "tensors.pt"
    if _digest(payload) != manifest["tensor_sha256"]:
        raise ValueError("exported tensor hash mismatch")
    tensors = torch.load(payload, map_location=device, weights_only=True)
    with torch.device("meta"):
        config = manifest["model_config"]
        if config["model_type"] == "ouro":
            model = OuroForCausalLM(OuroConfig.from_dict(config))
        elif config["model_type"] == "nanbeige":
            model = NanbeigeForCausalLM(NanbeigeConfig.from_dict(config))
        else:
            raise ValueError("unsupported exported model family")
    projections = {
        name
        for name, module in model.named_modules()
        if name.startswith("model.layers.") and isinstance(module, nn.Linear)
    }
    expected = model.state_dict()
    protected = expected.keys() - {name + ".weight" for name in projections}
    for name in protected:
        value = tensors[name]
        if value.shape != expected[name].shape or value.dtype != torch.bfloat16:
            raise ValueError(f"protected BF16 parameter mismatch: {name}")
        parent, field = name.rsplit(".", 1)
        model.get_submodule(parent).register_parameter(
            field, nn.Parameter(value, requires_grad=False)
        )
    packed = {
        name: (tensors[name + ".packed_weight"], tensors[name + ".weight_scales"])
        for name in projections
    }
    scales = {}
    groups = {
        "qkv": ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"),
        "o": ("self_attn.o_proj",),
        "gate_up": ("mlp.gate_proj", "mlp.up_proj"),
        "down": ("mlp.down_proj",),
    }
    for index in range(model.config.num_hidden_layers):
        prefix = f"model.layers.{index}"
        for group, members in groups.items():
            value = tensors[prefix + ".producers." + group + ".scales"]
            scales.update({prefix + "." + member: value for member in members})
    layout = ScaleLayout(manifest["layout"]["max_loops"], tuple(manifest["layout"]["boundaries"]))
    quantize_native_core(
        model,
        scales,
        layout,
        manifest["mode"],
        max_rows=max_rows,
        fuse_norm=manifest["fused_norm"],
        packed=packed,
    )
    model.load_state_dict(tensors, strict=True, assign=True)
    model.model.rotary_emb.inv_freq = model.model.rotary_emb.frequencies(device=device)
    if any(parameter.is_meta for parameter in model.parameters()):
        raise ValueError("export omitted protected model parameters")
    return model.eval()


# Names retained for frozen Ouro scripts; new callers use the family-neutral API.
export_native_ouro = export_native_model
load_native_ouro = load_native_model
