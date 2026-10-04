"""SH1 FP8 checkpoint export: one encoded core matrix and no training auxiliaries."""

import hashlib
import json
from pathlib import Path

import torch
from torch import nn

from vllm_rlt.models.ouro import OuroConfig, OuroForCausalLM

from .adapters.ouro import OuroAdapter
from .backends import NativeFP8Linear


def export_ouro(adapter: OuroAdapter, folder: Path, *, model_revision: str) -> None:
    names = {
        name
        for name, module in adapter.model.named_modules()
        if name.startswith("model.layers.") and isinstance(module, nn.Linear)
    }
    if names != {key.replace("__", ".") for key in adapter.quantized}:
        raise ValueError("export requires every registered core projection")
    weights, weight_scales, activation_scales = {}, {}, {}
    for name in sorted(names):
        module = adapter.quantized[name.replace(".", "__")]
        if module.layout.stages != 1:
            raise ValueError("native staged row scales are not yet qualified")
        weights[name] = module.packed_weight.detach().cpu().t()
        weight_scales[name] = module.weight_scale.detach().cpu()
        activation_scales[name] = module.log_scale.detach().exp().cpu()
    protected = {
        name: value.detach().to(device="cpu", dtype=torch.bfloat16)
        for name, value in adapter.model.state_dict().items()
        if name not in {module + ".weight" for module in names}
    }
    folder.mkdir(parents=True, exist_ok=False)
    tensors = folder / "tensors.pt"
    torch.save(
        {
            "packed": weights,
            "weight_scales": weight_scales,
            "activation_scales": activation_scales,
            "protected": protected,
        },
        tensors,
    )
    digest = hashlib.sha256()
    with tensors.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    manifest = {
        "schema_version": 1,
        "format": "fp8_e4m3fn",
        "scale_policy": "SH1",
        "model_revision": model_revision,
        "model_config": adapter.model.config.to_dict(),
        "packed_weight_matrices": len(weights),
        "packed_weight_copies": 1,
        "packed_weight_bytes": sum(weight.numel() for weight in weights.values()),
        "scale_bytes": 8 * len(weights),
        "protected_bytes": sum(
            value.numel() * value.element_size() for value in protected.values()
        ),
        "tensor_sha256": digest.hexdigest(),
        "native_quality": "not_run",
        "state_dtype": "bfloat16",
        "kv_dtype": "bfloat16",
        "online_auxiliary_operators": [],
    }
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def load_exported_ouro(folder: Path, device: torch.device) -> OuroForCausalLM:
    manifest = json.loads((folder / "manifest.json").read_text())
    if manifest["format"] != "fp8_e4m3fn" or manifest["scale_policy"] != "SH1":
        raise ValueError("unsupported deployment format or scale policy")
    digest = hashlib.sha256()
    with (folder / "tensors.pt").open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    if digest.hexdigest() != manifest["tensor_sha256"]:
        raise ValueError("exported tensor hash mismatch")
    tensors = torch.load(folder / "tensors.pt", weights_only=True, map_location="cpu")
    with torch.device("meta"):
        model = OuroForCausalLM(OuroConfig.from_dict(manifest["model_config"]))
    for name, packed in tensors["packed"].items():
        parent, field = name.rsplit(".", 1)
        model.get_submodule(parent).add_module(
            field,
            NativeFP8Linear.from_packed(
                packed.to(device),
                tensors["weight_scales"][name].to(device),
                tensors["activation_scales"][name].to(device),
            ),
        )
    for name, value in tensors["protected"].items():
        parent, field = name.rsplit(".", 1)
        model.get_submodule(parent).register_parameter(
            field, nn.Parameter(value.to(device=device, dtype=torch.bfloat16), requires_grad=False)
        )
    if any(parameter.is_meta for parameter in model.parameters()):
        raise ValueError("export omitted model parameters")
    model.model.rotary_emb.inv_freq = model.model.rotary_emb.frequencies(device=device)
    return model.eval()
