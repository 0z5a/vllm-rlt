"""One native INT4 matrix per Huginn core projection, with protected boundaries."""

import hashlib
import json
from pathlib import Path

import torch
from torch import nn

from vllm_rlt.models.huginn import HuginnConfig, HuginnForCausalLM

from .int4_backend import NativeInt4Linear
from .quantizers import int4_pack


def core_projections(model: HuginnForCausalLM) -> dict[str, nn.Linear]:
    """Keep injection, prelude/coda, norms, Q/K bias and tied readout protected."""
    names = [
        f"transformer.core_block.{index}.{suffix}"
        for index in range(model.config.n_layers_in_recurrent_block)
        for suffix in ("attn.Wqkv", "attn.proj", "mlp.fc", "mlp.proj")
    ]
    result = {}
    for name in names:
        layer = model.get_submodule(name)
        if not isinstance(layer, nn.Linear) or layer.bias is not None:
            raise ValueError(f"expected an unconverted bias-free core projection: {name}")
        result[name] = layer
    return result


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_huginn_int4(
    model: HuginnForCausalLM,
    folder: Path,
    *,
    model_revision: str,
    group_size: int = 128,
    method: str = "RTN",
    packed: dict[str, tuple[torch.Tensor, torch.Tensor]] | None = None,
) -> None:
    if group_size not in (32, 64, 128, 256):
        raise ValueError("unsupported native INT4 group size")
    projections = core_projections(model)
    if method not in ("RTN", "GPTQ-first", "GPTQ-all", "GPTQ-matched"):
        raise ValueError("unregistered INT4 baseline")
    if (packed is None) != (method == "RTN"):
        raise ValueError("GPTQ export requires calibrated packed matrices")
    if packed is not None and packed.keys() != projections.keys():
        raise ValueError("calibrated weights must cover all core projections")
    if model.lm_head.weight is not model.transformer.wte.weight:
        raise ValueError("Huginn requires one tied embedding/head parameter")
    codes, scales = {}, {}
    for name, layer in projections.items():
        weight, scale = (
            int4_pack(layer.weight.detach(), group_size) if packed is None else packed[name]
        )
        groups = (layer.in_features + group_size - 1) // group_size
        if weight.dtype != torch.uint8 or weight.shape != (
            layer.out_features,
            groups * group_size // 2,
        ):
            raise ValueError(f"packed matrix shape or dtype mismatch: {name}")
        scale = scale.to(device="cpu", dtype=torch.bfloat16)
        if scale.shape != (layer.out_features, groups) or not bool(
            torch.isfinite(scale).all() & (scale > 0).all()
        ):
            raise ValueError(f"invalid native BF16 group scales: {name}")
        codes[name], scales[name] = weight.cpu(), scale
    omitted = {name + ".weight" for name in projections} | {"lm_head.weight"}
    protected = {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
        if name not in omitted
    }
    for name, value in protected.items():
        expected = torch.float32 if name == "freqs_cis" else torch.bfloat16
        if value.dtype != expected:
            raise ValueError(f"protected tensor precision mismatch: {name}")
    folder.mkdir(parents=True, exist_ok=False)
    payload = folder / "tensors.pt"
    torch.save(dict(codes=codes, scales=scales, protected=protected), payload)
    manifest = dict(
        schema_version=1,
        format="huginn_signed_int4_low_nibble_first",
        method=method,
        group_size=group_size,
        model_revision=model_revision,
        model_config=model.config.to_dict(),
        packed_weight_matrices=len(codes),
        packed_weight_copies=1,
        packed_weight_bytes=sum(value.numel() for value in codes.values()),
        protected_aliases={"lm_head.weight": "transformer.wte.weight"},
        protected_buffer_dtype="float32",
        activation_dtype="bfloat16",
        group_scale_dtype="bfloat16",
        state_dtype="bfloat16",
        kv_dtype="bfloat16",
        native_backend="torch_tinygemm_bf16",
        native_quality="not_run",
        tensor_sha256=_digest(payload),
    )
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def load_huginn_int4(folder: Path, device: torch.device) -> HuginnForCausalLM:
    manifest = json.loads((folder / "manifest.json").read_text())
    if (
        manifest["schema_version"] != 1
        or manifest["format"] != "huginn_signed_int4_low_nibble_first"
    ):
        raise ValueError("unsupported Huginn INT4 artifact")
    if device.type != "cuda":
        raise ValueError("native INT4 execution requires CUDA")
    payload = folder / "tensors.pt"
    if _digest(payload) != manifest["tensor_sha256"]:
        raise ValueError("exported tensor hash mismatch")
    if manifest["protected_aliases"] != {"lm_head.weight": "transformer.wte.weight"}:
        raise ValueError("artifact changed the tied Huginn readout")
    tensors = torch.load(payload, map_location="cpu", weights_only=True)
    with torch.device("meta"):
        model = HuginnForCausalLM(HuginnConfig.from_dict(manifest["model_config"]))
    projections = core_projections(model)
    if (
        projections.keys() != tensors["codes"].keys()
        or projections.keys() != tensors["scales"].keys()
    ):
        raise ValueError("artifact changed the core projection set")
    expected = model.state_dict()
    protected = expected.keys() - {name + ".weight" for name in projections} - {"lm_head.weight"}
    if protected != tensors["protected"].keys():
        raise ValueError("artifact changed the protected tensor set")
    for name in protected:
        value = tensors["protected"][name]
        dtype = torch.float32 if name == "freqs_cis" else torch.bfloat16
        if value.shape != expected[name].shape or value.dtype != dtype:
            raise ValueError(f"protected tensor shape or dtype mismatch: {name}")
        if name == "freqs_cis":
            model.register_buffer(name, value.to(device), persistent=True)
        else:
            parent, field = name.rsplit(".", 1)
            model.get_submodule(parent).register_parameter(
                field, nn.Parameter(value.to(device), requires_grad=False)
            )
    model.lm_head.weight = model.transformer.wte.weight
    for name, layer in projections.items():
        parent, field = name.rsplit(".", 1)
        model.get_submodule(parent).add_module(
            field,
            NativeInt4Linear(
                tensors["codes"][name].to(device),
                tensors["scales"][name].to(device),
                layer.in_features,
                manifest["group_size"],
            ),
        )
    return model.eval()
