"""Portable INT4 core artifacts loaded into one native tinygemm matrix per layer."""

import json
from pathlib import Path

import torch
from torch import nn

from vllm_rlt.models.hrm_text import HrmTextConfig, HrmTextForCausalLM
from vllm_rlt.models.loopformer import LoopFormerConfig, LoopFormerForCausalLM
from vllm_rlt.models.nanbeige import NanbeigeConfig, NanbeigeForCausalLM
from vllm_rlt.models.ouro import OuroConfig, OuroForCausalLM
from vllm_rlt.models.parcae import ParcaeConfig, ParcaeForCausalLM

from .int4_backend import NativeInt4Linear
from .native_export import _digest
from .quantizers import int4_pack

Int4Model = (
    OuroForCausalLM
    | NanbeigeForCausalLM
    | HrmTextForCausalLM
    | LoopFormerForCausalLM
    | ParcaeForCausalLM
)


def core_projections(model: Int4Model) -> dict[str, nn.Linear]:
    """Physical shared matrices; clocks, gates, embeddings and heads stay protected."""
    if isinstance(model, ParcaeForCausalLM):
        names = [
            f"transformer.core_block.{index}.{suffix}"
            for index in range(model.config.n_layers_in_recurrent_block)
            for suffix in ("attn.c_q", "attn.c_k", "attn.c_v", "attn.c_proj", "mlp.fc", "mlp.proj")
        ]
    elif isinstance(model, LoopFormerForCausalLM):
        names = [
            f"gpt.transformer.h.blocks.{index}.{suffix}"
            for index in range(model.config.n_layer)
            for suffix in ("attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj")
        ]
    elif isinstance(model, HrmTextForCausalLM):
        names = [
            f"model.{module}_module.layers.{index}.{suffix}"
            for module in ("H", "L")
            for index in range(model.config.module_layers)
            for suffix in ("attn.gqkv_proj", "attn.o_proj", "mlp.gate_up_proj", "mlp.down_proj")
        ]
    else:
        names = [
            f"model.layers.{index}.{suffix}"
            for index in range(model.config.num_hidden_layers)
            for suffix in (
                "self_attn.q_proj",
                "self_attn.k_proj",
                "self_attn.v_proj",
                "self_attn.o_proj",
                "mlp.gate_proj",
                "mlp.up_proj",
                "mlp.down_proj",
            )
        ]
    result = {}
    for name in names:
        layer = model.get_submodule(name)
        if not isinstance(layer, nn.Linear):
            raise ValueError(f"expected an unconverted physical projection: {name}")
        result[name] = layer
    return result


def export_int4_model(
    model: Int4Model,
    folder: Path,
    *,
    model_revision: str,
    method: str = "RTN",
    packed: dict[str, tuple[torch.Tensor, torch.Tensor]] | None = None,
    group_size: int = 128,
) -> None:
    """Store signed low-nibble codes; native BF16 scale rounding is explicit."""
    projections = core_projections(model)
    if packed is not None and projections.keys() != packed.keys():
        raise ValueError("calibrated weights must cover every physical projection")
    if method not in ("RTN", "GPTQ-first", "GPTQ-all", "GPTQ-matched"):
        raise ValueError("unregistered INT4 weight baseline")
    if (packed is None) != (method == "RTN"):
        raise ValueError("GPTQ export requires its calibrated packed matrices")
    codes, scales = {}, {}
    for name, layer in projections.items():
        if layer.bias is not None:
            raise ValueError("INT4 core projections must be bias-free")
        weight, scale = (
            int4_pack(layer.weight.detach(), group_size) if packed is None else packed[name]
        )
        columns = ((layer.in_features + group_size - 1) // group_size) * group_size
        if weight.shape != (layer.out_features, columns // 2) or scale.shape != (
            layer.out_features,
            columns // group_size,
        ):
            raise ValueError(f"packed shape does not match {name}")
        codes[name], scales[name] = weight.cpu(), scale.to(device="cpu", dtype=torch.bfloat16)
        if weight.dtype != torch.uint8 or not bool(
            torch.isfinite(scales[name]).all() & (scales[name] > 0).all()
        ):
            raise ValueError(f"invalid codes or native BF16 scales for {name}")
    aliases = (
        {"lm_head.weight": "transformer.wte.weight"} if isinstance(model, ParcaeForCausalLM) else {}
    )
    if aliases and model.lm_head.weight is not model.transformer.wte.weight:
        raise ValueError("Parcae requires a tied embedding/head")
    protected = {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
        if name not in {key + ".weight" for key in projections} and name not in aliases
    }
    if any(
        value.dtype != (torch.float32 if name == "freqs_cis" else torch.bfloat16)
        for name, value in protected.items()
    ):
        raise ValueError("protected tensors require BF16 weights and FP32 rotary buffers")
    folder.mkdir(parents=True, exist_ok=False)
    payload = folder / "tensors.pt"
    torch.save(dict(codes=codes, scales=scales, protected=protected), payload)
    manifest = dict(
        schema_version=1,
        format="signed_int4_low_nibble_first",
        native_backend="torch_tinygemm_bf16",
        method=method,
        group_size=group_size,
        model_revision=model_revision,
        model_config=model.config.to_dict(),
        packed_weight_matrices=len(codes),
        packed_weight_copies=1,
        packed_weight_bytes=sum(value.numel() for value in codes.values()),
        protected_aliases=aliases,
        group_scale_dtype="bfloat16",
        activation_dtype="bfloat16",
        state_dtype="bfloat16",
        kv_dtype="bfloat16",
        native_quality="not_run",
        tensor_sha256=_digest(payload),
    )
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def load_int4_model(folder: Path, device: torch.device) -> Int4Model:
    manifest = json.loads((folder / "manifest.json").read_text())
    if manifest["schema_version"] != 1 or manifest["format"] != "signed_int4_low_nibble_first":
        raise ValueError("unsupported INT4 model artifact")
    if device.type != "cuda":
        raise ValueError("native INT4 model execution requires CUDA")
    payload = folder / "tensors.pt"
    if _digest(payload) != manifest["tensor_sha256"]:
        raise ValueError("exported INT4 tensor hash mismatch")
    tensors = torch.load(payload, map_location="cpu", weights_only=True)
    with torch.device("meta"):
        config = manifest["model_config"]
        if config["model_type"] == "ouro":
            model = OuroForCausalLM(OuroConfig.from_dict(config))
        elif config["model_type"] == "nanbeige":
            model = NanbeigeForCausalLM(NanbeigeConfig.from_dict(config))
        elif config["model_type"] == "hrm_text":
            model = HrmTextForCausalLM(HrmTextConfig.from_dict(config))
        elif config["model_type"] == "loopformer":
            model = LoopFormerForCausalLM(LoopFormerConfig.from_dict(config))
        elif config["model_type"] == "parcae":
            model = ParcaeForCausalLM(ParcaeConfig.from_dict(config))
        else:
            raise ValueError("unsupported INT4 model family")
    projections = core_projections(model)
    if (
        projections.keys() != tensors["codes"].keys()
        or projections.keys() != tensors["scales"].keys()
    ):
        raise ValueError("artifact omitted or added a core projection")
    expected = model.state_dict()
    aliases = (
        {"lm_head.weight": "transformer.wte.weight"} if isinstance(model, ParcaeForCausalLM) else {}
    )
    if aliases and manifest["protected_aliases"] != aliases:
        raise ValueError("artifact changed the tied Parcae readout")
    protected = expected.keys() - {name + ".weight" for name in projections} - aliases.keys()
    if protected != tensors["protected"].keys():
        raise ValueError("artifact changed the protected parameter set")
    for name in protected:
        value = tensors["protected"][name]
        dtype = torch.float32 if name == "freqs_cis" else torch.bfloat16
        if value.shape != expected[name].shape or value.dtype != dtype:
            raise ValueError(f"protected tensor mismatch: {name}")
        if name == "freqs_cis":
            model.register_buffer(name, value.to(device), persistent=True)
        else:
            parent, field = name.rsplit(".", 1)
            model.get_submodule(parent).register_parameter(
                field, nn.Parameter(value.to(device), requires_grad=False)
            )
    if isinstance(model, ParcaeForCausalLM):
        model.lm_head.weight = model.transformer.wte.weight
    for name, layer in projections.items():
        parent, field = name.rsplit(".", 1)
        model.get_submodule(parent).add_module(
            field,
            NativeInt4Linear(
                tensors["codes"].pop(name).to(device),
                tensors["scales"].pop(name).to(device),
                layer.in_features,
                manifest["group_size"],
            ),
        )
    if not isinstance(model, (LoopFormerForCausalLM, ParcaeForCausalLM)):
        model.model.rotary_emb.inv_freq = model.model.rotary_emb.frequencies(device=device)
    return model.eval()
