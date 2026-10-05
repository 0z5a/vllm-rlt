"""Encoded LoopFormer FP8 artifacts with protected time conditioning and tied readout."""

import json
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn

from vllm_rlt.models.loopformer import LoopFormerConfig, LoopFormerForCausalLM

from .fused_fp8 import RowwiseFP8Linear
from .int4_export import core_projections
from .loopformer_fp8 import FP8LoopFormerBlock, quantize_loopformer_fp8
from .native_export import _digest
from .quantizers import ScaleLayout


def export_loopformer_fp8(
    model: LoopFormerForCausalLM, folder: Path, *, model_revision: str
) -> None:
    blocks = list(model.gpt.transformer.h.blocks)
    if not blocks or not all(isinstance(block, FP8LoopFormerBlock) for block in blocks):
        raise ValueError("export requires every LoopFormer core block converted")
    producer = blocks[0].producers["attn_c_attn"]
    if any(
        p.layout != producer.layout or p.mode != producer.mode
        for block in blocks
        for p in block.producers.values()
    ):
        raise ValueError("one registered scale layout and mode must cover all core blocks")
    weights = [
        layer.packed_weight for layer in model.modules() if isinstance(layer, RowwiseFP8Linear)
    ]
    if len(weights) != len(blocks) * 4 or len({w.data_ptr() for w in weights}) != len(weights):
        raise ValueError("each physical core projection must own one packed weight")
    folder.mkdir(parents=True, exist_ok=False)
    payload = folder / "tensors.pt"
    torch.save({name: value.detach().cpu() for name, value in model.state_dict().items()}, payload)
    manifest = dict(
        schema_version=1,
        format="loopformer_rowwise_fp8_e4m3fn",
        model_revision=model_revision,
        model_config=model.config.to_dict(),
        layout=asdict(producer.layout),
        mode=producer.mode,
        normalization_fusion=False,
        packed_weight_matrices=len(weights),
        packed_weight_copies=1,
        packed_weight_bytes=sum(w.numel() for w in weights),
        protected_conditioning_dtype="bfloat16",
        state_dtype="bfloat16",
        kv_dtype="bfloat16",
        native_quality="not_run",
        tensor_sha256=_digest(payload),
    )
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def load_loopformer_fp8(
    folder: Path, device: torch.device, *, max_rows: int
) -> LoopFormerForCausalLM:
    manifest = json.loads((folder / "manifest.json").read_text())
    if manifest["schema_version"] != 1 or manifest["format"] != "loopformer_rowwise_fp8_e4m3fn":
        raise ValueError("unsupported native LoopFormer FP8 artifact")
    if device.type != "cuda":
        raise ValueError("native FP8 execution requires CUDA")
    payload = folder / "tensors.pt"
    if _digest(payload) != manifest["tensor_sha256"]:
        raise ValueError("exported tensor hash mismatch")
    tensors = torch.load(payload, map_location="cpu", weights_only=True)
    with torch.device("meta"):
        model = LoopFormerForCausalLM(LoopFormerConfig.from_dict(manifest["model_config"]))
    projections = core_projections(model)
    expected = model.state_dict()
    protected = expected.keys() - {name + ".weight" for name in projections}
    scale_names = {
        name: name.rsplit(".", 2)[0]
        + ".producers."
        + ".".join(name.split(".")[-2:]).replace(".", "_")
        + ".scales"
        for name in projections
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
        if value.dtype != torch.bfloat16 or value.shape != expected[name].shape:
            raise ValueError(f"protected tensor shape or dtype mismatch: {name}")
        parent, field = name.rsplit(".", 1)
        model.get_submodule(parent).register_parameter(
            field, nn.Parameter(value.to(device), requires_grad=False)
        )
    scales = {name: tensors[key].to(device) for name, key in scale_names.items()}
    packed = {
        name: (
            tensors[name + ".packed_weight"].to(device),
            tensors[name + ".weight_scales"].to(device),
        )
        for name in projections
    }
    layout = ScaleLayout(manifest["layout"]["max_loops"], tuple(manifest["layout"]["boundaries"]))
    quantize_loopformer_fp8(
        model, scales, layout, manifest["mode"], max_rows=max_rows, packed=packed
    )
    return model.eval()
