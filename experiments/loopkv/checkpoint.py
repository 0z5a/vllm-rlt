"""Read verified safetensors with PyTorch alone in an immutable existing environment."""

import json
import math
import mmap
import struct
from pathlib import Path

import torch
from torch import nn

from vllm_rlt.models import (
    HrmTextConfig,
    HrmTextForCausalLM,
    HuginnConfig,
    HuginnForCausalLM,
    LoopFormerConfig,
    LoopFormerForCausalLM,
    NanbeigeConfig,
    NanbeigeForCausalLM,
    OuroConfig,
    OuroForCausalLM,
    ParcaeForCausalLM,
    model_type_from_config,
)


def load_model(
    folder: Path, device: torch.device, dtype: torch.dtype
) -> (
    OuroForCausalLM
    | NanbeigeForCausalLM
    | HuginnForCausalLM
    | HrmTextForCausalLM
    | LoopFormerForCausalLM
    | ParcaeForCausalLM
):
    """Construct a native recurrent model without optional Hub dependencies."""
    config = json.loads((folder / "config.json").read_text())
    if model_type_from_config(config) == "parcae":
        return ParcaeForCausalLM.from_pretrained(folder, device=device, dtype=dtype)
    with torch.device("meta"):
        if config["model_type"] == "loopformer":
            model = LoopFormerForCausalLM(LoopFormerConfig.from_dict(config))
        elif config["model_type"] == "hrm_text":
            model = HrmTextForCausalLM(HrmTextConfig.from_dict(config))
        elif config["model_type"] == "ouro":
            model = OuroForCausalLM(OuroConfig.from_dict(config))
        elif config["model_type"] == "nanbeige":
            model = NanbeigeForCausalLM(NanbeigeConfig.from_dict(config))
        elif config["model_type"] == "huginn_raven":
            model = HuginnForCausalLM(HuginnConfig.from_dict(config))
        else:
            raise ValueError(f"unsupported standalone model: {config['model_type']}")
    load_weights(model, folder, device, dtype)
    if isinstance(model, HuginnForCausalLM):
        if not torch.equal(model.lm_head.weight, model.transformer.wte.weight):
            raise ValueError("tied checkpoint embeddings differ")
        model.lm_head.weight = model.transformer.wte.weight
    elif not isinstance(model, LoopFormerForCausalLM):
        model.model.rotary_emb.inv_freq = model.model.rotary_emb.frequencies(device=device)
    return model.eval()


def load_weights(model: nn.Module, folder: Path, device: torch.device, dtype: torch.dtype) -> None:
    """Stream each tensor into owned storage; never retain the file-backed mapping."""
    index = folder / "model.safetensors.index.json"
    weight_map = json.loads(index.read_text())["weight_map"] if index.is_file() else None
    names = sorted(set(weight_map.values())) if weight_map else ["model.safetensors"]
    expected = model.state_dict()
    buffers = dict(model.named_buffers())
    loaded: set[str] = set()
    dtypes = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}
    for name in names:
        if Path(name).name != name:
            raise ValueError("shard must be a local basename")
        with (folder / name).open("rb") as file:
            header_size = struct.unpack("<Q", file.read(8))[0]
            if header_size > 100_000_000:
                raise ValueError("invalid safetensors header length")
            header = json.loads(file.read(header_size))
            with mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_COPY) as mapping:
                for key, spec in header.items():
                    if key == "__metadata__":
                        continue
                    if key in loaded or key not in expected:
                        raise ValueError(f"unexpected or duplicate tensor: {key}")
                    if tuple(spec["shape"]) != tuple(expected[key].shape):
                        raise ValueError(f"checkpoint shape mismatch: {key}")
                    if weight_map is not None and weight_map[key] != name:
                        raise ValueError(f"checkpoint shard mismatch: {key}")
                    begin, end = spec["data_offsets"]
                    source_dtype = dtypes[spec["dtype"]]
                    elements = math.prod(spec["shape"])
                    if end - begin != elements * source_dtype.itemsize:
                        raise ValueError(f"checkpoint byte count mismatch: {key}")
                    view = torch.frombuffer(
                        mapping, dtype=source_dtype, count=elements, offset=8 + header_size + begin
                    ).reshape(spec["shape"])
                    target_dtype = buffers[key].dtype if key in buffers else dtype
                    owned = view.to(device=device, dtype=target_dtype, copy=True)
                    parent, field = key.rsplit(".", 1) if "." in key else ("", key)
                    module = model.get_submodule(parent)
                    if key in buffers:
                        module.register_buffer(field, owned, persistent=True)
                    else:
                        module.register_parameter(field, nn.Parameter(owned, requires_grad=False))
                    del view
                    loaded.add(key)
    if loaded != expected.keys():
        raise ValueError(f"missing checkpoint tensors: {sorted(expected.keys() - loaded)}")
