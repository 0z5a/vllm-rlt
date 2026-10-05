"""Read verified safetensors with PyTorch alone in an immutable existing environment."""

import json
import math
import mmap
import struct
from pathlib import Path

import torch
from torch import nn


def load_weights(model: nn.Module, folder: Path, device: torch.device, dtype: torch.dtype) -> None:
    """Stream each tensor into owned storage; never retain the file-backed mapping."""
    index = folder / "model.safetensors.index.json"
    weight_map = json.loads(index.read_text())["weight_map"] if index.is_file() else None
    names = sorted(set(weight_map.values())) if weight_map else ["model.safetensors"]
    parameters = dict(model.named_parameters(remove_duplicate=False))
    expected = model.state_dict(keep_vars=True)
    shared: dict[int, nn.Parameter] = {}
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
                    target_dtype = dtype if key in parameters else expected[key].dtype
                    owned = view.to(device=device, dtype=target_dtype, copy=True)
                    parent, _, field = key.rpartition(".")
                    module = model.get_submodule(parent)
                    if key in parameters:
                        identity = id(parameters[key])
                        if identity in shared:
                            if not torch.equal(shared[identity], owned):
                                raise ValueError(f"tied checkpoint tensors differ: {key}")
                        else:
                            shared[identity] = nn.Parameter(owned, requires_grad=False)
                        module.register_parameter(field, shared[identity])
                    else:
                        module.register_buffer(field, owned, persistent=True)
                    del view
                    loaded.add(key)
    if loaded != expected.keys():
        raise ValueError(f"missing checkpoint tensors: {sorted(expected.keys() - loaded)}")
