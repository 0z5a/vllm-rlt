"""Private single-file input adapter; the public Ouro loader stays unchanged.

Uses the independently copied Torch-only transport after complete header checks.
Official file hashes and the exact production source must be admitted by callers.
This qualifies native model mathematics, not the public safetensors loader.
"""

import json
import math
import struct
from pathlib import Path

import torch
from peer_checkpoint import load_weights
from vllm_rlt.models.ouro import OuroConfig, OuroForCausalLM


def unique_fields(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate header field: {key}")
        result[key] = value
    return result


def validate_header(path, expected):
    """Validate every tensor before allocating any parameter storage."""
    size = path.stat().st_size
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError("truncated header length")
        length = struct.unpack("<Q", prefix)[0]
        if not 2 <= length <= min(100_000_000, size - 8):
            raise ValueError("invalid header length")
        header = json.loads(stream.read(length), object_pairs_hook=unique_fields)
    tensors = {key: value for key, value in header.items() if key != "__metadata__"}
    if tensors.keys() != expected.keys():
        raise ValueError("checkpoint keys differ from native Ouro")
    bytes_per_element = {"BF16": 2, "F16": 2, "F32": 4}
    intervals = []
    for name, spec in tensors.items():
        if set(spec) != {"shape", "dtype", "data_offsets"}:
            raise ValueError(f"invalid tensor metadata: {name}")
        shape, offsets = spec["shape"], spec["data_offsets"]
        if (
            any(type(d) is not int or d <= 0 for d in shape)
            or tuple(shape) != tuple(expected[name].shape)
            or spec["dtype"] not in bytes_per_element
            or len(offsets) != 2
            or any(type(offset) is not int for offset in offsets)
        ):
            raise ValueError(f"invalid tensor shape/dtype/offsets: {name}")
        begin, end = offsets
        if not 0 <= begin < end <= size - 8 - length:
            raise ValueError(f"tensor is outside payload: {name}")
        if end - begin != math.prod(shape) * bytes_per_element[spec["dtype"]]:
            raise ValueError(f"tensor byte count differs: {name}")
        intervals.append((begin, end))
    cursor = 0
    for begin, end in sorted(intervals):
        if begin != cursor:
            raise ValueError("overlapping or incomplete tensor payload")
        cursor = end
    if cursor != size - 8 - length:
        raise ValueError("unclaimed trailing tensor payload")


def load_ouro(folder: Path, *, device="cpu", dtype=torch.bfloat16):
    """Use the original config/model/parameter mapping, with a private input reader."""
    if dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise ValueError("unsupported output dtype")
    if (folder / "model.safetensors.index.json").exists():
        raise ValueError("this private official-input adapter supports one file only")
    path = folder / "model.safetensors"
    if path.is_symlink() or not path.is_file():
        raise ValueError("requires a regular local safetensors file")
    config = OuroConfig.from_dict(json.loads((folder / "config.json").read_text()))
    with torch.device("meta"):
        model = OuroForCausalLM(config)
    validate_header(path, dict(model.named_parameters()))
    load_weights(model, folder, torch.device(device), dtype)
    model.model.rotary_emb.inv_freq = model.model.rotary_emb.frequencies(device=device)
    return model.eval()
