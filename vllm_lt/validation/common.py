"""Shared public JSON, provenance and input checks for repository tools."""

import hashlib
import json
import math
import subprocess
from pathlib import Path

from vllm_lt.models.config import OuroConfig


def keys(value, required, *, optional=(), name="object"):
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    missing, unknown = set(required) - value.keys(), value.keys() - set(required) - set(optional)
    if missing or unknown:
        raise ValueError(
            f"{name}: missing fields {sorted(missing)}, unknown fields {sorted(unknown)}"
        )


def integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def validate_text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")


def reject_constant(value):
    raise ValueError(f"nonfinite JSON value: {value}")


def finite_float(value):
    number = float(value)
    if not math.isfinite(number):
        reject_constant(value)
    return number


def object_pairs(pairs):
    result = {}
    for name, value in pairs:
        if name in result:
            raise ValueError(f"duplicate JSON field: {name}")
        result[name] = value
    return result


def read_json(path):
    return json.loads(
        Path(path).read_text(encoding="utf-8"),
        parse_constant=reject_constant,
        parse_float=finite_float,
        object_pairs_hook=object_pairs,
    )


def write_json(path, value):
    """Write only finite UTF-8 JSON; directory ownership belongs to the caller."""
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def digest(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def make_file_record(path, *, relative_to=None):
    path = Path(path)
    checksum = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            checksum.update(chunk)
            size += len(chunk)
    return {
        "path": str(path.relative_to(relative_to) if relative_to else path.resolve()),
        "size_bytes": size,
        "sha256": checksum.hexdigest(),
    }


def validate_version(value, artifact_type):
    if type(value.get("schema_version")) is not int or value["schema_version"] != 1:
        raise ValueError("unsupported schema_version")
    if value.get("artifact_type") != artifact_type:
        raise ValueError(f"expected artifact_type={artifact_type}")


def tokens(value, length, name):
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{name} must contain exactly {length} token IDs")
    for token in value:
        integer(token, name)
        if token >= OuroConfig.vocab_size:
            raise ValueError(f"{name} token outside Ouro vocabulary")


def constants(actual, expected, name):
    keys(actual, expected, name=name)
    for key, value in expected.items():
        found = actual[key]
        numeric = type(value) is float and type(found) in (int, float)
        if (not numeric and type(found) is not type(value)) or found != value:
            raise ValueError(f"{name}.{key} must equal {value!r}")


def source_manifest():
    root = Path(__file__).resolve().parents[2]

    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args])

    names = git("ls-files", "-z", "--cached", "--others", "--exclude-standard").decode().split("\0")
    return {
        "root": str(root),
        "commit": git("rev-parse", "HEAD").decode().strip(),
        "status": git("status", "--porcelain", "--untracked-files=all").decode(),
        "files": [
            make_file_record(root / name, relative_to=root)
            for name in sorted(set(names))
            if name and (root / name).is_file()
        ],
    }


def model_files(model_path):
    files = [
        path
        for path in sorted(model_path.iterdir())
        if path.is_file() and path.suffix in {".safetensors", ".json", ".txt", ".model"}
    ]
    names = {path.name for path in files}
    if not {"config.json", "tokenizer.json", "tokenizer_config.json"} <= names or not any(
        path.suffix == ".safetensors" for path in files
    ):
        raise ValueError("prepared model requires config, tokenizer files and safetensors weights")
    index = model_path / "model.safetensors.index.json"
    if index.exists():
        mapping = read_json(index).get("weight_map", {})
        if not mapping or not set(mapping.values()) <= names:
            raise ValueError("checkpoint index refers to missing weight shards")
    return [make_file_record(path, relative_to=model_path) for path in files]
