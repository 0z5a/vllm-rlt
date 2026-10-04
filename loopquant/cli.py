"""Preflight, activation tracing, and audited result rendering for loopquant."""

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import platform
import sys
from pathlib import Path

import torch

from .data import Window, validate_windows
from .report import TrialPair, speedup_table


def read_config(path: Path) -> dict:
    config = json.loads(path.read_text())
    for name in ("model_revision", "runtime_revision"):
        revision = config[name]
        if len(revision) != 40 or any(ch not in "0123456789abcdef" for ch in revision):
            raise ValueError(f"{name} must be a full immutable commit")
    if config["format"] not in ("bf16", "fp8_e4m3fn", "int4_symmetric"):
        raise ValueError("format has not been implemented")
    return config


def preflight(config: dict) -> dict:
    manifest = Path(config["data_manifest"])
    files = json.loads(manifest.read_text())
    windows = []
    for filename in files["files"]:
        for line in (manifest.parent / filename).read_text().splitlines():
            row = json.loads(line)
            row["token_ids"] = tuple(row["token_ids"])
            windows.append(Window(**row))
    validate_windows(windows)
    names = ("torch", "triton", "transformers", "safetensors", "tokenizers")
    packages = {
        name: importlib.metadata.version(name) if importlib.util.find_spec(name) else None
        for name in names
    }
    return {
        "python": sys.executable,
        "python_version": platform.python_version(),
        "packages": packages,
        "model_revision": config["model_revision"],
        "runtime_revision": config["runtime_revision"],
        "data_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "windows": len(windows),
        "documents": len({row.document_id for row in windows}),
        "cuda_available": torch.cuda.is_available(),
        "native_backend": "unverified; run the reserved GPU probe",
        "gates": dict.fromkeys(
            ("G0_semantics", "G1_potential", "G2_export_quality", "G3_performance"), "not_run"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("preflight")
    check.add_argument("--config", type=Path, required=True)
    check.add_argument("--output", type=Path, required=True)
    report = commands.add_parser("report")
    report.add_argument("--pairs", type=Path, required=True)
    report.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "preflight":
        result = preflight(read_config(args.config))
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    else:
        raw = json.loads(args.pairs.read_text())
        pairs = {case: [TrialPair(**row) for row in rows] for case, rows in raw.items()}
        args.output.write_text(speedup_table(pairs))


if __name__ == "__main__":
    main()
