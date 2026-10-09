"""Check that the production string-prompt HTTP path preserves frozen token IDs."""

import argparse
import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["prompts", "protocol", "selection-input", "source", "model", "output"]:
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    prompts = json.loads(args.prompts.read_text())["records"]
    selected = json.loads(args.selection_input.read_text())
    assert protocol["status"] == "CANONICAL_PROMPTS_FROZEN_NOT_GRANTED_NOT_ADMITTED"
    assert digest(args.prompts) == protocol["resolved_prompt_manifest_sha256"]
    assert (
        digest(args.source / "vllm_rlt/config.py") == protocol["source_config_sha256"]
    )
    for name in [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
    ]:
        assert digest(args.model / name) == selected["model_files"][name]
    versions = {
        name: importlib.metadata.version(name)
        for name in ["transformers", "tokenizers"]
    }
    assert versions == protocol["tokenizer_versions"]
    assert len(prompts) == len({row["prompt_id"] for row in prompts})
    assert args.output.parent.is_dir() and not args.output.exists()
    sys.path.insert(0, str(args.source))
    import torch

    from vllm_rlt.models import load_tokenizer

    loader_path = Path(load_tokenizer.__code__.co_filename).resolve()
    assert loader_path == (args.source / "vllm_rlt/models/__init__.py").resolve()
    assert not torch.cuda.is_initialized()
    tokenizer = load_tokenizer(str(args.model), revision=protocol["model_revision"])
    unique = {}
    resolved = []
    mismatches = []
    for row in prompts:
        key = tuple(row["prompt_token_ids"])
        if key not in unique:
            text = tokenizer.decode(
                key, skip_special_tokens=False, clean_up_tokenization_spaces=False
            )
            # This deliberately uses the same default encode call as EngineWorker.
            unique[key] = text, tokenizer.encode(text)
        text, actual_ids = unique[key]
        resolved.append(
            {**row, "prompt": text, "actual_HTTP_encode_token_ids": actual_ids}
        )
        if actual_ids != row["prompt_token_ids"]:
            mismatches.append(row["prompt_id"])
    assert not torch.cuda.is_initialized()
    receipt = {
        "state": "EXACT_HTTP_STRING_PROMPTS_PASS"
        if not mismatches
        else "HTTP_TOKEN_ROUNDTRIP_MISMATCH",
        "protocol_sha256": digest(args.protocol),
        "prompts_sha256": digest(args.prompts),
        "preparer_sha256": digest(Path(__file__)),
        "loader_source_file": "vllm_rlt/models/__init__.py",
        "loader_source_sha256": digest(loader_path),
        "loader_origin_matches_bound_source": True,
        "tokenizer_versions": versions,
        "records": resolved,
        "unique_token_sequences": len(unique),
        "mismatched_prompt_ids": mismatches,
        "CUDA_initialized": False,
        "checkpoint_weights_read": False,
        "HTTP_or_GPU_E2E_executed": False,
    }
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: receipt[key]
                for key in [
                    "state",
                    "unique_token_sequences",
                    "mismatched_prompt_ids",
                    "CUDA_initialized",
                    "HTTP_or_GPU_E2E_executed",
                ]
            }
        )
    )
    assert not mismatches, (
        "Frozen token inputs cannot be replaced by different HTTP work"
    )


if __name__ == "__main__":
    main()
