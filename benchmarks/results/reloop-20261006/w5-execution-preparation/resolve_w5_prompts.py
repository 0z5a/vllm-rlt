"""Resolve original W5 traces with the existing pinned canonical tokenizer."""

import argparse
import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path
from typing import TypedDict, cast


class TraceRow(TypedDict):
    prompt_id: str
    case_id: str
    ordinal: int
    target_prompt_tokens: int


class Trace(TypedDict):
    status: str
    seed_text: str
    records: list[TraceRow]


class Case(TypedDict):
    case_id: str
    batch_limit_B: int
    outstanding_C: int
    prompt_ids: list[str]
    prompt_length_order: list[int]
    first_B_target_kv_bytes: int
    capacity_status: str
    native_e2e: str


class Templates(TypedDict):
    status: str
    source_config_sha256: str
    model_config_sha256: str
    model_revision: str
    prompt_template_sha256: str
    selected_confirmation_input_sha256: str
    common_pool_bytes: int
    cases: list[Case]


class SelectionInput(TypedDict):
    model_files: dict[str, str]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("templates", "trace", "selection-input", "source", "model", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--kv-gib", type=int, default=64)
    args = parser.parse_args()
    templates = cast(Templates, json.loads(args.templates.read_text()))
    trace = cast(Trace, json.loads(args.trace.read_text()))
    assert digest(args.templates) == (
        "c42eb26b929e95d8b36c465810776d3004bd8ee4a2a6b0fd0993e2d8c005351f"
    )
    assert (
        templates["status"] == "ALL_ORIGINAL_W5_INPUT_TEMPLATES_PREPARED_NOT_ADMITTED"
    )
    assert trace["status"] == "TEMPLATE_NOT_CANONICALLY_TOKENIZED"
    assert digest(args.trace) == templates["prompt_template_sha256"]
    assert (
        digest(args.selection_input) == templates["selected_confirmation_input_sha256"]
    )
    assert (
        digest(args.source / "vllm_rlt/config.py") == templates["source_config_sha256"]
    )
    assert digest(args.model / "config.json") == templates["model_config_sha256"]
    assert args.kv_gib > 0
    assert (
        len(trace["records"]) == len({r["prompt_id"] for r in trace["records"]}) == 820
    )
    for case in templates["cases"]:
        rows = [r for r in trace["records"] if r["case_id"] == case["case_id"]]
        assert len(rows) == 4 * case["outstanding_C"]
        assert case["prompt_ids"] == [r["prompt_id"] for r in rows]
        assert case["prompt_length_order"] == [r["target_prompt_tokens"] for r in rows]
    selected = cast(SelectionInput, json.loads(args.selection_input.read_text()))
    # Tokenization does not read or load checkpoint weights.
    tokenizer_files = (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
    )
    assert all(
        digest(args.model / name) == selected["model_files"][name]
        for name in tokenizer_files
    )
    versions = {
        n: importlib.metadata.version(n) for n in ("transformers", "tokenizers")
    }
    assert versions == {"transformers": "4.54.1", "tokenizers": "0.21.4"}
    sys.path.insert(0, str(args.source))
    import torch

    import vllm_rlt
    from vllm_rlt.models import load_tokenizer

    assert Path(vllm_rlt.__file__).resolve().parents[1] == args.source.resolve()
    assert not torch.cuda.is_initialized()
    tokenizer = load_tokenizer(str(args.model), revision=templates["model_revision"])
    seed = tokenizer.encode(trace["seed_text"], add_special_tokens=False)
    assert seed and not torch.cuda.is_initialized()
    records = [
        {
            **row,
            "prompt_token_ids": (
                seed * ((row["target_prompt_tokens"] + len(seed) - 1) // len(seed))
            )[: row["target_prompt_tokens"]],
        }
        for row in trace["records"]
    ]
    args.output.mkdir(exist_ok=False)
    prompts = args.output / "resolved-prompts.json"
    prompts.write_text(json.dumps({"records": records}, indent=2) + "\n")
    protocol = json.loads(args.templates.read_text())
    protocol.update(
        status="CANONICAL_PROMPTS_FROZEN_NOT_GRANTED_NOT_ADMITTED",
        input_template_sha256=digest(args.templates),
        resolver_sha256=digest(Path(__file__)),
        resolved_prompt_manifest_sha256=digest(prompts),
        tokenizer_versions=versions,
        tokenization={
            "add_special_tokens": False,
            "chat_template": False,
            "seed_token_repetition": True,
        },
        common_pool_bytes=args.kv_gib * 1024**3,
    )
    for case in protocol["cases"]:
        case["static_target_status"] = (
            "PLANNED_CAPACITY_EXCEEDED"
            if case["first_B_target_kv_bytes"] > protocol["common_pool_bytes"]
            else "STATIC_RESERVATION_FITS"
        )
    target = args.output / "resolved-protocol.json"
    target.write_text(json.dumps(protocol, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": protocol["status"],
                "records": len(records),
                "prompt_tokens": sum(len(r["prompt_token_ids"]) for r in records),
                "protocol_sha256": digest(target),
                "CUDA_initialized": torch.cuda.is_initialized(),
            }
        )
    )


if __name__ == "__main__":
    main()
