"""Compare released LoopFormer weights in author, dense and cached CPU execution."""

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import torch

from experiments.loopkv.checkpoint import load_model
from tests.reference.loopformer import dense_loopformer_reference
from vllm_rlt.core.kv_cache_manager import KVCacheManager
from vllm_rlt.models import LoopFormerForCausalLM


def error(reference, actual, atol, rtol):
    delta = (reference.float() - actual.float()).abs()
    failed = (delta > atol + rtol * reference.float().abs()) | ~torch.isfinite(delta)
    locations = failed.nonzero()
    first = tuple(locations[0].tolist()) if len(locations) else None
    return {
        "max_abs": float(delta.max()),
        "failed_elements": int(failed.sum()),
        "elements": reference.numel(),
        "first_difference": (
            {"index": first, "reference": float(reference[first]), "actual": float(actual[first])}
            if first is not None
            else None
        ),
    }


def greedy_difference(reference, actual):
    expected, observed = reference.argmax(-1), actual.argmax(-1)
    differences = (expected != observed).nonzero().flatten()
    if not len(differences):
        return None
    position = int(differences[0])
    left, right = reference[position].float(), actual[position].float()
    return {
        "different_positions": differences.tolist(),
        "first_position": position,
        "reference_token": int(expected[position]),
        "actual_token": int(observed[position]),
        "reference_top2_margin": float(left.topk(2).values.diff().abs().item()),
        "actual_top2_margin": float(right.topk(2).values.diff().abs().item()),
    }


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--author-source", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prefill-only", action="store_true")
    args = parser.parse_args()
    source_hash = hashlib.sha256(args.author_source.read_bytes()).hexdigest()
    assert source_hash == "a4fdd1330a88a859f5f726a15fdce5b6c37d523fd7f04b9c9765d6c58133e665"
    manifest = json.loads((args.model / "verified-manifest.json").read_text())
    assert manifest["revision"] == "2b4fbaaf4e2510353ef7cfe07a07c671b5226739"
    for item in manifest["files"]:
        payload = args.model / item["path"]
        assert payload.stat().st_size == item["bytes"]
        assert hashlib.sha256(payload.read_bytes()).hexdigest() == item["sha256"]
    assert hashlib.sha256((args.model / "model.safetensors").read_bytes()).hexdigest() == (
        "077fae449dd3af29d0313412ef50036e1d9a37fb8b11d447fbb41da0462d9185"
    )
    prompt_bytes = args.prompts.read_bytes()
    assert hashlib.sha256(prompt_bytes).hexdigest() == (
        "e4d8931479e1961f441fb00cacdab8a2d99f7764c99000f8453619b86dee64b2"
    )
    prompt = json.loads(prompt_bytes)[0]
    spec = importlib.util.spec_from_file_location("pinned_loopformer_author", args.author_source)
    author = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = author
    spec.loader.exec_module(author)
    torch.set_num_threads(2)
    native = load_model(args.model, torch.device("cpu"), torch.float32)
    assert isinstance(native, LoopFormerForCausalLM)
    cfg = native.config
    values = {k: v for k, v in cfg.to_dict().items() if k not in ("model_type", "architectures")}
    with torch.device("meta"):
        original = author.GPT(author.GPTConfig(**values)).eval()
    weights = {name.removeprefix("gpt."): value for name, value in native.state_dict().items()}
    weights["lm_head.weight"] = weights["transformer.wte.weight"]
    original.load_state_dict(weights, strict=True, assign=True)
    del weights
    rows = []
    for dtype, atol, rtol in ((torch.float32, 3e-6, 3e-5), (torch.bfloat16, 0.03, 0.02)):
        native.to(dtype)
        original.to(dtype)
        for count in (1, 7, 33):
            tokens = torch.tensor(prompt[: count + 2])
            captured = [[] for _ in range(cfg.n_layer)]
            handles = []
            for layer, block in enumerate(original.transformer.h.blocks):

                def capture(_module, _inputs, output, layer=layer):
                    captured[layer].append(
                        tuple(
                            value[0].reshape(len(tokens), cfg.n_head, cfg.head_dim).clone()
                            for value in output.chunk(3, -1)[1:]
                        )
                    )

                handles.append(block.attn.c_attn.register_forward_hook(capture))
            author_logits, _ = original(tokens[None])
            for handle in handles:
                handle.remove()
            states, dense_logits, dense_kv = dense_loopformer_reference(native, tokens)
            cache = KVCacheManager(
                cfg.n_layer, cfg.n_head, cfg.head_dim, 64, 16, max_loops=8, dtype=dtype
            )
            assert cache.allocate("checkpoint", len(tokens))
            native_logits = []
            state_checks = []
            chunks = (
                [list(range(len(tokens)))]
                if args.prefill_only
                else [list(range(count)), [count], [count + 1]]
            )
            for positions in chunks:
                hidden = native.prelude(tokens[positions])
                for depth in range(8):
                    hidden, _ = native.recurrent(
                        hidden,
                        ["checkpoint"] * len(positions),
                        [depth] * len(positions),
                        positions,
                        cache,
                    )
                    state_checks.append(
                        dict(
                            error(states[depth][positions], hidden, atol, rtol),
                            depth=depth,
                            positions=positions,
                        )
                    )
                native_logits.append(native.coda(hidden))
            native_logits = torch.cat(native_logits)
            author_kv, native_kv = [], []
            for (depth, layer), wanted in dense_kv.items():
                assert len(captured[layer]) == 8
                for reference, observed, cached in zip(
                    wanted,
                    captured[layer][depth],
                    cache.read(layer, "checkpoint", depth, len(tokens)),
                ):
                    author_kv.append(
                        dict(error(reference, observed, atol, rtol), depth=depth, layer=layer)
                    )
                    native_kv.append(
                        dict(error(reference, cached, atol, rtol), depth=depth, layer=layer)
                    )
            cache.free("checkpoint")

            def aggregate(checks):
                return {
                    "max_abs": max(row["max_abs"] for row in checks),
                    "failed_elements": sum(row["failed_elements"] for row in checks),
                    "elements": sum(row["elements"] for row in checks),
                    "first_failure": next((row for row in checks if row["failed_elements"]), None),
                }

            row = {
                "dtype": str(dtype),
                "prefill": count,
                "forced_decode": 2,
                "native_chunking": "full_prefill" if args.prefill_only else "cached_decode",
                "atol": atol,
                "rtol": rtol,
                "author_logits": error(dense_logits, author_logits[0], atol, rtol),
                "native_logits": error(dense_logits, native_logits, atol, rtol),
                "author_kv": aggregate(author_kv),
                "native_kv": aggregate(native_kv),
                "native_states": aggregate(state_checks),
                "author_greedy_exact": torch.equal(
                    dense_logits.argmax(-1), author_logits[0].argmax(-1)
                ),
                "native_greedy_exact": torch.equal(
                    dense_logits.argmax(-1), native_logits.argmax(-1)
                ),
                "native_greedy_difference": greedy_difference(dense_logits, native_logits),
                "author_greedy_difference": greedy_difference(dense_logits, author_logits[0]),
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
            args.out.write_text(
                json.dumps(
                    {
                        "scope": "official weights; CPU teacher forcing, not task quality or speed",
                        "source_sha256": source_hash,
                        "checkpoint": manifest,
                        "torch": torch.__version__,
                        "cases": rows,
                    },
                    indent=2,
                )
                + "\n"
            )
    raise SystemExit(
        int(
            any(
                row[field]["failed_elements"]
                for row in rows
                for field in (
                    "author_logits",
                    "native_logits",
                    "author_kv",
                    "native_kv",
                    "native_states",
                )
            )
        )
    )


if __name__ == "__main__":
    main()
