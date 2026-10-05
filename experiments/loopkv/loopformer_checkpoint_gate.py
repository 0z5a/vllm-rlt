"""Compare released LoopFormer weights in author, dense and cached CPU execution."""

import argparse
import hashlib
import importlib.util
import json
import sys
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import torch

from experiments.loopkv.checkpoint import load_model
from tests.reference.loopformer import dense_loopformer_reference
from vllm_rlt.core.kv_cache_manager import KVCacheManager
from vllm_rlt.kernels.paged_attention import torch_paged_attention
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


def conditioning_errors(native, original, rows, dtype):
    """Observe B1 versus flattened-row conditioning, without changing either path."""
    embedding_error = modulation_error = 0.0
    for depth in range(8):
        time = torch.tensor([depth / 8], dtype=dtype)
        step = torch.full_like(time, 1 / 8)
        expected = original.time_embedder(time) + original.dt_embedder(step)
        observed = native.gpt.time_embedder(time.expand(rows)) + native.gpt.dt_embedder(
            step.expand(rows)
        )
        embedding_error = max(embedding_error, float((observed - expected).abs().max()))
        for left, right in zip(original.transformer.h.blocks, native.gpt.transformer.h.blocks):
            difference = right.adaLN_modulation(observed) - left.adaLN_modulation(expected)
            modulation_error = max(modulation_error, float(difference.abs().max()))
    return {
        "rows": rows,
        "embedding_max_abs": embedding_error,
        "modulation_max_abs": modulation_error,
    }


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--author-source", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prefill-only", action="store_true")
    parser.add_argument("--shared-attention-diagnostic", action="store_true")
    parser.add_argument("--matched-conditioning-diagnostic", action="store_true")
    args = parser.parse_args()
    assert not args.matched_conditioning_diagnostic or args.prefill_only
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
    attention_calls = 0
    conditioning_calls = 0

    def shared_attention(q, k, v, attn_mask=None, dropout_p=0, is_causal=False):
        nonlocal attention_calls
        attention_calls += 1
        assert is_causal and attn_mask is None and dropout_p == 0 and q.shape == k.shape == v.shape
        batch, heads, length, width = q.shape
        tables = torch.arange(batch).repeat_interleave(length)[:, None]
        lengths = torch.arange(1, length + 1).repeat(batch)
        output = torch_paged_attention(
            q.transpose(1, 2).reshape(-1, heads, width),
            k.transpose(1, 2),
            v.transpose(1, 2),
            tables,
            lengths,
        )
        return output.reshape(batch, length, heads, width).transpose(1, 2)

    rows = []
    for dtype, atol, rtol in ((torch.float32, 3e-6, 3e-5), (torch.bfloat16, 0.03, 0.02)):
        native.to(dtype)
        original.to(dtype)
        for count in (1, 7, 33):
            tokens = torch.tensor(prompt[: count + 2])
            captured = [[] for _ in range(cfg.n_layer)]
            author_states = []
            handles = []
            handles.append(
                original.transformer.h.register_forward_hook(
                    lambda _module, _inputs, output: author_states.append(output[0].clone())
                )
            )
            for layer, block in enumerate(original.transformer.h.blocks):

                def capture(_module, _inputs, output, layer=layer):
                    captured[layer].append(
                        tuple(
                            value[0].reshape(len(tokens), cfg.n_head, cfg.head_dim).clone()
                            for value in output.chunk(3, -1)[1:]
                        )
                    )

                handles.append(block.attn.c_attn.register_forward_hook(capture))
            with ExitStack() as stack:
                if args.shared_attention_diagnostic:
                    stack.enter_context(
                        patch.object(author.F, "scaled_dot_product_attention", shared_attention)
                    )
                if args.matched_conditioning_diagnostic:
                    modules = [original.time_embedder, original.dt_embedder] + [
                        block.adaLN_modulation for block in original.transformer.h.blocks
                    ]
                    for module in modules:

                        def expanded(value, forward=module.forward):
                            nonlocal conditioning_calls
                            conditioning_calls += 1
                            assert value.shape[0] == 1
                            return forward(value.expand(len(tokens), *value.shape[1:]))[:1]

                        stack.enter_context(patch.object(module, "forward", expanded))
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
            author_state_checks = []
            author_states_exact = author_kv_exact = True
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
                    wanted = author_states[depth][positions]
                    actual = hidden[:, : cfg.n_embd]
                    author_state_checks.append(error(wanted, actual, atol, rtol))
                    author_states_exact &= torch.equal(
                        wanted.contiguous().view(torch.uint8), actual.contiguous().view(torch.uint8)
                    )
                native_logits.append(native.coda(hidden))
            native_logits = torch.cat(native_logits)
            author_kv, native_kv, native_author_kv = [], [], []
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
                    native_author_kv.append(error(observed, cached, atol, rtol))
                    author_kv_exact &= torch.equal(
                        observed.contiguous().view(torch.uint8),
                        cached.contiguous().view(torch.uint8),
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
                "native_author_logits": error(author_logits[0], native_logits, atol, rtol),
                "native_author_bitwise_equal": torch.equal(
                    author_logits[0].contiguous().view(torch.uint8),
                    native_logits.contiguous().view(torch.uint8),
                ),
                "native_author_kv": aggregate(native_author_kv),
                "native_author_kv_bitwise_equal": author_kv_exact,
                "native_author_states": aggregate(author_state_checks),
                "native_author_states_bitwise_equal": author_states_exact,
                "conditioning": [
                    conditioning_errors(native, original, size, dtype)
                    for size in sorted({len(positions) for positions in chunks})
                ],
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
                        "scope": (
                            "author arithmetic diagnostic; not unchanged author qualification"
                            if args.shared_attention_diagnostic
                            or args.matched_conditioning_diagnostic
                            else "official weights; CPU teacher forcing, not task quality or speed"
                        ),
                        "substituted_attention_calls": attention_calls,
                        "expanded_conditioning_calls": conditioning_calls,
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
