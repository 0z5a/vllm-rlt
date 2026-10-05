"""Check pinned author generation against native KV and a dense functional oracle."""

import argparse
import hashlib
import json
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

import torch

from experiments.loopkv.capture import dump
from tests.reference.parcae import dense_parcae_reference
from vllm_rlt.core.kv_cache_manager import KVCacheManager
from vllm_rlt.kernels.paged_attention import torch_paged_attention
from vllm_rlt.models import ParcaeConfig, ParcaeForCausalLM


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--author-repo", type=Path, required=True)
    parser.add_argument("--official-config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--shared-attention-diagnostic", action="store_true")
    args = parser.parse_args()
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=args.author_repo, text=True
    ).strip()
    assert revision == "69284c13746e849104f738d6d1a347b1f457df76"
    assert not subprocess.check_output(["git", "diff", "HEAD"], cwd=args.author_repo)
    sys.path.insert(0, str(args.author_repo.resolve()))
    from parcae_lm.attention_backends import flash_attn
    from parcae_lm.models.parcae.config import ParcaeConfig as AuthorConfig
    from receval.models.parcae import ModelingParcae

    attention_calls = 0

    def shared_attention(q, k, v, causal=False, window_size=(-1, -1)):
        nonlocal attention_calls
        attention_calls += 1
        assert causal and window_size == (-1, -1) and q.shape == k.shape == v.shape
        batch, length, heads, width = q.shape
        tables = torch.arange(batch, device=q.device).repeat_interleave(length)[:, None]
        lengths = torch.arange(1, length + 1, device=q.device).repeat(batch)
        return torch_paged_attention(q.reshape(-1, heads, width), k, v, tables, lengths).reshape(
            q.shape
        )

    torch.set_num_threads(2)
    torch.manual_seed(41)
    config = ParcaeConfig(
        n_embd=32,
        intermediate_size=64,
        num_attention_heads=4,
        num_key_value_heads=4,
        n_layers_in_prelude=2,
        n_layers_in_recurrent_block=2,
        n_layers_in_coda=2,
        mean_recurrence=3,
        block_size=64,
        vocab_size=1024,
    )
    native = ParcaeForCausalLM(config).eval()
    values = json.loads(args.official_config.read_text())
    values.pop("_class_name")
    values.pop("rope_settings")  # The pinned author's default is the same 50,000 base.
    for key, value in config.to_dict().items():
        if key not in ("model_type", "_class_name", "rope_base", "eos_token_id"):
            values[key] = value
    values.update(
        padded_vocab_size=config.vocab_size,
        recurrent_embedding_dimension=config.n_embd,
        recurrent_intermediation_embedding_dimension=config.intermediate_size,
    )
    original = ModelingParcae(AuthorConfig(**values)).eval()
    original.load_state_dict(native.state_dict(), strict=True)
    assert original.emb_scale == original.config.init.logit_scale == 1
    rows = []
    for dtype, atol, rtol in [(torch.float32, 3e-6, 3e-5), (torch.bfloat16, 0.03, 0.02)]:
        native.to(dtype)
        original.to(dtype)
        # Both paths use BF16 weights with the originally computed FP32 RoPE table.
        original.freqs_cis = native.freqs_cis.clone()
        for count in (1, 3, 7):
            tokens = (torch.arange(count) * 263 + 257) % config.vocab_size
            torch.manual_seed(109)
            with (
                patch.object(flash_attn, "flash_attn_func", shared_attention)
                if args.shared_attention_diagnostic
                else nullcontext()
            ):
                expected = original.forward_for_generation(tokens[None])["logits"][0]
            torch.manual_seed(109)
            initial = original.initialize_state(original.transformer.wte(tokens[None]))[0]
            _, dense, _ = dense_parcae_reference(native, tokens, initial)
            cache = KVCacheManager(
                config.num_hidden_layers,
                4,
                8,
                128,
                2,
                max_loops=3,
                dtype=dtype,
                recurrent_layers=native.recurrent_kv_layers,
            )
            cache.allocate("a", count)
            boundary = cache._prepare_batch(["a"] * count, [0] * count, list(range(count)))
            torch.manual_seed(109)
            hidden = native.prelude_prepared(tokens, boundary, cache)
            torch.testing.assert_close(hidden[:, : config.n_embd], initial, rtol=0, atol=0)
            for depth in range(config.mean_recurrence):
                core = cache._prepare_batch(["a"] * count, [depth] * count, list(range(count)))
                hidden, _ = native.recurrent_prepared(hidden, core, cache)
            logits = native.coda_prepared(hidden, boundary, cache)
            native_errors = (logits - expected).abs()
            dense_errors = (dense - expected).abs()
            allowed = atol + rtol * expected.abs()
            rows.append(
                {
                    "dtype": str(dtype),
                    "tokens": count,
                    "atol": atol,
                    "rtol": rtol,
                    "native_max_abs_logit_error": native_errors.max().item(),
                    "dense_max_abs_logit_error": dense_errors.max().item(),
                    "native_close": bool(torch.all(native_errors <= allowed)),
                    "dense_close": bool(torch.all(dense_errors <= allowed)),
                    "greedy_exact": torch.equal(logits.argmax(-1), expected.argmax(-1)),
                }
            )
            cache.free("a")
    dump(
        args.out,
        {
            "scope": (
                "diagnostic author attention substituted with native CPU paged arithmetic; "
                "not unchanged author qualification"
                if args.shared_attention_diagnostic
                else "tiny author forward, copied weights and FP32 RoPE; not official checkpoint"
            ),
            "substituted_attention_calls": attention_calls,
            "source_revision": revision,
            "generation_source_sha256": hashlib.sha256(
                (args.author_repo / "receval/models/parcae.py").read_bytes()
            ).hexdigest(),
            "cases": rows,
        },
    )
    raise SystemExit(int(any(not row["native_close"] or not row["dense_close"] for row in rows)))


if __name__ == "__main__":
    main()
