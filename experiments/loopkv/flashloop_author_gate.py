"""Qualify the weight API adapter against pinned HF Ouro inside the author engine."""

import argparse
import copy
import hashlib
import importlib
import json
import sys
from pathlib import Path
from types import ModuleType

import torch

from experiments.loopkv.flashloop_adapter import adapt_native_ouro
from vllm_rlt.models import OuroConfig, OuroForCausalLM


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flashloop-repo", type=Path, required=True)
    parser.add_argument("--author-model-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    manifest_bytes = (args.flashloop_repo / "SOURCE_MANIFEST.json").read_bytes()
    assert hashlib.sha256(manifest_bytes).hexdigest() == (
        "b86b38f8c8275f0bd6bc4d46793af9fc2a68e5c92fe66f4711a46166f3752dd6"
    )
    for name, digest in json.loads(manifest_bytes).items():
        assert hashlib.sha256((args.flashloop_repo / name).read_bytes()).hexdigest() == digest
    for name, digest in {
        "modeling_ouro.py": "c5c68fbb368ce2909c257ae2afc50719be8c91539333d3295e19312c4316f413",
        "configuration_ouro.py": "950443e32929047aa08d02abad2e1888bc1914b3db988d3d675f70787f65dafb",
    }.items():
        assert hashlib.sha256((args.author_model_dir / name).read_bytes()).hexdigest() == digest
    sys.path.insert(0, str(args.flashloop_repo.resolve()))
    from flashloop import FlashLoopConfig, FlashLoopEngine

    package = ModuleType("loopkv_author_ouro")
    package.__path__ = [str(args.author_model_dir.resolve())]
    sys.modules[package.__name__] = package
    author = importlib.import_module(package.__name__ + ".modeling_ouro")
    torch.set_num_threads(2)
    torch.manual_seed(127)
    config = OuroConfig(
        vocab_size=128,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=24,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=128,
        max_position_embeddings=256,
        layer_types=["full_attention"] * 24,
    )
    native = OuroForCausalLM(config).to(torch.bfloat16)
    original = (
        author.OuroForCausalLM(author.OuroConfig(**config.to_dict())).eval().to(torch.bfloat16)
    )
    original.load_state_dict(native.state_dict(), strict=True)
    original.model.rotary_emb.inv_freq = native.model.rotary_emb.inv_freq.clone()
    rows = []
    for name, settings in [
        (
            "dense",
            dict(
                enable_token_sparse_prefill=False,
                enable_sparse_decode=False,
                quantize_cross_loop_kv=False,
            ),
        ),
        ("sparse-bf16", dict(quantize_cross_loop_kv=False)),
        ("packed-int4", {}),
    ]:
        for count in (1, 65, 129):
            expected = FlashLoopEngine(copy.deepcopy(original), config=FlashLoopConfig(**settings))
            actual = FlashLoopEngine(
                adapt_native_ouro(copy.deepcopy(native)),
                config=FlashLoopConfig(**settings),
            )
            tokens = torch.arange(count + 2) % 127 + 1
            maximum = 0.0
            for step in range(3):
                left = (
                    expected.prefill(tokens[:count][None])
                    if step == 0
                    else expected.decode(tokens[count + step - 1 : count + step][None])
                )
                right = (
                    actual.prefill(tokens[:count][None])
                    if step == 0
                    else actual.decode(tokens[count + step - 1 : count + step][None])
                )
                torch.testing.assert_close(left.logits, right.logits, atol=0, rtol=0)
                maximum = max(maximum, float((left.logits - right.logits).abs().max()))
                for layer in range(24):
                    for depth in range(4):
                        indices = torch.arange(count + step)
                        for a, b in zip(
                            expected.state.cache.gather(layer, depth, indices),
                            actual.state.cache.gather(layer, depth, indices),
                        ):
                            torch.testing.assert_close(a, b, atol=0, rtol=0)
            assert (
                expected.state.cache.physical_layout_audit()
                == actual.state.cache.physical_layout_audit()
            )
            rows.append(
                {
                    "mode": name,
                    "prompt": count,
                    "decode": 2,
                    "logits_and_all_kv_bitwise_exact": True,
                    "max_logit": maximum,
                    "physical_layout": actual.state.cache.physical_layout_audit(),
                }
            )
            print(rows[-1], flush=True)
    result = {
        "scope": (
            "tiny CPU BF16 weight-interface qualification against pinned HF Ouro in unchanged "
            "FlashLoop; no CUDA, official weight or speed claim"
        ),
        "original_rope": "same original FP32 frequencies in both paths",
        "author_modeling_sha256": hashlib.sha256(
            (args.author_model_dir / "modeling_ouro.py").read_bytes()
        ).hexdigest(),
        "flashloop_source_pin": "aeaadee10d75b7aa8c4a42a809f15c75d6d3bdc6",
        "bridge_sha256": hashlib.sha256(
            Path(__file__).with_name("flashloop_adapter.py").read_bytes()
        ).hexdigest(),
        "cases": rows,
    }
    args.out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
