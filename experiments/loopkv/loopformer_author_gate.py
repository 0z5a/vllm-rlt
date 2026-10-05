"""Compare the pinned author's dense forward with the independent tiny oracle."""

import argparse
import hashlib
import importlib.util
import sys
from pathlib import Path

import torch

from experiments.loopkv.capture import dump
from tests.reference.loopformer import dense_loopformer_reference
from vllm_rlt.models import LoopFormerConfig, LoopFormerForCausalLM


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--author-source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    digest = hashlib.sha256(args.author_source.read_bytes()).hexdigest()
    assert digest == "a4fdd1330a88a859f5f726a15fdce5b6c37d523fd7f04b9c9765d6c58133e665"
    spec = importlib.util.spec_from_file_location("pinned_loopformer_author", args.author_source)
    author = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = author
    spec.loader.exec_module(author)
    torch.set_num_threads(4)
    torch.manual_seed(92)
    cfg = LoopFormerConfig(
        vocab_size=32,
        n_embd=16,
        n_head=2,
        n_layer=2,
        intermediate_dim=24,
        block_size=64,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    native = LoopFormerForCausalLM(cfg).eval()
    values = {k: v for k, v in cfg.to_dict().items() if k not in ("model_type", "architectures")}
    original = author.GPT(author.GPTConfig(**values)).eval()
    weights = {name.removeprefix("gpt."): tensor for name, tensor in native.state_dict().items()}
    weights["lm_head.weight"] = weights["transformer.wte.weight"]
    original.load_state_dict(weights, strict=True)
    rows = []
    for dtype, atol, rtol in ((torch.float32, 3e-6, 3e-5), (torch.bfloat16, 0.03, 0.02)):
        native.to(dtype)
        original.to(dtype)
        for count in (1, 3, 7):
            tokens = torch.arange(count) * 3 % cfg.vocab_size
            output, _ = original(tokens[None])
            _, logits, _ = dense_loopformer_reference(native, tokens)
            torch.testing.assert_close(logits, output[0], atol=atol, rtol=rtol)
            rows.append(
                {
                    "dtype": str(dtype),
                    "tokens": count,
                    "atol": atol,
                    "rtol": rtol,
                    "max_abs_logit_error": (logits.float() - output[0].float()).abs().max().item(),
                    "tokens_exact": torch.equal(logits.argmax(-1), output[0].argmax(-1)),
                }
            )
    dump(
        args.out,
        {
            "source_sha256": digest,
            "source_revision": "2b4fbaaf4e2510353ef7cfe07a07c671b5226739",
            "scope": "author GPT vs independent functional oracle on copied tiny weights",
            "cases": rows,
        },
    )


if __name__ == "__main__":
    main()
