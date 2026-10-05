"""Compare pinned FlashLoop's dense control with native teacher-forced Ouro."""

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

import torch

from experiments.loopkv.flashloop_adapter import adapt_native_ouro
from vllm_rlt.core.kv_cache_manager import KVCacheManager
from vllm_rlt.models import OuroConfig, OuroForCausalLM


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flashloop-repo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    manifest_bytes = (args.flashloop_repo / "SOURCE_MANIFEST.json").read_bytes()
    assert hashlib.sha256(manifest_bytes).hexdigest() == (
        "b86b38f8c8275f0bd6bc4d46793af9fc2a68e5c92fe66f4711a46166f3752dd6"
    )
    for name, digest in json.loads(manifest_bytes).items():
        assert hashlib.sha256((args.flashloop_repo / name).read_bytes()).hexdigest() == digest
    sys.path.insert(0, str(args.flashloop_repo.resolve()))
    from flashloop import FlashLoopConfig, FlashLoopEngine

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
        max_position_embeddings=128,
        layer_types=["full_attention"] * 24,
    )
    reference = OuroForCausalLM(config)
    rows = []
    for dtype, atol, rtol in [
        (torch.float32, 3e-6, 3e-5),
        (torch.bfloat16, 0.03, 0.02),
    ]:
        reference.to(dtype)
        for count in (1, 7, 65):
            model = adapt_native_ouro(copy.deepcopy(reference))
            engine = FlashLoopEngine(
                model,
                config=FlashLoopConfig(
                    enable_token_sparse_prefill=False,
                    enable_sparse_decode=False,
                    quantize_cross_loop_kv=False,
                ),
            )
            cache = KVCacheManager(24, 1, 128, 512, 16, max_loops=4, dtype=dtype)
            cache.allocate("a", count + 2)
            tokens = torch.arange(count + 2) % 127 + 1
            max_logit = max_kv = 0.0
            logit_failures = kv_failures = 0
            greedy_exact = True
            for positions in (list(range(count)), [count], [count + 1]):
                hidden = reference.prelude(tokens[positions])
                for depth in range(4):
                    hidden, _ = reference.recurrent(
                        hidden,
                        ["a"] * len(positions),
                        [depth] * len(positions),
                        positions,
                        cache,
                    )
                expected = reference.coda(hidden)[-1:]
                output = (
                    engine.prefill(tokens[:count][None])
                    if positions[0] == 0
                    else engine.decode(tokens[positions][None])
                )
                observed = output.logits
                logit_failures += int(
                    ((observed - expected).abs() > atol + rtol * expected.abs()).sum()
                )
                greedy_exact &= torch.equal(observed.argmax(-1), expected.argmax(-1))
                max_logit = max(max_logit, float((observed - expected).abs().max()))
                for layer in range(24):
                    for depth in range(4):
                        length = positions[-1] + 1
                        actual = engine.state.cache.gather(layer, depth, torch.arange(length))
                        wanted = cache.read(layer, "a", depth, length)
                        for x, y in zip(actual, wanted):
                            x = x[0].transpose(0, 1)
                            kv_failures += int(((x - y).abs() > atol + rtol * y.abs()).sum())
                            max_kv = max(max_kv, float((x - y).abs().max()))
            rows.append(
                {
                    "dtype": str(dtype),
                    "prompt": count,
                    "decode": 2,
                    "max_logit": max_logit,
                    "max_kv": max_kv,
                    "logit_failed_elements": logit_failures,
                    "kv_failed_elements": kv_failures,
                    "greedy_exact": greedy_exact,
                    "atol": atol,
                    "rtol": rtol,
                }
            )
            cache.free("a")
            print(rows[-1], flush=True)
    report = {
        "scope": (
            "tiny adapted FlashLoop dense control; no packed codec, official model or speed qualification"
        ),
        "source_pin": "aeaadee10d75b7aa8c4a42a809f15c75d6d3bdc6",
        "cases": rows,
        "bridge_sha256": hashlib.sha256(
            Path(__file__).with_name("flashloop_adapter.py").read_bytes()
        ).hexdigest(),
    }
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    raise SystemExit(
        int(any(row["logit_failed_elements"] or row["kv_failed_elements"] for row in rows))
    )


if __name__ == "__main__":
    main()
