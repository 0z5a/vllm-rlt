"""Compare native fixed-R8 cache execution with the CPU-qualified dense adapter."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import torch

from loopquant.adapters.loopformer import LoopFormerAdapter
from loopquant.checkpoint import load_weights
from vllm_rlt.core.kv_cache_manager import KVCacheManager
from vllm_rlt.models.loopformer import LoopFormerConfig, LoopFormerForCausalLM


def comparison(actual: torch.Tensor, expected: torch.Tensor) -> dict:
    actual, expected = actual.float(), expected.float()
    return dict(
        within_budget=bool(torch.allclose(actual, expected, atol=0.03, rtol=0.02)),
        exact=bool(torch.equal(actual, expected)),
        max_abs=float((actual - expected).abs().max()),
    )


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--requests", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tiny", action="store_true")
    args = parser.parse_args()
    started = time.time()
    torch.manual_seed(17)
    torch.backends.cuda.matmul.allow_tf32 = False
    if args.tiny:
        from tests.loopquant_int4_checks import int4_reference_model

        model = int4_reference_model("loopformer").to(dtype=torch.bfloat16)
        requests = [
            dict(request_id=f"tiny-{n}", token_ids=list(range(3, 3 + n)))
            for n in (5, 7)
        ]
        device, backend = torch.device("cpu"), "torch"
    else:
        assert args.model is not None and args.requests is not None
        manifest = json.loads((args.model / "verified-manifest.json").read_text())
        assert manifest["independently_verified"]
        assert manifest["revision"] == "2b4fbaaf4e2510353ef7cfe07a07c671b5226739"
        for row in manifest["files"]:
            with (args.model / row["name"]).open("rb") as stream:
                assert (
                    hashlib.file_digest(stream, "sha256").hexdigest() == row["sha256"]
                )
        config = LoopFormerConfig.from_dict(
            json.loads((args.model / "config.json").read_text())
        )
        with torch.device("meta"):
            model = LoopFormerForCausalLM(config)
        device, backend = torch.device("cuda"), "triton"
        assert torch.cuda.get_device_capability() == (9, 0)
        load_weights(model, args.model, device, torch.bfloat16)
        assert hashlib.sha256(args.requests.read_bytes()).hexdigest() == (
            "b06bfc5c7b1537d2071a381eb4528e963a6f46a603e8ac0a897cba59ad05b693"
        )
        requests = json.loads(args.requests.read_text())
        assert len(requests) == 32
    model.eval().requires_grad_(False)
    config = model.config
    adapter = LoopFormerAdapter(model)
    length = max(len(row["token_ids"]) for row in requests)
    cache = KVCacheManager(
        config.num_hidden_layers,
        config.num_key_value_heads,
        config.head_dim,
        8 * math.ceil(length / 16),
        16,
        8,
        device=device,
        dtype=torch.bfloat16,
        backend=backend,
    )
    records = []
    with args.output.with_suffix(".jsonl").open("x") as log:
        for request in requests:
            tokens = torch.tensor(request["token_ids"], device=device, dtype=torch.long)
            reference = adapter(
                tokens[None], torch.ones_like(tokens[None], dtype=torch.bool), 8
            )
            for incremental in (False, True):
                assert cache.num_used_blocks == 0 and cache.allocate("g0", len(tokens))
                chunks = (
                    [[p] for p in range(len(tokens))]
                    if incremental
                    else [list(range(len(tokens)))]
                )
                state_checks, logit_checks, argmax, clocks, prefixes = (
                    [],
                    [],
                    [],
                    [],
                    [],
                )
                for positions in chunks:
                    allocation = cache._get_allocation("g0")
                    addresses = [
                        (allocation.block_tables[d][p // 16], p % 16)
                        for d in range(8)
                        for p in range(positions[0])
                    ]
                    blocks = torch.tensor(
                        [b for b, _ in addresses], device=device, dtype=torch.long
                    )
                    offsets = torch.tensor(
                        [p for _, p in addresses], device=device, dtype=torch.long
                    )
                    before = [
                        storage[blocks, :, offsets].clone()
                        for storage in (cache.key_cache, cache.value_cache)
                    ]
                    hidden = model.prelude(tokens[positions])
                    for depth in range(8):
                        hidden, _ = model.recurrent(
                            hidden,
                            ["g0"] * len(positions),
                            [depth] * len(positions),
                            positions,
                            cache,
                            compute_gate=False,
                        )
                        state_checks.append(
                            comparison(hidden, reference.states[depth][0, positions])
                        )
                        clocks.append(bool((hidden[:, -1] == (depth + 1) / 8).all()))
                    actual, expected = (
                        model.coda(hidden),
                        reference.logits[0, positions],
                    )
                    logit_checks.append(comparison(actual, expected))
                    argmax.append(
                        bool(torch.equal(actual.argmax(-1), expected.argmax(-1)))
                    )
                    prefixes.append(
                        all(
                            torch.equal(old, storage[blocks, :, offsets])
                            for old, storage in zip(
                                before,
                                (cache.key_cache, cache.value_cache),
                                strict=True,
                            )
                        )
                    )
                    for position in positions:
                        cache.finalize_token("g0", position, 7)
                cache.free("g0")
                assert cache.num_used_blocks == 0
                row = dict(
                    request_id=request["request_id"],
                    tokens=len(tokens),
                    incremental=incremental,
                    states=state_checks,
                    logits=logit_checks,
                    argmax_exact=all(argmax),
                    clocks_exact=all(clocks),
                    prior_kv_exact=all(prefixes),
                    kv_after_drain=0,
                )
                records.append(row)
                log.write(json.dumps(row) + "\n")
                log.flush()
                print(
                    json.dumps(
                        {k: v for k, v in row.items() if k not in ("states", "logits")}
                    ),
                    flush=True,
                )
    passed = all(
        r["clocks_exact"]
        and r["prior_kv_exact"]
        and r["argmax_exact"]
        and all(c["within_budget"] for c in r["states"] + r["logits"])
        for r in records
    )
    result = dict(
        status="pass" if passed else "inconclusive",
        records=records,
        full_checkpoint=not args.tiny,
        atol=0.03,
        rtol=0.02,
        loops=8,
        backend=backend,
        torch_version=torch.__version__,
        start_unix=started,
        end_unix=time.time(),
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope="native BF16 vs CPU-qualified dense SDPA adapter; no quantized quality or timing",
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
