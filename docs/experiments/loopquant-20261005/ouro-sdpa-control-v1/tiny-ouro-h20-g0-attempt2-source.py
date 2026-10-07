"""Audit native Ouro BF16 state, logits and cache against the dense SDPA adapter."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import torch

from loopquant.adapters.ouro import OuroAdapter
from loopquant.checkpoint import load_weights
from vllm_rlt.core.kv_cache_manager import KVCacheManager
from vllm_rlt.models.ouro import OuroConfig, OuroForCausalLM


def comparison(actual: torch.Tensor, expected: torch.Tensor) -> dict:
    actual, expected = actual.float(), expected.float()
    return dict(
        within_budget=bool(torch.allclose(actual, expected, atol=0.02, rtol=0.02)),
        exact=bool(torch.equal(actual, expected)),
        max_abs=float((actual - expected).abs().max()),
    )


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--requests", type=Path)
    parser.add_argument("--model-manifest", type=Path)
    parser.add_argument("--backend", choices=["torch", "triton"], default="triton")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tiny", action="store_true")
    args = parser.parse_args()
    started = time.time()
    torch.manual_seed(17)
    torch.backends.cuda.matmul.allow_tf32 = False
    if args.tiny:
        from tests.helpers import tiny_ouro_config

        model = OuroForCausalLM(
            tiny_ouro_config(hidden_size=64, head_dim=32, intermediate_size=128)
        ).to(dtype=torch.bfloat16)
        requests = [
            dict(request_id=f"tiny-{n}", token_ids=list(range(3, 3 + n)))
            for n in (5, 7)
        ]
        device, backend = torch.device("cpu"), "torch"
        model.model.rotary_emb.inv_freq = model.model.rotary_emb.frequencies(
            device=device
        )
    else:
        assert args.model is not None and args.requests is not None
        assert args.model_manifest is not None
        manifest = json.loads(args.model_manifest.read_text())
        assert manifest["status"] == "PREPARATION_PASS"
        assert manifest["model_directory"] == str(args.model)
        weight = next(
            row for row in manifest["inputs"] if row["name"] == "model.safetensors"
        )
        assert (
            weight["sha256"]
            == "58872a72616c736595b8b7662079c5b12c5a162ec16eae94f21c348dfa9885af"
        )
        for row in manifest["inputs"]:
            path = args.model / row["name"]
            assert path.stat().st_size == row["size"]
            with path.open("rb") as stream:
                assert (
                    hashlib.file_digest(stream, "sha256").hexdigest() == row["sha256"]
                )
        config = OuroConfig.from_dict(
            json.loads((args.model / "config.json").read_text())
        )
        with torch.device("meta"):
            model = OuroForCausalLM(config)
        device, backend = torch.device("cuda"), args.backend
        assert torch.cuda.get_device_capability() == (9, 0)
        load_weights(model, args.model, device, torch.bfloat16)
        model.model.rotary_emb.inv_freq = model.model.rotary_emb.frequencies(
            device=device
        )
        assert hashlib.sha256(args.requests.read_bytes()).hexdigest() == (
            "9d370c5b159841199657563264656cb82cb524812cd987919f7c1dfc4de43009"
        )
        requests = json.loads(args.requests.read_text())
        assert len(requests) == 32
    model.eval().requires_grad_(False)
    config = model.config
    adapter = OuroAdapter(model, attention_backend="sdpa")
    length = max(len(row["token_ids"]) for row in requests)
    cache = KVCacheManager(
        config.num_hidden_layers,
        config.num_key_value_heads,
        config.head_dim,
        4 * math.ceil(length / 16),
        16,
        4,
        device=device,
        dtype=torch.bfloat16,
        backend=backend,
    )
    records = []
    with args.output.with_suffix(".jsonl").open("x") as log:
        for request in requests:
            tokens = torch.tensor(request["token_ids"], device=device, dtype=torch.long)
            reference = adapter(
                tokens[None], torch.ones_like(tokens[None], dtype=torch.bool), 4
            )
            for incremental in (False, True):
                assert cache.num_used_blocks == 0 and cache.allocate("g0", len(tokens))
                chunks = (
                    [[p] for p in range(len(tokens))]
                    if incremental
                    else [list(range(len(tokens)))]
                )
                state_checks, logit_checks, argmax, prefixes = (
                    [],
                    [],
                    [],
                    [],
                )
                for positions in chunks:
                    allocation = cache._get_allocation("g0")
                    addresses = [
                        (allocation.block_tables[d][p // 16], p % 16)
                        for d in range(4)
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
                    for depth in range(4):
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
                        actual = model.coda(hidden)
                        expected = model.coda(reference.states[depth][0, positions])
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
                        cache.finalize_token("g0", position, 3)
                cache.free("g0")
                assert cache.num_used_blocks == 0
                row = dict(
                    request_id=request["request_id"],
                    tokens=len(tokens),
                    incremental=incremental,
                    states=state_checks,
                    logits=logit_checks,
                    argmax_exact=all(argmax),
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
        r["prior_kv_exact"]
        and r["argmax_exact"]
        and all(c["within_budget"] for c in r["states"] + r["logits"])
        for r in records
    )
    result = dict(
        status="pass" if passed else "inconclusive",
        records=records,
        full_checkpoint=not args.tiny,
        atol=0.02,
        rtol=0.02,
        loops=4,
        model_revision="574fa66cb8bf5abdc979642d01cf2b79b16bfab1",
        backend=backend,
        torch_version=torch.__version__,
        start_unix=started,
        end_unix=time.time(),
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope="Native BF16 vs dense SDPA adapter, R1/2/3/4 state and readout, full/incremental cache. No PTQ/QAT search, quality or timing; separate from UNKNOWN old5090 INT4 job.",
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
