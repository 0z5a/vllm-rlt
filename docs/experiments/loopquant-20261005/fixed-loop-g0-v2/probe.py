"""Audit fixed-loop BF16 state, logits and cache against dense eager and SDPA references."""

import argparse
from contextlib import nullcontext
import hashlib
import json
import math
from pathlib import Path
import time

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from loopquant.adapters.ouro import OuroAdapter
from loopquant.adapters.nanbeige import NanbeigeAdapter
from loopquant.checkpoint import load_weights
from vllm_rlt.core.kv_cache_manager import KVCacheManager
from vllm_rlt.models.ouro import OuroConfig, OuroForCausalLM
from vllm_rlt.models.nanbeige import NanbeigeConfig, NanbeigeForCausalLM


def comparison(actual: torch.Tensor, expected: torch.Tensor) -> dict:
    actual, expected = actual.float(), expected.float()
    return dict(
        within_budget=bool(torch.allclose(actual, expected, atol=0.02, rtol=0.02)),
        exact=bool(torch.equal(actual, expected)),
        max_abs=float((actual - expected).abs().max()),
    )


@torch.inference_mode()
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--requests", type=Path)
    parser.add_argument("--model-plan", type=Path)
    parser.add_argument("--family", choices=["ouro", "nanbeige"], required=True)
    parser.add_argument("--backend", choices=["torch", "triton"], default="triton")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument(
        "--reference-sdpa", choices=["eager", "auto", "math"], default="eager"
    )
    args = parser.parse_args(argv)
    started = time.time()
    torch.manual_seed(17)
    torch.backends.cuda.matmul.allow_tf32 = False
    if args.family == "ouro":
        model_type, config_type, adapter_type = OuroForCausalLM, OuroConfig, OuroAdapter
        repository, revision = (
            "ByteDance/Ouro-2.6B",
            "1ed04250da1a9936042725d302e81c8fa2ab5abd",
        )
    else:
        model_type, config_type, adapter_type = (
            NanbeigeForCausalLM,
            NanbeigeConfig,
            NanbeigeAdapter,
        )
        repository, revision = (
            "Nanbeige/Nanbeige4.2-3B",
            "b82e54bd609793562a75cbf9337970a93369eab5",
        )
    if args.tiny:
        from tests.helpers import tiny_ouro_config, tiny_nanbeige_config

        config = (
            tiny_ouro_config(hidden_size=64, head_dim=32, intermediate_size=128)
            if args.family == "ouro"
            else tiny_nanbeige_config()
        )
        model = model_type(config).to(dtype=torch.bfloat16)
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
        assert args.model_plan is not None
        plan = json.loads(args.model_plan.read_text())
        manifest = json.loads((args.model / "verified-manifest.json").read_text())
        assert plan["model"] == manifest["model"] == repository
        assert plan["revision"] == manifest["revision"] == revision
        assert manifest["independently_verified"]
        assert manifest["files"] == plan["files"]
        for row in plan["files"]:
            path = args.model / row["name"]
            assert path.stat().st_size == row["bytes"]
            with path.open("rb") as stream:
                assert (
                    hashlib.file_digest(stream, "sha256").hexdigest() == row["sha256"]
                )
        config = config_type.from_dict(
            json.loads((args.model / "config.json").read_text())
        )
        with torch.device("meta"):
            model = model_type(config)
        device, backend = torch.device("cuda"), args.backend
        assert torch.cuda.get_device_capability() == (9, 0)
        load_weights(model, args.model, device, torch.bfloat16)
        model.model.rotary_emb.inv_freq = model.model.rotary_emb.frequencies(
            device=device
        )
        assert (
            hashlib.sha256(args.requests.read_bytes()).hexdigest()
            == (plan["requests_sha256"])
        )
        requests = json.loads(args.requests.read_text())
        assert len(requests) == 32
    model.eval().requires_grad_(False)
    config = model.config
    loops = config.total_ut_steps
    assert loops == (4 if args.family == "ouro" else 2)
    adapter = adapter_type(
        model, attention_backend="eager" if args.reference_sdpa == "eager" else "sdpa"
    )
    length = max(len(row["token_ids"]) for row in requests)
    cache = KVCacheManager(
        config.num_hidden_layers,
        config.num_key_value_heads,
        config.head_dim,
        loops * math.ceil(length / 16),
        16,
        loops,
        device=device,
        dtype=torch.bfloat16,
        backend=backend,
    )
    records = []
    with args.output.with_suffix(".jsonl").open("x") as log:
        for request in requests:
            tokens = torch.tensor(request["token_ids"], device=device, dtype=torch.long)
            context = (
                sdpa_kernel(SDPBackend.MATH)
                if args.reference_sdpa == "math"
                else nullcontext()
            )
            with context:
                reference = adapter(
                    tokens[None], torch.ones_like(tokens[None], dtype=torch.bool), loops
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
                        for d in range(loops)
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
                    for depth in range(loops):
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
                        cache.finalize_token("g0", position, loops - 1)
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
        loops=loops,
        family=args.family,
        repository=repository,
        model_revision=revision,
        model_plan_sha256=hashlib.sha256(args.model_plan.read_bytes()).hexdigest()
        if args.model_plan
        else None,
        backend=backend,
        reference_sdpa=args.reference_sdpa,
        torch_version=torch.__version__,
        start_unix=started,
        end_unix=time.time(),
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope="Native BF16 vs dense adapter: official CPU qualification applies to eager; auto/math SDPA are separate internal controls, all registered loop states and readouts, full/incremental cache. Not original-author CUDA attention qualification. No PTQ/QAT search, quality or timing; separate from UNKNOWN old5090 INT4 job.",
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
