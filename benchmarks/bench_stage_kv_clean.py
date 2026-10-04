"""Run one unchanged-parent or candidate arm in a fresh Python process."""

import argparse
import hashlib
import json
import runpy
from pathlib import Path

import torch

import vllm_rlt
from vllm_rlt import CacheConfig, SchedulerConfig
from vllm_rlt.engine.llm_engine import LLMEngine
from vllm_rlt.models.huginn import HuginnConfig, HuginnForCausalLM


def run(args) -> None:
    source = Path(vllm_rlt.__file__).resolve().parents[1]
    assert source == args.source.resolve(), source
    serve = runpy.run_path(str(Path(__file__).with_name("bench_stage_kv.py")))["serve"]
    torch.set_num_threads(1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        for loops, concurrency in ((3, 1), (3, 16), (3, 64), (32, 1), (32, 16)):
            torch.manual_seed(args.seed)
            model = HuginnForCausalLM(
                HuginnConfig(
                    n_embd=16,
                    n_heads=2,
                    n_layers=8,
                    n_layers_in_prelude=2,
                    n_layers_in_recurrent_block=4,
                    n_layers_in_coda=2,
                    intermediate_size=32,
                    mean_recurrence=loops,
                    block_size=64,
                    vocab_size=13,
                    padded_vocab_size=13,
                    bos_token_id=0,
                    eos_token_id=1,
                    pad_token_id=0,
                )
            ).eval()
            options = {
                "num_blocks": loops * concurrency * 5,
                "block_size": 2,
                "incremental_allocation": True,
            }
            if args.arm == "stage":
                options["stage_aware"] = True
            engine = LLMEngine(
                model,
                cache_config=CacheConfig(**options),
                scheduler_config=SchedulerConfig(
                    max_num_seqs=concurrency, max_num_batched_tokens=64, prefill_chunk_size=2
                ),
            )
            expected = json.loads(
                (args.expected / f"outputs-s{args.seed}-r{loops}-c{concurrency}.json").read_text()
            )
            _, warm, _ = serve(engine, loops, concurrency, args.seed)
            assert warm == expected
            elapsed, actual, metrics = serve(engine, loops, concurrency, args.seed)
            assert actual == expected
            cache = engine.cache_manager
            storage = cache.key_cache.numel() + cache.value_cache.numel()
            if args.arm == "stage":
                storage += cache.boundary_keys.numel() + cache.boundary_values.numel()
            row = {
                "seed": args.seed,
                "loops": loops,
                "concurrency": concurrency,
                "arm": args.arm,
                "order": args.order,
                "source_root": str(source),
                "wall_seconds": elapsed,
                "response_tokens": 8 * concurrency,
                "tokens_per_second": 8 * concurrency / elapsed,
                "requests": 2 * concurrency,
                "kv_tensor_bytes": storage * next(model.parameters()).element_size(),
                "outputs_sha256": hashlib.sha256(
                    json.dumps(actual, sort_keys=True).encode()
                ).hexdigest(),
                "exact_outputs": actual == expected,
                "used_blocks_after": cache.num_used_blocks,
                "torch": torch.__version__,
                "device": "cpu",
                "threads": 1,
                **metrics,
            }
            stream.write(json.dumps(row, sort_keys=True) + "\n")
            stream.flush()
            engine.close()
            print(
                f"seed={args.seed} order={args.order} {args.arm} R={loops} C={concurrency} exact",
                flush=True,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--order", type=int, required=True)
    parser.add_argument("--arm", choices=("parent", "stage"), required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--expected", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
