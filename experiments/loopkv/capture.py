"""Capture an official checkpoint's baseline through the existing engine.

Detailed tracing is diagnostic; it is excluded from performance claims.
Prompt files contain a JSON list of token-ID lists, with a recorded SHA256.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import Counter
from dataclasses import asdict
from importlib import metadata
from pathlib import Path

import torch

from experiments.loopkv.capacity import Geometry, account
from vllm_rlt import CacheConfig, ExecutionConfig, ExitConfig, SamplingParams, SchedulerConfig
from vllm_rlt.engine.llm_engine import LLMEngine
from vllm_rlt.models import AutoModelForCausalLM, OuroConfig, OuroForCausalLM
from vllm_rlt.request import Stage


def dump(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--requests", type=int)
    parser.add_argument("--num-blocks", type=int, required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--policy", choices=("ouro", "ouro_delayed"), default="ouro")
    parser.add_argument("--alias", action="store_true")
    parser.add_argument("--checkpoint-reader", choices=("native", "torch"), default="native")
    parser.add_argument("--async-scheduling", action="store_true")
    parser.add_argument("--graphs", action="store_true")
    args = parser.parse_args()
    manifest = json.loads((args.model / "verified-manifest.json").read_text())
    prompts = json.loads(args.prompts.read_text())
    if args.requests is not None:
        if args.requests < 1:
            raise ValueError("requests must be positive")
        prompts = [prompts[i % len(prompts)] for i in range(args.requests)]
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(17)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cache_config = CacheConfig(args.num_blocks, 16, alias_last_exited=args.alias)
    scheduler = SchedulerConfig(
        max_num_seqs=args.batch, max_num_batched_tokens=max(128, args.batch)
    )
    execution = ExecutionConfig(
        async_scheduling=args.async_scheduling,
        static_buffers=args.graphs,
        cuda_graphs=args.graphs,
    )
    source = Path(__file__).resolve().parents[2]
    tracked_diff = subprocess.check_output(["git", "diff", "HEAD"], cwd=source)
    packages = {
        d.metadata["Name"]: d.version
        for d in metadata.distributions()
        if d.metadata["Name"].lower() in ("torch", "triton", "safetensors", "transformers")
    }
    dump(
        args.out / "manifest.json",
        {
            "checkpoint": manifest,
            "checkpoint_reader": args.checkpoint_reader,
            "git_sha": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=source, text=True
            ).strip(),
            "tracked_patch_sha256": hashlib.sha256(tracked_diff).hexdigest(),
            "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "prompts_sha256": hashlib.sha256(args.prompts.read_bytes()).hexdigest(),
            "python": sys.executable,
            "pid": os.getpid(),
            "packages": packages,
            "cache": asdict(cache_config),
            "scheduler": asdict(scheduler),
            "execution": asdict(execution),
            "policy": args.policy,
            "sampling": {
                "max_tokens": args.max_tokens,
                "exit_threshold": args.threshold,
                "seed": 17,
                "ignore_eos": True,
            },
            "request_count": len(prompts),
            "scope": "diagnostic_engine_capture_not_performance",
            "gpu": subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=index,uuid,name,memory.total,driver_version",
                    "--format=csv",
                ],
                text=True,
            ),
        },
    )
    if args.checkpoint_reader == "native":
        model = AutoModelForCausalLM.from_pretrained(
            str(args.model), device="cuda", dtype=torch.bfloat16
        )
    else:
        from experiments.loopkv.checkpoint import load_weights

        config_data = json.loads((args.model / "config.json").read_text())
        if config_data["model_type"] != "ouro":
            raise ValueError("the standalone reader is qualified for Ouro only")
        with torch.device("meta"):
            model = OuroForCausalLM(OuroConfig.from_dict(config_data))
        load_weights(model, args.model, torch.device("cuda"), torch.bfloat16)
        model.model.rotary_emb.inv_freq = model.model.rotary_emb.frequencies(device="cuda")
        model.eval()
    config = model.config
    shape = Geometry(
        config.num_hidden_layers, config.num_key_value_heads, config.head_dim, config.total_ut_steps
    )
    max_request_blocks = max(
        ((len(prompt) + args.max_tokens + 15) // 16) * shape.loops for prompt in prompts
    )
    if max_request_blocks > args.num_blocks:
        raise ValueError("one complete worst-future request exceeds the KV pool")
    engine = LLMEngine(
        model,
        cache_config=cache_config,
        scheduler_config=scheduler,
        attention_backend="triton",
        execution_config=execution,
        exit_config=ExitConfig(args.policy),
    )
    params = SamplingParams(
        max_tokens=args.max_tokens, exit_threshold=args.threshold, seed=17, ignore_eos=True
    )
    for index, prompt in enumerate(prompts):
        engine.add_request(str(index), prompt, params)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    completed, steps = {}, []
    start = time.perf_counter()
    while engine.has_unfinished_requests():
        for output in engine.step():
            if output.finished:
                completed[output.request_id] = asdict(output)
        batch = engine.last_schedule
        if batch is not None:
            steps.append(
                {
                    "stage": batch.stage.value,
                    "rows": batch.num_tokens,
                    "submitted_rows": engine.model_runner.last_submitted_size,
                    "requests": [i.request.request_id for i in batch.items],
                    "resident_requests": len(engine.cache_manager._allocations),
                }
            )
    engine.model_runner.synchronize()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    # The first output comes from prefill; its exit depth is not a decode KV position.
    exits = [r for output in completed.values() for r in output["exit_depths"][1:]]
    output_count = sum(len(o["token_ids"]) for o in completed.values())
    ledgers = [
        account(shape, len(o["prompt_token_ids"]), o["exit_depths"][1:]) for o in completed.values()
    ]
    summary = {
        "scope": "diagnostic_engine_capture_not_performance",
        "seconds": elapsed,
        "completed_requests": len(completed),
        "output_tokens": output_count,
        "decode_exit_histogram": dict(Counter(exits)),
        "cached_decode_positions": len(exits),
        "executed_decode_loops": sum(exits),
        "recurrent_batch_histogram": dict(
            Counter(s["rows"] for s in steps if s["stage"] == Stage.RECURRENT.value)
        ),
        "promotion_payload_bytes_from_trace": sum(
            r["avoidable_promotion_payload_bytes"] for r in ledgers
        ),
        "kv_arena_reserved_bytes": engine.cache_manager.num_blocks
        * engine.cache_manager.bytes_per_block,
        "runtime_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "runtime_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "free_blocks_after_drain": engine.cache_manager.num_free_blocks,
        "geometry": asdict(shape),
        "finalization_exposed_ms": None,
    }
    dump(args.out / "run_summary.json", summary)
    for filename, values in (("requests.jsonl", completed.values()), ("steps.jsonl", steps)):
        (args.out / filename).write_text("".join(json.dumps(row) + "\n" for row in values))
    engine.close()
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
