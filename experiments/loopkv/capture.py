"""Run an official checkpoint through the existing engine.

The default detailed trace is diagnostic. --measure runs one complete warmup
and omits per-step tracing. This measures a finite engine batch, not HTTP serving.
Prompt files contain a JSON list of token-ID lists, with a recorded SHA256.
"""

import argparse
import hashlib
import importlib
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
from vllm_rlt.models import AutoModelForCausalLM
from vllm_rlt.request import Stage


def dump(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def drive(engine, prompts, params, *, trace, stop_spec=None):
    completed, steps = {}, []
    matchers, matches = {}, {}
    if stop_spec is not None:
        from experiments.loopkv.text_stops import ByteStops

        pieces = tuple(bytes.fromhex(value) for value in stop_spec["token_bytes_hex"])
        matchers = {str(i): ByteStops(pieces, stop_spec["stops"]) for i in range(len(prompts))}
    torch.cuda.synchronize()
    start = time.perf_counter()
    for index, prompt in enumerate(prompts):
        engine.add_request(str(index), prompt, params)
    while engine.has_unfinished_requests():
        for output in engine.step():
            if stop_spec is not None:
                match = matchers[output.request_id].update(output.token_ids)
                if match is not None:
                    matches[output.request_id] = match
                    if not output.finished:
                        output = engine.abort_request(output.request_id)
            if output.finished:
                completed[output.request_id] = output
        if trace:
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
    serialized = {rid: asdict(output) for rid, output in completed.items()}
    for rid, (text, offset) in matches.items():
        serialized[rid]["text_stop"] = {"text": text, "byte_offset": offset}
    return serialized, steps, elapsed


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
    parser.add_argument("--min-loops", type=int, default=2)
    parser.add_argument("--max-loops", type=int)
    parser.add_argument("--policy", choices=("ouro", "ouro_delayed"), default="ouro")
    parser.add_argument("--alias", action="store_true")
    parser.add_argument("--compact", action="store_true")
    parser.add_argument("--reclaim-skipped-credits", action="store_true")
    parser.add_argument("--checkpoint-reader", choices=("native", "torch"), default="native")
    parser.add_argument("--async-scheduling", action="store_true")
    parser.add_argument("--graphs", action="store_true")
    parser.add_argument("--measure", action="store_true")
    parser.add_argument("--profile-range", action="store_true")
    parser.add_argument("--quality-stops", type=Path)
    parser.add_argument("--load-lock", type=Path)
    args = parser.parse_args()
    scope = (
        "finite_batch_engine_e2e" if args.measure else "diagnostic_engine_capture_not_performance"
    )
    manifest = json.loads((args.model / "verified-manifest.json").read_text())
    stop_spec = None
    if args.quality_stops is not None:
        if args.measure or args.profile_range or args.async_scheduling:
            raise ValueError("quality text stops currently require synchronous diagnostic mode")
        stop_spec = json.loads(args.quality_stops.read_text())
        if (
            hashlib.sha256((args.model / "tokenizer.json").read_bytes()).hexdigest()
            != stop_spec["tokenizer_sha256"]
        ):
            raise ValueError("quality stop table belongs to a different tokenizer")
        scope = "quality_engine_generation_with_eos_and_text_stops"
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
    cache_config = CacheConfig(
        args.num_blocks,
        16,
        alias_last_exited=args.alias,
        compact_last_exited=args.compact,
        reclaim_skipped_credits=args.reclaim_skipped_credits,
    )
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
    modules = {}
    for name in (
        "vllm_rlt.engine.llm_engine",
        "vllm_rlt.core.alias_kv_cache",
        "vllm_rlt.core.compact_kv_cache",
        "vllm_rlt.core.kv_cache_manager",
        "vllm_rlt.kernels.triton_attention",
        "vllm_rlt.models.ouro",
        "vllm_rlt.models.nanbeige",
        "experiments.loopkv.checkpoint",
        "experiments.loopkv.text_stops",
    ):
        path = Path(importlib.import_module(name).__file__).resolve()
        modules[name] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
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
            "modules": modules,
            "environment": {
                key: os.environ.get(key)
                for key in (
                    "CUDA_VISIBLE_DEVICES",
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "TRITON_CACHE_DIR",
                    "CUDA_CACHE_PATH",
                    "PYTHONPATH",
                )
            },
            "cache": asdict(cache_config),
            "scheduler": asdict(scheduler),
            "execution": asdict(execution),
            "policy": args.policy,
            "sampling": {
                "max_tokens": args.max_tokens,
                "exit_threshold": args.threshold,
                "min_loops": args.min_loops,
                "max_loops": args.max_loops,
                "seed": 17,
                "ignore_eos": stop_spec is None,
            },
            "quality_stops_sha256": hashlib.sha256(args.quality_stops.read_bytes()).hexdigest()
            if args.quality_stops is not None
            else None,
            "request_count": len(prompts),
            "scope": scope,
            "warmup_rounds": int(args.measure),
            "profile_range": args.profile_range,
            "measurement_boundary": "all submissions through full drain, after warmup",
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
    load_lock = None
    if args.load_lock is not None:
        import fcntl

        load_lock = args.load_lock.open("a")
        fcntl.flock(load_lock, fcntl.LOCK_EX)
    if args.checkpoint_reader == "native":
        model = AutoModelForCausalLM.from_pretrained(
            str(args.model), device="cuda", dtype=torch.bfloat16
        )
    else:
        from experiments.loopkv.checkpoint import load_model

        model = load_model(args.model, torch.device("cuda"), torch.bfloat16)
    if load_lock is not None:
        load_lock.close()
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
        max_tokens=args.max_tokens,
        exit_threshold=args.threshold,
        min_loops=args.min_loops,
        max_loops=args.max_loops,
        seed=17,
        ignore_eos=stop_spec is None,
    )
    if args.measure:
        drive(engine, prompts, params, trace=False)
        if engine.cache_manager.num_free_blocks != args.num_blocks:
            raise RuntimeError("warmup did not return all KV blocks")
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    if args.profile_range:
        torch.cuda.cudart().cudaProfilerStart()
    completed, steps, elapsed = drive(
        engine, prompts, params, trace=not args.measure, stop_spec=stop_spec
    )
    if args.profile_range:
        torch.cuda.cudart().cudaProfilerStop()
    # The first output comes from prefill; its exit depth is not a decode KV position.
    exits = [r for output in completed.values() for r in output["exit_depths"][1:]]
    output_count = sum(len(o["token_ids"]) for o in completed.values())
    ledgers = [
        account(shape, len(o["prompt_token_ids"]), o["exit_depths"][1:]) for o in completed.values()
    ]
    summary = {
        "scope": scope,
        "seconds": elapsed,
        "tokens_per_second": output_count / elapsed,
        "completed_requests": len(completed),
        "output_tokens": output_count,
        "decode_exit_histogram": dict(Counter(exits)),
        "cached_decode_positions": len(exits),
        "executed_decode_loops": sum(exits),
        "recurrent_batch_histogram": dict(
            Counter(s["rows"] for s in steps if s["stage"] == Stage.RECURRENT.value)
        )
        if steps
        else None,
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
    if args.compact:
        cache = engine.cache_manager
        summary["compact"] = {
            "peak_live_records": cache.peak_live_records,
            "record_bytes": shape.record_bytes,
            "peak_unique_payload_bytes": cache.peak_live_records * shape.record_bytes,
            "peak_persistent_metadata_bytes": cache.peak_metadata_bytes,
            "live_records_after_drain": cache.live_records,
            "admission": "reclaimed_past_skips_full_worst_future"
            if args.reclaim_skipped_credits
            else "full_worst_future_credits",
            "payload_fragmentation_bytes": 0,
        }
    dump(args.out / "run_summary.json", summary)
    for filename, values in (("requests.jsonl", completed.values()), ("steps.jsonl", steps)):
        (args.out / filename).write_text("".join(json.dumps(row) + "\n" for row in values))
    engine.close()
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
