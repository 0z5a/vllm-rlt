"""Single-load official-weight execution comparison; diagnostics, not timing evidence."""

import argparse
import fcntl
import gc
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import torch

from experiments.loopkv.capture import drive, dump
from experiments.loopkv.checkpoint import load_model
from experiments.loopkv.metrics import WorkCounters
from vllm_rlt import CacheConfig, ExecutionConfig, ExitConfig, SamplingParams, SchedulerConfig
from vllm_rlt.core.compact_kv_cache import CompactKVCacheManager
from vllm_rlt.engine.llm_engine import LLMEngine


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--load-lock", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    device = torch.device(plan.get("device", "cuda"))
    cuda = device.type == "cuda"
    backend = plan.get("attention_backend", "triton" if cuda else "torch")
    source = Path(__file__).resolve().parents[2]
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    assert sha == plan["source_sha"]
    assert not subprocess.check_output(["git", "diff", "HEAD"], cwd=source)
    args.out.mkdir(parents=True, exist_ok=False)
    model_path, prompt_path = Path(plan["model"]), Path(plan["prompts"])
    prompts = json.loads(prompt_path.read_text())
    stops_path = Path(plan["quality_stops"]) if plan.get("quality_stops") else None
    stop_spec = json.loads(stops_path.read_text()) if stops_path else None
    if stop_spec:
        assert (
            hashlib.sha256((model_path / "tokenizer.json").read_bytes()).hexdigest()
            == stop_spec["tokenizer_sha256"]
        )
    dump(
        args.out / "plan.json",
        {
            "plan": plan,
            "plan_sha256": hashlib.sha256(args.plan.read_bytes()).hexdigest(),
            "checkpoint": json.loads((model_path / "verified-manifest.json").read_text()),
            "prompts_sha256": hashlib.sha256(prompt_path.read_bytes()).hexdigest(),
            "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "pid": os.getpid(),
            "torch": torch.__version__,
            "device": str(device),
            "attention_backend": backend,
            "gpu": torch.cuda.get_device_name(device) if cuda else None,
            "quality_stops_sha256": hashlib.sha256(stops_path.read_bytes()).hexdigest()
            if stops_path
            else None,
            "scope": "quality_generation_not_performance"
            if stop_spec
            else "official_weight_diagnostic_not_performance_or_quality",
        },
    )
    torch.set_num_threads(plan.get("cpu_threads", 4))
    torch.manual_seed(17)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    with args.load_lock.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        model = load_model(model_path, device, torch.bfloat16)
        if cuda:
            torch.cuda.synchronize(device)
    dump(args.out / "MODEL_LOADED.json", {"unix": time.time(), "pid": os.getpid()})
    for case in plan["cases"]:
        baselines = {}
        for arm in plan["arms"]:
            engine = LLMEngine(
                model,
                attention_backend=backend,
                exit_config=ExitConfig(plan["policy"]),
                cache_config=CacheConfig(
                    case["blocks"],
                    16,
                    alias_last_exited=arm["alias"],
                    compact_last_exited=arm.get("compact", False),
                    reclaim_skipped_credits=arm.get("reclaim", False),
                ),
                scheduler_config=SchedulerConfig(
                    max_num_seqs=case["batch"], max_num_batched_tokens=max(128, case["batch"])
                ),
                execution_config=ExecutionConfig(
                    async_scheduling=arm["async"],
                    static_buffers=arm["graphs"],
                    cuda_graphs=arm["graphs"],
                ),
            )
            if cuda:
                torch.cuda.reset_peak_memory_stats(device)
            counters = WorkCounters()
            # Match stochastic recurrent initialization across storage arms.
            torch.manual_seed(plan.get("state_seed", 17))
            completed, steps, elapsed = drive(
                engine,
                [prompts[i % len(prompts)] for i in range(case["requests"])],
                SamplingParams(
                    max_tokens=case["outputs"],
                    min_loops=plan["min_loops"],
                    max_loops=plan["max_loops"],
                    exit_threshold=plan["threshold"],
                    seed=17,
                    ignore_eos=stop_spec is None,
                ),
                trace=True,
                counters=counters,
                stop_spec=stop_spec,
            )
            folder = args.out / f"{case['name']}-{arm['name']}"
            folder.mkdir()
            dump(folder / "requests.json", completed)
            dump(folder / "steps.json", steps)
            assert len(completed) == case["requests"]
            assert engine.cache_manager.num_free_blocks == case["blocks"]
            if not arm["alias"] and not arm.get("compact", False):
                baselines[arm["execution"]] = completed
            baseline = baselines[arm["execution"]]
            summary = {
                "case": case,
                "arm": arm,
                "state_seed": plan.get("state_seed", 17),
                "seconds_not_performance": elapsed,
                "output_tokens": sum(len(row["token_ids"]) for row in completed.values()),
                "token_different_requests": sum(
                    row["token_ids"] != baseline[rid]["token_ids"] for rid, row in completed.items()
                ),
                "exit_different_requests": sum(
                    row["exit_depths"] != baseline[rid]["exit_depths"]
                    for rid, row in completed.items()
                ),
                "exact_request_objects": completed == baseline,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if cuda else None,
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if cuda else None,
                "free_blocks_after_drain": engine.cache_manager.num_free_blocks,
                **counters.summary(),
            }
            if isinstance(engine.cache_manager, CompactKVCacheManager):
                cache = engine.cache_manager
                assert cache.live_records == cache._reserved_records == 0
                summary["compact"] = {
                    "peak_live_records": cache.peak_live_records,
                    "peak_metadata_bytes": cache.peak_metadata_bytes,
                    "live_records_after_drain": cache.live_records,
                    "reserved_records_after_drain": cache._reserved_records,
                }
                del cache
            if engine.model_runner.graphs is not None:
                graphs = engine.model_runner.graphs
                assert graphs.captures and graphs.replays > graphs.captures
                summary["graphs"] = {
                    "captures": graphs.captures,
                    "replays": graphs.replays,
                    "fallbacks": graphs.fallbacks,
                }
                del graphs
            dump(folder / "summary.json", summary)
            print(json.dumps(summary), flush=True)
            engine.close()
            del engine
            gc.collect()
            if cuda:
                torch.cuda.empty_cache()
    dump(args.out / "DONE.json", {"returncode": 0, "end_unix": time.time()})


if __name__ == "__main__":
    main()
