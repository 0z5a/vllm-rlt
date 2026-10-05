"""Measure warmed complete generation in balanced metadata off/on pairs."""

import argparse
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
from vllm_rlt import (
    CacheConfig,
    ExecutionConfig,
    ExitConfig,
    SamplingParams,
    SchedulerConfig,
)
from vllm_rlt.engine.llm_engine import LLMEngine


def run_arm(model, plan, case, storage, enabled, prompts, out, reference):
    cuda = next(model.parameters()).device.type == "cuda"
    engine = LLMEngine(
        model,
        attention_backend="triton" if cuda else "torch",
        cache_config=CacheConfig(case["blocks"], 16, compact_last_exited=storage == "compact"),
        scheduler_config=SchedulerConfig(max_num_seqs=case["batch"], max_num_batched_tokens=128),
        exit_config=ExitConfig("ouro"),
        execution_config=ExecutionConfig(prefill_batch_metadata=enabled),
    )
    params = SamplingParams(
        max_tokens=plan["outputs"],
        min_loops=plan["loops"],
        max_loops=plan["loops"],
        exit_threshold=1.0,
        seed=17,
        ignore_eos=True,
    )
    out.mkdir(parents=True)
    for phase in ("reference",) if reference is None else ("warmup", "measured"):
        torch.manual_seed(17)
        counters = WorkCounters()
        if cuda:
            torch.cuda.reset_peak_memory_stats()
        completed, steps, seconds = drive(
            engine,
            prompts,
            params,
            trace=phase == "reference",
            counters=counters,
        )
        row = {
            "case": case,
            "storage": storage,
            "enabled": enabled,
            "phase": phase,
            "seconds": seconds,
            "work": counters.summary(),
            "output_tokens": sum(len(r["token_ids"]) for r in completed.values()),
            "free_blocks": engine.cache_manager.num_free_blocks,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated() if cuda else None,
            "peak_reserved_bytes": torch.cuda.max_memory_reserved() if cuda else None,
        }
        if storage == "compact":
            row["compact_live_records"] = engine.cache_manager.live_records
            row["compact_reserved_records"] = engine.cache_manager._reserved_records
        dump(out / f"{phase}-requests.json", completed)
        dump(out / f"{phase}.json", row)
        if phase == "reference":
            dump(out / "steps.json", steps)
        assert len(completed) == case["requests"]
        assert row["output_tokens"] == case["requests"] * plan["outputs"]
        assert row["free_blocks"] == case["blocks"]
        assert all(
            r["finished"] and r["exit_depths"] == [plan["loops"]] * plan["outputs"]
            for r in completed.values()
        )
        if storage == "compact":
            assert row["compact_live_records"] == row["compact_reserved_records"] == 0
        if reference is not None:
            assert completed == reference["requests"], (
                "complete request objects differ; raw files retained"
            )
            assert row["work"] == reference["work"], "scheduling work differs; raw files retained"
    engine.close()
    del engine
    gc.collect()
    if cuda:
        torch.cuda.empty_cache()
    return {"requests": completed, "work": row["work"], "seconds": seconds}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    source = Path(plan["source"])
    assert (
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
        == plan["source_sha"]
    )
    assert not subprocess.check_output(["git", "diff", "HEAD"], cwd=source)
    prompts_path, model_path = Path(plan["prompts"]), Path(plan["model"])
    assert hashlib.sha256(prompts_path.read_bytes()).hexdigest() == plan["prompts_sha256"]
    checkpoint = json.loads(Path(plan["checkpoint_manifest"]).read_text())
    for item in checkpoint["files"]:
        path = model_path / item["path"]
        assert path.stat().st_size == item["bytes"]
        with path.open("rb") as stream:
            assert hashlib.file_digest(stream, "sha256").hexdigest() == item["sha256"]
    prompts = json.loads(prompts_path.read_text())
    assert all(len(p) == plan["prompt_length"] for p in prompts)
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(17)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(plan["device"])
    dump(
        args.out / "plan.json",
        {
            "plan": plan,
            "checkpoint": checkpoint,
            "pid": os.getpid(),
            "plan_sha256": hashlib.sha256(args.plan.read_bytes()).hexdigest(),
            "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name() if device.type == "cuda" else None,
            "scope": "finite complete engine generation; not HTTP or quality evaluation",
            "timing_boundary": (
                "request submission to synchronized full drain; one complete warmup; "
                "includes equal CPU work counters; excludes loading, serialization and disk writes"
            ),
        },
    )
    model = load_model(model_path, device, torch.bfloat16)
    assert model.config.total_ut_steps == plan["loops"]
    dump(args.out / "MODEL_LOADED.json", {"pid": os.getpid(), "time_unix": time.time()})
    print("MODEL_LOADED", flush=True)
    for case in plan["cases"]:
        inputs = [prompts[i % len(prompts)] for i in range(case["requests"])]
        reference = run_arm(
            model,
            plan,
            case,
            "native",
            False,
            inputs,
            args.out / f"{case['name']}-reference",
            None,
        )
        for storage in plan["storage"]:
            for pair, order in enumerate(plan["orders"]):
                seconds = {}
                for enabled in order:
                    label = "on" if enabled else "off"
                    path = args.out / f"{case['name']}-{storage}-pair{pair}-{label}"
                    result = run_arm(model, plan, case, storage, enabled, inputs, path, reference)
                    seconds[label] = result["seconds"]
                row = {
                    "case": case["name"],
                    "storage": storage,
                    "pair": pair,
                    "order": order,
                    "seconds": seconds,
                    "speedup": seconds["off"] / seconds["on"],
                }
                dump(args.out / f"{case['name']}-{storage}-pair{pair}.json", row)
                print(json.dumps(row), flush=True)
    dump(args.out / "DONE.json", {"returncode": 0, "time_unix": time.time()})


if __name__ == "__main__":
    main()
