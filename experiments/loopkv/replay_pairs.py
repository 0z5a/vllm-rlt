"""Warm and measure complete frozen schedules in balanced native/alias pairs."""

import argparse
import gc
import hashlib
import json
import os
import subprocess
from pathlib import Path

import torch

from experiments.loopkv.capture import drive, dump
from experiments.loopkv.checkpoint import load_model
from experiments.loopkv.metrics import WorkCounters
from experiments.loopkv.schedule_trace import Replay
from vllm_rlt import CacheConfig, ExecutionConfig, ExitConfig, SamplingParams, SchedulerConfig
from vllm_rlt.engine.llm_engine import LLMEngine


def measure(model, case, reference, events, expected_work, alias, out):
    engine = LLMEngine(
        model,
        attention_backend="triton",
        cache_config=CacheConfig(case["blocks"], 16, alias_last_exited=alias),
        scheduler_config=SchedulerConfig(max_num_seqs=case["batch"], max_num_batched_tokens=128),
        exit_config=ExitConfig("ouro_delayed"),
        execution_config=ExecutionConfig(
            async_scheduling=True, static_buffers=True, cuda_graphs=True
        ),
    )
    prompts = [reference[str(i)]["prompt_token_ids"] for i in range(len(reference))]
    exits = {rid: item["exit_depths"] for rid, item in reference.items()}
    params = SamplingParams(
        max_tokens=128, min_loops=2, max_loops=4, exit_threshold=0.5, seed=17, ignore_eos=True
    )
    for phase in ("warmup", "measured"):
        replay = Replay(engine, events, exits)
        counters = WorkCounters()
        torch.cuda.reset_peak_memory_stats()
        captures = engine.model_runner.graphs.captures
        replays = engine.model_runner.graphs.replays
        fallbacks = engine.model_runner.graphs.fallbacks
        completed, _, seconds = drive(engine, prompts, params, trace=False, counters=counters)
        replay.assert_drained()
        row = {
            "phase": phase,
            "seconds": seconds,
            "alias": alias,
            "exact_request_objects": completed == reference,
            "work": counters.summary(),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "graphs": {
                "captures": engine.model_runner.graphs.captures - captures,
                "replays": engine.model_runner.graphs.replays - replays,
                "fallbacks": engine.model_runner.graphs.fallbacks - fallbacks,
            },
        }
        dump(out / f"{phase}.json", row)
        dump(out / f"{phase}-requests.json", completed)
        assert row["exact_request_objects"], "fixed-schedule output/exit mismatch"
        # JSON counters have string keys; normalize the in-memory result identically.
        assert json.loads(json.dumps(row["work"])) == expected_work
    engine.close()
    del replay, engine
    gc.collect()
    torch.cuda.empty_cache()
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    assert sha == plan["source_sha"] and not subprocess.check_output(["git", "diff", "HEAD"])
    args.out.mkdir(exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(17)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dump(
        args.out / "plan.json",
        {
            "plan": plan,
            "plan_sha256": hashlib.sha256(args.plan.read_bytes()).hexdigest(),
            "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(),
            "scope": "fixed_work_complete_schedule_replay_not_live_serving",
            "boundary": "request submission through full drain after one complete warmup; "
            "includes replay scheduling and equal CPU counters; excludes loading, "
            "trace parsing, diagnostic comparisons and file writes",
        },
    )
    # The outer controller owns both GPU locks and heavy-IO for the whole quiet run.
    model = load_model(Path(plan["model"]), torch.device("cuda"), torch.bfloat16)
    for case in plan["cases"]:
        data = {}
        for name, item in case["inputs"].items():
            path = Path(item["path"])
            assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]
            data[name] = json.loads(path.read_text())
        for pair, order in enumerate(case["orders"]):
            results = {}
            for alias in order:
                label = "alias" if alias else "native"
                folder = args.out / f"{case['name']}-pair{pair}-{label}"
                folder.mkdir()
                results[label] = measure(
                    model,
                    case,
                    data["requests"],
                    data["trace"],
                    data["summary"]["work"],
                    alias,
                    folder,
                )
            row = {
                "case": case["name"],
                "pair": pair,
                "order": order,
                "speedup": results["native"]["seconds"] / results["alias"]["seconds"],
            }
            dump(args.out / f"{case['name']}-pair{pair}.json", row)
            print(json.dumps(row), flush=True)
    dump(args.out / "DONE.json", {"returncode": 0})


if __name__ == "__main__":
    main()
