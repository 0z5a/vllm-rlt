"""Finite CUDA self-replay and alias replay qualification; no performance claim."""

import argparse
import fcntl
import gc
import hashlib
import json
import subprocess
from pathlib import Path

import torch

from experiments.loopkv.capture import drive, dump
from experiments.loopkv.checkpoint import load_model
from experiments.loopkv.metrics import WorkCounters
from experiments.loopkv.schedule_trace import Capture, Replay
from vllm_rlt import CacheConfig, ExecutionConfig, ExitConfig, SamplingParams, SchedulerConfig
from vllm_rlt.engine.llm_engine import LLMEngine
from vllm_rlt.models import OuroConfig, OuroForCausalLM


def run_case(model, prompts, batch, outputs, folder, compact=False):
    expected, events = None, None
    rows = []
    modes = (
        ("capture", "self-replay", "compact-replay")
        if compact
        else ("capture", "self-replay", "alias-replay")
    )
    for mode in modes:
        engine = LLMEngine(
            model,
            attention_backend="triton",
            cache_config=CacheConfig(
                batch * 64,
                16,
                alias_last_exited=mode == "alias-replay",
                compact_last_exited=mode == "compact-replay",
            ),
            scheduler_config=SchedulerConfig(max_num_seqs=batch, max_num_batched_tokens=128),
            exit_config=ExitConfig("ouro_delayed"),
            execution_config=ExecutionConfig(
                async_scheduling=True, static_buffers=True, cuda_graphs=True
            ),
        )
        trace = (
            Capture(engine)
            if mode == "capture"
            else Replay(
                engine, events, {rid: item["exit_depths"] for rid, item in expected.items()}
            )
        )
        counters = WorkCounters()
        completed, _, _ = drive(
            engine,
            [prompts[i % len(prompts)] for i in range(batch * 2)],
            SamplingParams(
                max_tokens=outputs,
                min_loops=2,
                max_loops=4,
                exit_threshold=0.5,
                seed=17,
                ignore_eos=True,
            ),
            trace=False,
            counters=counters,
        )
        if mode == "capture":
            expected, events = completed, trace.events
            dump(folder / "trace.json", events)
        else:
            trace.assert_drained()
        graphs = engine.model_runner.graphs
        row = {
            "mode": mode,
            "exact_request_objects": completed == expected,
            "captures": graphs.captures,
            "replays": graphs.replays,
            "fallbacks": graphs.fallbacks,
            "work": counters.summary(),
            "free_blocks": engine.cache_manager.num_free_blocks,
        }
        assert row["replays"] > row["captures"] > 0
        assert row["free_blocks"] == batch * 64
        if mode != "capture":
            assert row["work"] == rows[0]["work"]
        rows.append(row)
        dump(folder / f"{mode}-requests.json", completed)
        dump(folder / f"{mode}.json", row)
        engine.close()
        del graphs, trace, engine
        gc.collect()
        torch.cuda.empty_cache()
    assert rows[1]["exact_request_objects"], "native self-replay changed token/exit results"
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--load-lock", type=Path, required=True)
    parser.add_argument("--compact", action="store_true")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    assert sha == plan["source_sha"] and not subprocess.check_output(["git", "diff", "HEAD"])
    args.out.mkdir(exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(17)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if plan["model"]:
        with args.load_lock.open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            model = load_model(Path(plan["model"]), torch.device("cuda"), torch.bfloat16)
            torch.cuda.synchronize()
        checkpoint = json.loads((Path(plan["model"]) / "verified-manifest.json").read_text())
        prompts = json.loads(Path(plan["prompts"]).read_text())
    else:
        model = (
            OuroForCausalLM(
                OuroConfig(
                    vocab_size=64,
                    hidden_size=32,
                    intermediate_size=64,
                    num_hidden_layers=2,
                    num_attention_heads=4,
                    num_key_value_heads=2,
                    head_dim=64,
                    max_position_embeddings=128,
                    total_ut_steps=4,
                )
            )
            .to(device="cuda", dtype=torch.bfloat16)
            .eval()
        )
        checkpoint, prompts = None, [[2, 3] * 17, [4] * 33]
    dump(
        args.out / "plan.json",
        {
            "plan": plan,
            "checkpoint": checkpoint,
            "plan_sha256": hashlib.sha256(args.plan.read_bytes()).hexdigest(),
            "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(),
            "scope": "fixed_logical_schedule_cuda_qualification_not_performance",
        },
    )
    for batch in plan["batches"]:
        folder = args.out / f"b{batch}"
        folder.mkdir()
        rows = run_case(model, prompts, batch, plan["outputs"], folder, args.compact)
        print(json.dumps({"batch": batch, "results": rows}), flush=True)
    dump(args.out / "DONE.json", {"returncode": 0})


if __name__ == "__main__":
    main()
