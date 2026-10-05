"""Serialize resident submissions to compare kernels under identical logical batches."""

import argparse
import fcntl
import gc
import hashlib
import json
import subprocess
from dataclasses import asdict
from pathlib import Path

import torch

from experiments.loopkv.capture import dump
from experiments.loopkv.checkpoint import load_model
from vllm_rlt import CacheConfig, ExecutionConfig, ExitConfig, SamplingParams, SchedulerConfig
from vllm_rlt.core.scheduler import SchedulerOutput
from vllm_rlt.engine.llm_engine import LLMEngine
from vllm_rlt.request import Stage


def signature(batch: SchedulerOutput | None):
    if batch is None:
        return None
    return [
        batch.stage.value,
        [
            [
                item.request.request_id,
                item.request.position,
                item.request.loops_done,
                item.request.num_scheduled_outputs,
                item.token_start,
                item.token_count,
            ]
            for item in batch.items
        ],
    ]


@torch.inference_mode()
def compare(engines: list[LLMEngine], limit: int):
    """Collect each stage before updating policy; this is not the async E2E scheduler."""
    completed = [{}, {}]
    rows = []
    schedule_hash = hashlib.sha256()
    for step in range(limit):
        if all(not engine.has_unfinished_requests() for engine in engines):
            break
        batches = [engine.scheduler.schedule() for engine in engines]
        logical = [signature(batch) for batch in batches]
        if logical[0] != logical[1] or logical[0] is None:
            return {"complete": False, "first_schedule_difference": logical, "rows": rows}
        schedule_hash.update(json.dumps(logical[0], separators=(",", ":")).encode())
        results, tickets, hidden = [], [], []
        for engine, batch in zip(engines, batches):
            ticket = engine.model_runner.submit(batch)
            results.append(ticket.collect())
            tickets.append(ticket)
            hidden.append(torch.stack([item.request.hidden_state for item in batch.items]))
        # collect() has synchronized each producer before these cross-stream reads.
        delta = (hidden[0].float() - hidden[1].float()).abs()
        scores = [abs(a - b) for a, b in zip(*results)] if results[0] is not None else []
        rows.append(
            {
                "step": step,
                "stage": batches[0].stage.value,
                "rows": len(batches[0].items),
                "hidden_exact": torch.equal(hidden[0], hidden[1]),
                "hidden_max_abs_error": delta.max().item(),
                "result_exact": results[0] == results[1],
                "result_max_abs_error": max(scores, default=0),
                "different_result_rows": sum(a != b for a, b in zip(*results))
                if results[0] is not None
                else 0,
            }
        )
        if results[0] != results[1]:
            rows[-1]["logical_batch"] = logical[0]
            rows[-1]["native_results"], rows[-1]["alias_results"] = results
        for index, (engine, batch, ticket, result) in enumerate(
            zip(engines, batches, tickets, results)
        ):
            if batch.stage == Stage.CODA:
                for row, item in enumerate(batch.items):
                    item.request.input_token_tensor = ticket.device_values[row]
            for output in engine._update(batch, result):
                if output.finished:
                    completed[index][output.request_id] = asdict(output)
    else:
        raise AssertionError("serialized oracle exceeded its finite stage limit")
    assert all(not engine.has_unfinished_requests() for engine in engines)
    return {
        "complete": True,
        "exact_request_objects": completed[0] == completed[1],
        "completed": completed,
        "matched_schedule_sha256": schedule_hash.hexdigest(),
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--load-lock", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    assert (
        subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        == plan["source_sha"]
    )
    assert not subprocess.check_output(["git", "diff", "HEAD"])
    args.out.mkdir(exist_ok=False)
    dump(
        args.out / "plan.json",
        {
            "plan": plan,
            "sha256": hashlib.sha256(args.plan.read_bytes()).hexdigest(),
            "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "checkpoint": json.loads((Path(plan["model"]) / "verified-manifest.json").read_text()),
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(),
            "scope": "serialized_kernel_oracle_not_async_e2e_or_performance",
        },
    )
    torch.set_num_threads(4)
    torch.manual_seed(17)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    with args.load_lock.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        model = load_model(Path(plan["model"]), torch.device("cuda"), torch.bfloat16)
        torch.cuda.synchronize()
    prompts = json.loads(Path(plan["prompts"]).read_text())
    for batch in plan["batches"]:
        engines = [
            LLMEngine(
                model,
                attention_backend="triton",
                cache_config=CacheConfig(batch * 64, 16, alias_last_exited=alias),
                scheduler_config=SchedulerConfig(max_num_seqs=batch, max_num_batched_tokens=128),
                exit_config=ExitConfig("ouro_delayed"),
                execution_config=ExecutionConfig(
                    async_scheduling=True, static_buffers=True, cuda_graphs=True
                ),
            )
            for alias in (False, True)
        ]
        params = SamplingParams(
            max_tokens=128, min_loops=2, max_loops=4, exit_threshold=0.5, seed=17, ignore_eos=True
        )
        for engine in engines:
            for i in range(batch * 2):
                engine.add_request(str(i), prompts[i % len(prompts)], params)
        result = compare(engines, batch * 2 * (128 * 7 + 130))
        result["graphs"] = [
            {
                "captures": e.model_runner.graphs.captures,
                "replays": e.model_runner.graphs.replays,
                "fallbacks": e.model_runner.graphs.fallbacks,
            }
            for e in engines
        ]
        result["free_blocks"] = [e.cache_manager.num_free_blocks for e in engines]
        if result["complete"]:
            assert all(value == batch * 64 for value in result["free_blocks"])
            assert all(g["replays"] > g["captures"] > 0 for g in result["graphs"])
        dump(args.out / f"b{batch}.json", result)
        print(
            json.dumps(
                {
                    "batch": batch,
                    "complete": result["complete"],
                    "compared_stages": len(result["rows"]),
                }
            ),
            flush=True,
        )
        for engine in engines:
            engine.close()
        del engine, engines
        gc.collect()
        torch.cuda.empty_cache()
    dump(args.out / "DONE.json", {"returncode": 0})


if __name__ == "__main__":
    main()
