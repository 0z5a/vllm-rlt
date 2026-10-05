"""Single-load, finite HTTP workloads; timing scope is declared in the frozen plan."""

import argparse
import asyncio
import fcntl
import gc
import hashlib
import json
import math
import subprocess
from pathlib import Path

import torch

from experiments.loopkv.capture import dump
from experiments.loopkv.checkpoint import load_model
from experiments.loopkv.http_load import run_load
from experiments.loopkv.replay_pairs import telemetry
from vllm_rlt import CacheConfig, ExecutionConfig, ExitConfig, SamplingParams, SchedulerConfig
from vllm_rlt.engine.llm_engine import LLMEngine


def percentile(values, quantile):
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * quantile) - 1)] if ordered else None


def summarize(result, slo):
    ttft, latency, intervals, request_p99, lag = [], [], [], [], []
    good_requests = good_tokens = 0
    for row in result["requests"]:
        times = row["token_times_s"]
        origin = row["planned_s"] if result["arrival_mode"] == "open_loop" else row["sent_s"]
        first = times[0] - origin
        elapsed = row["finished_s"] - origin
        itl = [right - left for left, right in zip(times, times[1:])]
        tail = percentile(itl, 0.99)
        ttft.append(first)
        latency.append(elapsed)
        intervals.extend(itl)
        if tail is not None:
            request_p99.append(tail)
        lag.append(row["sent_s"] - row["planned_s"])
        if (
            slo
            and first <= slo["ttft_s"]
            and elapsed <= slo["request_s"]
            and (tail is None or tail <= slo["request_p99_itl_s"])
        ):
            good_requests += 1
            good_tokens += len(times)
    seconds = result["seconds"]
    return {
        "complete_requests": len(ttft),
        "output_tokens": sum(len(row["token_ids"]) for row in result["requests"]),
        "request_goodput_s": good_requests / seconds if slo else None,
        "token_goodput_s": good_tokens / seconds if slo else None,
        "latency_seconds": {
            name: {str(q): percentile(values, q) for q in (0.5, 0.9, 0.99)}
            for name, values in (
                ("ttft", ttft),
                ("request", latency),
                ("token_itl", intervals),
                ("per_request_p99_itl", request_p99),
                ("client_dispatch_lag", lag),
            )
        },
        "quantiles": "nearest_rank_descriptive_no_confidence_interval",
        "latency_origin": "planned_arrival" if result["arrival_mode"] == "open_loop" else "send",
        "streaming": "tokens delivered in one HTTP chunk share the client receipt timestamp",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--load-lock", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    assert sha == plan["source_sha"] and not subprocess.check_output(["git", "diff", "HEAD"])
    args.out.mkdir(exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(17)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model_path = Path(plan["model"])
    dump(
        args.out / "plan.json",
        {
            "plan": plan,
            "source_sha": sha,
            "plan_sha256": hashlib.sha256(args.plan.read_bytes()).hexdigest(),
            "checkpoint": json.loads((model_path / "verified-manifest.json").read_text()),
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(),
            "scope": "localhost_token_ID_HTTP_not_production_text_endpoint",
        },
    )
    with args.load_lock.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        model = load_model(model_path, torch.device("cuda"), torch.bfloat16)
        torch.cuda.synchronize()
    for case in plan["cases"]:
        path = Path(case["workload"])
        assert hashlib.sha256(path.read_bytes()).hexdigest() == case["workload_sha256"]
        workload = json.loads(path.read_text())
        baselines = {}
        for arm in plan["arms"]:
            folder = args.out / (case["name"] + "-" + arm["name"])
            folder.mkdir()
            engine = LLMEngine(
                model,
                attention_backend="triton",
                exit_config=ExitConfig(plan["policy"]),
                cache_config=CacheConfig(
                    case["blocks"],
                    16,
                    alias_last_exited=arm["alias"],
                    compact_last_exited=arm["compact"],
                    reclaim_skipped_credits=arm["reclaim"],
                ),
                scheduler_config=SchedulerConfig(
                    max_num_seqs=case["batch"],
                    max_num_batched_tokens=max(128, case["batch"]),
                ),
                execution_config=ExecutionConfig(
                    async_scheduling=arm["async"],
                    static_buffers=arm["graphs"],
                    cuda_graphs=arm["graphs"],
                ),
            )
            params = SamplingParams(**plan["sampling"])
            for repetition in range(case["warmup_repetitions"]):
                warmup = asyncio.run(
                    run_load(
                        engine,
                        workload,
                        params,
                        concurrency=case["concurrency"],
                    )
                )
                dump(folder / f"warmup{repetition}.json", warmup)
            before = telemetry()
            graphs = engine.model_runner.graphs
            graph_before = (graphs.captures, graphs.replays, graphs.fallbacks) if graphs else None
            torch.cuda.reset_peak_memory_stats()
            result = asyncio.run(
                run_load(
                    engine,
                    workload,
                    params,
                    concurrency=case["concurrency"],
                )
            )
            dump(folder / "requests.json", result)
            results = {row["request_id"]: row for row in result["requests"]}
            if not arm["alias"] and not arm["compact"]:
                baselines[arm["execution"]] = results
            baseline = baselines[arm["execution"]]
            summary = summarize(result, case["slo"])
            summary.update(
                arm=arm,
                case=case,
                seconds=result["seconds"],
                work=result["work"],
                tokens_s=summary["output_tokens"] / result["seconds"],
                peak_client_concurrency=result["peak_client_concurrency"],
                token_different_requests=sum(
                    row["token_ids"] != baseline[rid]["token_ids"] for rid, row in results.items()
                ),
                exit_different_requests=sum(
                    row["exit_depths"] != baseline[rid]["exit_depths"]
                    for rid, row in results.items()
                ),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                telemetry_before=before,
                telemetry_after=telemetry(),
                graphs=dict(
                    zip(
                        ("captures", "replays", "fallbacks"),
                        (
                            after - prior
                            for after, prior in zip(
                                (graphs.captures, graphs.replays, graphs.fallbacks), graph_before
                            )
                        ),
                    )
                )
                if graphs
                else None,
                formal_minimum_met=(
                    summary["complete_requests"] >= 2000 and result["seconds"] >= 120
                ),
            )
            dump(folder / "summary.json", summary)
            print(
                json.dumps(
                    {
                        "case": case["name"],
                        "arm": arm["name"],
                        "requests": summary["complete_requests"],
                    }
                ),
                flush=True,
            )
            engine.close()
            del graphs, engine
            gc.collect()
            torch.cuda.empty_cache()
    dump(args.out / "DONE.json", {"returncode": 0})


if __name__ == "__main__":
    main()
