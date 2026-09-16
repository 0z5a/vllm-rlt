"""Real-model closed-loop engine benchmark with raw outputs and host token timings."""

import argparse
import gc
import hashlib
import json
import math
import os
import random
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import torch

from vllm_lt import CacheConfig, ExecutionConfig, ExitConfig, SamplingParams, SchedulerConfig
from vllm_lt.engine.llm_engine import LLMEngine
from vllm_lt.models import OuroConfig, OuroForCausalLM

VARIANTS = {
    "S": ExecutionConfig(),
    "AS": ExecutionConfig(async_scheduling=True, multi_stream=False),
    "AM": ExecutionConfig(async_scheduling=True, multi_stream=True),
    "A": ExecutionConfig(),
    "B": ExecutionConfig(static_buffers=True, pad_to_power_of_two=True),
    "C": ExecutionConfig(
        async_scheduling=True,
        multi_stream=False,
        static_buffers=True,
        pad_to_power_of_two=True,
    ),
    "D": ExecutionConfig(
        async_scheduling=True,
        static_buffers=True,
        pad_to_power_of_two=True,
    ),
}


def sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


def percentile(values, q):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(q * len(ordered)) - 1)] if ordered else None


def drive(engine, prompts, concurrency, output_length, seconds, requests, device):
    """Replenish until deadline/count; include admission and final drain in timing."""
    params = SamplingParams(
        max_tokens=output_length,
        max_loops=4,
        min_loops=2,
        exit_threshold=1.0,
        temperature=0.0,
        ignore_eos=True,
    )
    records, active = [], {}
    issued = 0
    sync(device)
    start = time.perf_counter()

    def submit():
        nonlocal issued
        rid = str(issued)
        prompt_index = issued % len(prompts)
        begin = time.perf_counter()
        engine.add_request(rid, prompts[prompt_index], params)
        active[rid] = dict(
            request_id=rid,
            prompt_index=prompt_index,
            start_seconds=begin - start,
            token_times_seconds=[],
        )
        issued += 1

    for _ in range(min(concurrency, requests) if requests else concurrency):
        submit()
    while engine.has_unfinished_requests():
        outputs = engine.step()
        observed = time.perf_counter() - start
        for output in outputs:
            row = active[output.request_id]
            row["token_times_seconds"].append(observed)
            if output.finished:
                row.update(
                    end_seconds=observed,
                    token_ids=output.token_ids,
                    exit_depths=output.exit_depths,
                    finish_reason=output.finish_reason,
                )
                records.append(row)
                del active[output.request_id]
                if (requests and issued < requests) or (not requests and observed < seconds):
                    submit()
    sync(device)
    elapsed = time.perf_counter() - start
    assert not active and len(records) == issued
    assert engine.cache_manager.num_used_blocks == 0
    assert not engine.model_runner.state_slots
    for row in records:
        assert len(row["token_ids"]) == output_length
        assert len(row["token_times_seconds"]) == output_length
        assert row["exit_depths"] == [4] * output_length
        assert row["finish_reason"] == "length"
    ttft = [r["token_times_seconds"][0] - r["start_seconds"] for r in records]
    e2e = [r["end_seconds"] - r["start_seconds"] for r in records]
    tpot = [
        (r["end_seconds"] - r["token_times_seconds"][0]) / (output_length - 1)
        for r in records
        if output_length > 1
    ]
    itl = [
        b - a
        for r in records
        for a, b in zip(r["token_times_seconds"], r["token_times_seconds"][1:])
    ]
    summary = dict(
        elapsed_seconds=elapsed,
        completed_requests=issued,
        output_tokens=issued * output_length,
        output_tokens_per_second=issued * output_length / elapsed,
        requests_per_second=issued / elapsed,
        error_count=0,
        all_exit_depths_four=True,
    )
    for name, values in (("ttft", ttft), ("e2e", e2e), ("tpot", tpot), ("itl", itl)):
        summary[name + "_seconds"] = {
            "p50": percentile(values, 0.50),
            "p95": percentile(values, 0.95),
            "p99": percentile(values, 0.99),
        }
    if device == "cuda":
        summary["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        summary["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
    return records, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--seconds", type=float, default=60)
    parser.add_argument("--requests", type=int, default=0, help="Fixed count overrides seconds")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--input-length", type=int, default=128)
    parser.add_argument("--output-length", type=int, default=512)
    parser.add_argument("--variants", nargs="+", choices=list(VARIANTS), default=list("ABCD"))
    parser.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--num-blocks", type=int, default=1280)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--toy", action="store_true")
    args = parser.parse_args()
    if (
        not 1 <= args.concurrency <= 8
        or min(args.input_length, args.output_length, args.repeats, args.num_blocks) < 1
        or args.requests < 0
        or (not args.requests and args.seconds <= 0)
    ):
        parser.error("invalid positive count/duration or concurrency outside [1,8]")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.manual_seed(123)
    dtype = getattr(torch, args.dtype)
    if args.toy:
        model = OuroForCausalLM(OuroConfig.tiny()).to(device=args.device, dtype=dtype)
        prompts = [[2 + ((i + j) % 50) for j in range(args.input_length)] for i in range(8)]
    else:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
        texts = [
            "Explain matrix multiplication and give a worked example. ",
            "Describe how a computer executes a program step by step. ",
            "Explain photosynthesis and its role in an ecosystem. ",
            "Compare different methods for sorting a list of numbers. ",
            "Write a clear introduction to probability and statistics. ",
            "Explain the water cycle with concrete examples. ",
            "Describe the relationship between force and acceleration. ",
            "Explain how a database stores and retrieves information. ",
        ]
        prompts = []
        for text in texts:
            ids = tokenizer.encode(text, add_special_tokens=False)
            prompts.append((ids * math.ceil(args.input_length / len(ids)))[: args.input_length])
        model = OuroForCausalLM.from_pretrained(args.model, device=args.device, dtype=dtype)
    manifest = dict(
        arguments={**vars(args), "output": str(args.output)},
        cpu_affinity=sorted(os.sched_getaffinity(0)),
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        torch_version=torch.__version__,
        cuda_version=torch.version.cuda,
        device_name=torch.cuda.get_device_name() if args.device == "cuda" else "cpu",
        commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        dirty_status=subprocess.check_output(["git", "status", "--short"], text=True),
        prompts=prompts,
        prompt_hash=hashlib.sha256(json.dumps(prompts).encode()).hexdigest(),
        timing="host token observations; admission and drain included; warmup excluded",
        workload="closed-loop cyclic 8 prompts; timed runs may finish different counts",
        tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
        bf16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        caveat="BF16 multi-stream equivalence unresolved; engine-only, not HTTP",
    )
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    reference, summaries = {}, []
    for repeat in range(args.repeats):
        order = list(args.variants)
        random.Random(1729 + repeat).shuffle(order)
        for variant in order:
            gc.collect()
            if args.device == "cuda":
                torch.cuda.empty_cache()
            engine = LLMEngine(
                model,
                cache_config=CacheConfig(num_blocks=args.num_blocks, layout="last_exited"),
                scheduler_config=SchedulerConfig(max_num_seqs=8, max_num_batched_tokens=128),
                exit_config=ExitConfig("ouro_delayed"),
                execution_config=VARIANTS[variant],
                attention_backend="triton" if args.device == "cuda" else "torch",
            )
            assert engine.execution_config == VARIANTS[variant]
            if not VARIANTS[variant].static_buffers:
                assert not engine.model_runner.workspaces
                assert engine.model_runner.states is None
            drive(
                engine,
                prompts,
                args.concurrency,
                args.output_length,
                0,
                args.concurrency,
                args.device,
            )
            if args.device == "cuda":
                torch.cuda.reset_peak_memory_stats()
            rows, summary = drive(
                engine,
                prompts,
                args.concurrency,
                args.output_length,
                args.seconds,
                args.requests,
                args.device,
            )
            differences = 0
            for row in rows:
                key = row["prompt_index"]
                if key not in reference:
                    reference[key] = row["token_ids"]
                differences += row["token_ids"] != reference[key]
            summary.update(
                variant=variant,
                repeat=repeat + 1,
                concurrency=args.concurrency,
                execution=asdict(engine.execution_config),
                output_disagreements_with_first_observed_prompt=differences,
            )
            prefix = args.output / f"{variant}_repeat{repeat + 1}"
            prefix.with_suffix(".requests.jsonl").write_text(
                "".join(json.dumps(r) + "\n" for r in rows)
            )
            prefix.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2))
            summaries.append(summary)
            (args.output / "results.json").write_text(json.dumps(summaries, indent=2))
            print(json.dumps(summary), flush=True)
            engine.model_runner.synchronize()
            del engine


if __name__ == "__main__":
    main()
