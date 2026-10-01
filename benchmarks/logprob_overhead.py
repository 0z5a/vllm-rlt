"""Alternate unprofiled score modes; report profiler operator snapshots separately."""

import argparse
import json
import statistics
import time
from dataclasses import replace
from pathlib import Path

import torch
from transformers import AutoTokenizer

from vllm_rlt import (
    LLM,
    CacheConfig,
    ExecutionConfig,
    ExitConfig,
    SamplingParams,
    SchedulerConfig,
    SpeculativeConfig,
)
from vllm_rlt.models import OuroForCausalLM


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    torch.set_num_threads(4)
    native = OuroForCausalLM.from_pretrained(args.model, device="cuda", dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    prompts = [tokenizer.encode(text) for text in ("Compute 17 + 25.\n", "Compute 9 * 8.\n")]
    base = [
        SamplingParams(max_tokens=32, temperature=0.9, seed=31 + i, ignore_eos=True)
        for i in range(2)
    ]
    modes = {
        "none": base,
        **{
            name: [replace(p, logprobs=0, logprobs_mode=name) for p in base]
            for name in ("raw", "processed")
        },
    }
    report = {}
    for path in ("sync", "async", "speculative"):
        engine = LLM(
            native,
            attention_backend="triton",
            cache_config=CacheConfig(num_blocks=512),
            scheduler_config=SchedulerConfig(max_num_seqs=2),
            exit_config=ExitConfig("ouro_delayed" if path == "async" else "ouro"),
            execution_config=ExecutionConfig(cuda_graphs=True, async_scheduling=path == "async"),
            speculative_config=SpeculativeConfig(3) if path == "speculative" else None,
        )
        for params in modes.values():
            engine.generate(prompts, params)
        times = {name: [] for name in modes}
        memory = {name: [] for name in modes}
        for index in range(args.repeats):
            names = list(modes)
            names = names[index % 3 :] + names[: index % 3]
            outputs = {}
            for name in names:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                baseline = torch.cuda.memory_allocated()
                started = time.monotonic()
                outputs[name] = engine.generate(prompts, modes[name])
                torch.cuda.synchronize()
                times[name].append(time.monotonic() - started)
                memory[name].append(torch.cuda.max_memory_allocated() - baseline)
            assert all(len(o.token_ids) == 32 for o in outputs["none"])
            assert all(o.log_probs is None for o in outputs["none"])
            for name in ("raw", "processed"):
                assert [o.token_ids for o in outputs[name]] == [
                    o.token_ids for o in outputs["none"]
                ]
                assert all(len(o.log_probs) == len(o.token_ids) for o in outputs[name])
        medians = {name: statistics.median(values) for name, values in times.items()}
        profiles = {}
        for name in ("none", "raw", "processed"):
            for detail in ("ops", "full"):
                with torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ],
                    record_shapes=detail == "full",
                    with_stack=detail == "full",
                    profile_memory=detail == "full",
                ) as profile:
                    engine.generate(prompts, modes[name])
                    torch.cuda.synchronize()
                events = sorted(
                    profile.key_averages(), key=lambda e: e.self_cpu_time_total, reverse=True
                )
                profiles[f"{name}-{detail}"] = [
                    {
                        "operator": e.key,
                        "count": e.count,
                        "self_cpu_us": e.self_cpu_time_total,
                        "self_device_us": e.self_device_time_total,
                        "self_device_memory_bytes": e.self_device_memory_usage,
                    }
                    for e in events[:20]
                ]
        report[path] = {
            "seconds": times,
            "median_seconds": medians,
            "tokens_per_second": {name: 64 / value for name, value in medians.items()},
            "peak_extra_allocated_bytes": memory,
            "profiles": profiles,
            "tokens_unchanged": True,
        }
        engine.close()
        print(path, medians, flush=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        del engine
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
