"""Capture CPU/CUDA operators and shapes only, without stack/memory/custom ranges."""

import argparse
import gc
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch

from benchmarks.runtime_baseline import VARIANTS, drive
from vllm_lt import CacheConfig, ExitConfig, SamplingParams, SchedulerConfig
from vllm_lt.engine.llm_engine import LLMEngine
from vllm_lt.models import OuroForCausalLM


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--variants", nargs="+", choices=["S", "AS", "AM"], default=["S", "AS", "AM"]
    )
    parser.add_argument("--active-steps", type=int, default=24)
    args = parser.parse_args()
    if args.active_steps < 1:
        parser.error("active-steps must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.manual_seed(123)
    prompts = json.loads(args.manifest.read_text())["prompts"]
    assert len(prompts) == 8 and all(len(p) == 128 for p in prompts)
    model = OuroForCausalLM.from_pretrained(args.model, device="cuda", dtype=torch.bfloat16)
    settings = dict(
        activities=["CPU", "CUDA"],
        record_shapes=True,
        with_stack=False,
        profile_memory=False,
        with_flops=False,
        custom_ranges=False,
        active_engine_steps=args.active_steps,
        profiler_warmup_engine_steps=4,
        input_tokens=128,
        output_limit=512,
        concurrency=8,
        sampling="greedy, ignore_eos, fixed 4 loops",
        device=torch.cuda.get_device_name(),
        torch_version=torch.__version__,
        caveat="Diagnostic timing only; shape recording may perturb scheduling.",
    )
    (args.output / "settings.json").write_text(json.dumps(settings, indent=2))
    for variant in args.variants:
        dest = args.output / variant
        dest.mkdir()
        gc.collect()
        torch.cuda.empty_cache()
        engine = LLMEngine(
            model,
            cache_config=CacheConfig(num_blocks=1280, layout="last_exited"),
            scheduler_config=SchedulerConfig(max_num_seqs=8, max_num_batched_tokens=128),
            exit_config=ExitConfig("ouro_delayed"),
            execution_config=VARIANTS[variant],
            attention_backend="triton",
        )
        assert not engine.execution_config.static_buffers
        assert not engine.execution_config.pad_to_power_of_two
        print(variant, "full-length warmup", flush=True)
        drive(engine, prompts, 8, 512, 0, 8, "cuda")
        params = SamplingParams(
            max_tokens=512,
            max_loops=4,
            min_loops=2,
            exit_threshold=1,
            temperature=0,
            ignore_eos=True,
        )
        for i, prompt in enumerate(prompts):
            engine.add_request(f"profile-{i}", prompt, params)
        observed = {f"profile-{i}": 0 for i in range(8)}
        while min(observed.values()) < 96:
            for output in engine.step():
                observed[output.request_id] = len(output.token_ids)
        torch.cuda.synchronize()
        before = dict(observed)
        capture_start = time.time()
        print(variant, "capture operators and shapes", flush=True)
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True,
            with_stack=False,
            profile_memory=False,
            with_flops=False,
            schedule=torch.profiler.schedule(wait=0, warmup=4, active=args.active_steps, repeat=1),
        ) as prof:
            for _ in range(4 + args.active_steps):
                for output in engine.step():
                    observed[output.request_id] = len(output.token_ids)
                prof.step()
        prof.export_chrome_trace(str(dest / "trace.json"))
        averages = prof.key_averages(group_by_input_shape=True)
        (dest / "cpu_ops_shapes.txt").write_text(
            averages.table(sort_by="self_cpu_time_total", row_limit=100, max_src_column_width=120)
        )
        (dest / "cuda_ops_shapes.txt").write_text(
            averages.table(
                sort_by="self_device_time_total", row_limit=100, max_src_column_width=120
            )
        )
        operators = [
            dict(
                operator=item.key,
                input_shapes=item.input_shapes,
                calls=item.count,
                self_cpu_us=item.self_cpu_time_total,
                total_cpu_us=item.cpu_time_total,
                self_cuda_us=item.self_device_time_total,
                total_cuda_us=item.device_time_total,
            )
            for item in averages
        ]
        (dest / "ops_shapes.json").write_text(json.dumps(operators, indent=2))
        metadata = dict(
            variant=variant,
            execution=asdict(engine.execution_config),
            output_counts_before=before,
            output_counts_after=observed,
            capture_started_unix=capture_start,
            capture_finished_unix=time.time(),
            trace_bytes=(dest / "trace.json").stat().st_size,
            operator_shape_groups=len(operators),
        )
        (dest / "metadata.json").write_text(json.dumps(metadata, indent=2))
        for rid in list(engine.scheduler.requests):
            engine.abort_request(rid)
        engine.model_runner.synchronize()
        assert engine.cache_manager.num_used_blocks == 0
        del prof, averages, engine
        print(variant, "saved", metadata["trace_bytes"], "bytes", flush=True)


if __name__ == "__main__":
    main()
