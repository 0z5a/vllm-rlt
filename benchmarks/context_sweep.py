"""Decode-only context/concurrency sweep from a real, reproducible prefill.

Requests repeat one prompt but own independent physical KV pages. Prefill is
computed once, then copied to those pages; only the immutable prompt prefix is
reused between trials. No fabricated KV or uncomputed hidden states are used.
"""

import argparse
import gc
import hashlib
import json
import math
import os
import random
import time
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path

import torch

from benchmarks.runtime_baseline import VARIANTS, percentile
from vllm_lt import CacheConfig, ExitConfig, SamplingParams, SchedulerConfig
from vllm_lt.core.scheduler import Scheduler
from vllm_lt.engine.llm_engine import LLMEngine
from vllm_lt.kernels.flash_attention import FlashPagedAttention
from vllm_lt.kernels.paged_attention import triton_paged_attention
from vllm_lt.models import OuroConfig, OuroForCausalLM
from vllm_lt.request import Request, Stage
from vllm_lt.worker.model_runner import ModelRunner


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def build_prefix(engine, prompt, concurrency, output_length):
    """Run actual prefill once and replicate it into independently owned pages."""
    params = SamplingParams(max_tokens=output_length, ignore_eos=True)
    engine.add_request("0", prompt, params)
    source = engine.scheduler.requests["0"]
    synchronize(engine.cache_manager.device)
    start = time.perf_counter()
    last_report = start
    while source.num_prefilled_tokens < len(prompt):
        assert not engine.step(), "prefill must not sample before the checkpoint"
        now = time.perf_counter()
        if now - last_report >= 10:
            print(
                json.dumps(
                    dict(
                        phase="prefill",
                        tokens=source.num_prefilled_tokens,
                        total=len(prompt),
                        seconds=now - start,
                    )
                ),
                flush=True,
            )
            last_report = now
    synchronize(engine.cache_manager.device)
    prefill_seconds = time.perf_counter() - start
    hidden = source.hidden_state.clone()
    for i in range(1, concurrency):
        engine.add_request(str(i), prompt, params)
    while engine.scheduler.queues[Stage.WAITING]:
        before = len(engine.scheduler.queues[Stage.WAITING])
        engine.scheduler._admit()
        if len(engine.scheduler.queues[Stage.WAITING]) == before:
            raise RuntimeError("KV capacity cannot admit the requested concurrency")
    cache = engine.cache_manager
    allocations = dict(cache._allocations)
    original = allocations["0"]
    copy_start = time.perf_counter()
    for rid, allocation in allocations.items():
        if rid == "0":
            continue
        for source_table, destination in zip(original.block_tables, allocation.block_tables):
            for table in (source_table, destination):
                assert table == tuple(range(table[0], table[0] + len(table)))
            src = slice(source_table[0], source_table[-1] + 1)
            dst = slice(destination[0], destination[-1] + 1)
            cache.key_cache[dst].copy_(cache.key_cache[src])
            cache.value_cache[dst].copy_(cache.value_cache[src])
    synchronize(cache.device)
    return dict(
        hidden=hidden,
        allocations=allocations,
        free_blocks=list(cache._free_blocks),
        prompt=list(prompt),
        output_length=output_length,
        prefill_seconds=prefill_seconds,
        replication_seconds=time.perf_counter() - copy_start,
    )


def restore_prefix(
    engine,
    checkpoint,
    concurrency,
    variant,
    policy,
    batch_tokens,
    *,
    reuse_runner=False,
    buffer_mode="dynamic",
):
    """Discard only decode suffixes; attention lengths exclude old suffix data."""
    engine.model_runner.synchronize()
    cache = engine.cache_manager
    cache._allocations = dict(checkpoint["allocations"])
    cache._free_blocks = list(checkpoint["free_blocks"])
    engine._pending_coda.clear()
    engine._signals.clear()
    engine._inflight.clear()
    engine._overlap_boundary = False
    engine.last_schedule = None
    config = SchedulerConfig(
        max_num_seqs=concurrency,
        max_num_batched_tokens=batch_tokens,
        prefill_chunk_size=batch_tokens,
    )
    engine.scheduler = Scheduler(config, cache)
    engine.execution_config = replace(
        VARIANTS[variant],
        static_buffers=buffer_mode != "dynamic",
        pad_to_power_of_two=buffer_mode == "static-pad",
    )
    if not reuse_runner:
        engine.model_runner = ModelRunner(
            engine.model,
            cache,
            exit_config=engine.exit_config,
            execution_config=engine.execution_config,
            scheduler_config=config,
        )
    for i in range(concurrency):
        rid = str(i)
        allocation = cache._allocations[rid]
        for plane in allocation.written:
            for written in plane:
                written.prefix = len(checkpoint["prompt"])
                written.pending.clear()
        depth = 4 if policy == "fixed4" else 2 + i % 3
        params = SamplingParams(
            max_tokens=checkpoint["output_length"],
            ignore_eos=True,
            min_loops=depth - 1 if depth < 4 else 2,
            exit_threshold=0 if depth < 4 else 1,
        )
        request = Request(
            rid,
            list(checkpoint["prompt"]),
            params,
            num_prefilled_tokens=len(checkpoint["prompt"]),
            loops_done=4,
            hidden_state=checkpoint["hidden"].clone(),
        )
        engine.scheduler.requests[rid] = request
        engine.scheduler.enqueue(request, Stage.CODA)
    # Hidden-state clones were made on the default stream, after runner init.
    synchronize(cache.device)


def measure(engine, concurrency, output_length, policy):
    runner = engine.model_runner
    original_execute, original_prepare = runner._execute, runner.prepare
    batches, overlap = Counter(), Counter()

    def execute(batch, prepared=None):
        batches[(batch.stage.value, len(batch.items))] += 1
        return original_execute(batch, prepared)

    def prepare(batch):
        previous = tuple(runner.submission_events)
        if batch.stage == Stage.RECURRENT:
            overlap["preparations"] += 1
            overlap["gpu_busy_before_prepare"] += any(not e.query() for e in previous)
        result = original_prepare(batch)
        if batch.stage == Stage.RECURRENT:
            overlap["gpu_busy_after_prepare"] += any(not e.query() for e in previous)
        return result

    runner._execute, runner.prepare = execute, prepare
    synchronize(runner.device)
    if runner.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(runner.device)
    outputs, last_seen, delivered, intervals = {}, {}, Counter(), []
    start = time.perf_counter()
    try:
        while engine.has_unfinished_requests():
            result = engine.step()
            now = time.perf_counter()
            for out in result:
                rid = out.request_id
                delivered[rid] += 1
                if rid in last_seen:
                    intervals.append(now - last_seen[rid])
                last_seen[rid] = now
                if out.finished:
                    depth = 4 if policy == "fixed4" else 2 + int(rid) % 3
                    assert len(out.token_ids) == output_length == delivered[rid]
                    assert out.exit_depths == [4] + [depth] * (output_length - 1)
                    assert out.finish_reason == "length"
                    outputs[rid] = dict(token_ids=out.token_ids, exit_depths=out.exit_depths)
        runner.synchronize()
        elapsed = time.perf_counter() - start
    finally:
        runner._execute, runner.prepare = original_execute, original_prepare
    assert len(outputs) == concurrency
    assert not runner.state_slots and not engine._pending_coda
    assert not any(str(i) in engine.cache_manager._allocations for i in range(concurrency))
    core_calls = sum(n for (stage, _), n in batches.items() if stage == "recurrent")
    core_rows = sum(size * n for (stage, size), n in batches.items() if stage == "recurrent")
    summary = dict(
        elapsed_seconds=elapsed,
        requests=concurrency,
        output_tokens=concurrency * output_length,
        output_tokens_per_second=concurrency * output_length / elapsed,
        decode_tokens_per_second=concurrency * (output_length - 1) / elapsed,
        recurrent_calls=core_calls,
        recurrent_rows=core_rows,
        mean_recurrent_batch=core_rows / core_calls,
        overlap=dict(overlap),
        batches=[
            dict(stage=stage, rows=size, calls=n) for (stage, size), n in sorted(batches.items())
        ],
        itl_seconds=dict(p50=percentile(intervals, 0.5), p95=percentile(intervals, 0.95)),
    )
    if runner.device.type == "cuda":
        summary["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(runner.device)
        summary["peak_reserved_bytes"] = torch.cuda.max_memory_reserved(runner.device)
    return outputs, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--context", type=int, required=True)
    parser.add_argument("--concurrencies", nargs="+", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--output-length", type=int, default=128)
    parser.add_argument("--max-num-batched-tokens", type=int, default=512)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--mixed-at-max", action="store_true")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--toy", action="store_true")
    parser.add_argument(
        "--attention-backend",
        choices=["torch", "triton", "flash_attn", "flash_attn_2", "flash_attn_3", "flash_attn_4"],
    )
    parser.add_argument(
        "--compare-attention-backends",
        action="store_true",
        help="Paired Triton/FlashAttention decode from the same real Triton prefill.",
    )
    parser.add_argument(
        "--buffer-modes",
        nargs="+",
        choices=["dynamic", "static", "static-pad"],
        default=["dynamic"],
        help="Compare buffer reuse and power-of-two padding; CUDA graphs remain off.",
    )
    args = parser.parse_args()
    if (
        min(args.context, args.output_length, args.repeats, args.block_size, *args.concurrencies)
        < 1
    ):
        parser.error("counts must be positive")
    if args.output_length < 2 or max(args.concurrencies) > args.max_num_batched_tokens:
        parser.error("output-length must be >= 2 and batch-token budget must cover concurrency")
    if args.compare_attention_backends and (
        args.device != "cuda" or args.attention_backend not in (None, "triton")
    ):
        parser.error("attention A/B requires CUDA and a Triton prefill backend")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    if args.toy:
        torch.manual_seed(123)
        model = OuroForCausalLM(OuroConfig.tiny()).to(args.device)
        seed_tokens = [2, 3, 4, 5]
    else:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
        seed_tokens = tokenizer.encode(
            "Explain matrix multiplication with a worked example. ", add_special_tokens=False
        )
        model = OuroForCausalLM.from_pretrained(
            args.model, device=args.device, dtype=torch.bfloat16
        )
    config = model.config
    if args.context > config.max_position_embeddings:
        raise ValueError("context exceeds the checkpoint's positional limit")
    prompt_length = min(args.context, config.max_position_embeddings - args.output_length + 1)
    if prompt_length < 1:
        raise ValueError("output does not fit the checkpoint context")
    prompt = (seed_tokens * math.ceil(prompt_length / len(seed_tokens)))[:prompt_length]
    max_concurrency = max(args.concurrencies)
    capacity = prompt_length + args.output_length - 1
    blocks = math.ceil(capacity / args.block_size) * config.total_ut_steps * max_concurrency
    block_bytes = (
        2
        * config.num_hidden_layers
        * args.block_size
        * config.num_key_value_heads
        * config.head_dim
        * next(model.parameters()).element_size()
    )
    required_bytes = blocks * block_bytes
    if args.device == "cuda":
        free, total = torch.cuda.mem_get_info()
        if required_bytes + 8 * 2**30 > min(free, 0.9 * total):
            raise RuntimeError(
                f"KV capacity {required_bytes / 2**30:.2f} GiB exceeds memory budget"
            )
    engine = LLMEngine(
        model,
        cache_config=CacheConfig(num_blocks=blocks, block_size=args.block_size),
        scheduler_config=SchedulerConfig(
            max_num_seqs=max_concurrency,
            max_num_batched_tokens=args.max_num_batched_tokens,
            prefill_chunk_size=args.max_num_batched_tokens,
        ),
        exit_config=ExitConfig("ouro_delayed"),
        attention_backend=args.attention_backend
        or ("triton" if args.device == "cuda" else "torch"),
    )
    manifest = dict(
        arguments={**vars(args), "output": str(args.output)},
        prompt_length=prompt_length,
        final_kv_length=capacity,
        num_blocks=blocks,
        kv_bytes=required_bytes,
        attention=engine.cache_manager.attention_info,
        torch_version=torch.__version__,
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        cpu_affinity=sorted(os.sched_getaffinity(0)),
        prompt_sha256=hashlib.sha256(json.dumps(prompt).encode()).hexdigest(),
        protocol=(
            "decode only; real prefill copied to independent KV pages; homogeneous prompt; "
            "prefix reset before every warmup/measurement; buffer modes in arguments; no graphs"
        ),
    )
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest), flush=True)
    checkpoint = build_prefix(engine, prompt, max_concurrency, args.output_length)
    manifest.update(
        prefill_seconds=checkpoint["prefill_seconds"],
        replication_seconds=checkpoint["replication_seconds"],
    )
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    attention_options = {
        engine.cache_manager.backend: (
            engine.cache_manager.attention,
            engine.cache_manager.attention_info,
        )
    }
    if args.compare_attention_backends:
        flash = FlashPagedAttention(
            engine.cache_manager.device,
            next(model.parameters()).dtype,
            config.head_dim,
            args.block_size,
        )
        attention_options = {
            "triton": (triton_paged_attention, {"backend": "triton"}),
            "flash_attn": (flash, flash.info),
        }
    manifest["decode_attention_options"] = {
        key: value[1] for key, value in attention_options.items()
    }
    manifest["prefill_attention"] = manifest["attention"]
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    summaries = []
    cases = [(c, "fixed4") for c in sorted(set(args.concurrencies))]
    if args.mixed_at_max:
        cases.append((max_concurrency, "mixed234"))
    for concurrency, policy in cases:
        references = {}
        cross_reference = None
        for repeat in range(args.repeats):
            order = [
                (backend, mode, variant)
                for backend in attention_options
                for mode in args.buffer_modes
                for variant in ["S", "AS", "AM"]
            ]
            random.Random(1729 + repeat).shuffle(order)
            for attention_backend, buffer_mode, variant in order:
                engine.model_runner.synchronize()
                cache = engine.cache_manager
                cache.backend = attention_backend
                cache.attention, cache.attention_info = attention_options[attention_backend]
                gc.collect()
                restore_prefix(
                    engine,
                    checkpoint,
                    concurrency,
                    variant,
                    policy,
                    args.max_num_batched_tokens,
                    buffer_mode=buffer_mode,
                )
                measure(engine, concurrency, args.output_length, policy)  # Full-length warmup.
                restore_prefix(
                    engine,
                    checkpoint,
                    concurrency,
                    variant,
                    policy,
                    args.max_num_batched_tokens,
                    reuse_runner=True,
                    buffer_mode=buffer_mode,
                )
                outputs, summary = measure(engine, concurrency, args.output_length, policy)
                if attention_backend not in references:
                    references[attention_backend] = outputs
                if cross_reference is None:
                    cross_reference = outputs
                differences = [
                    rid for rid in outputs if outputs[rid] != references[attention_backend][rid]
                ]
                cross_differences = [rid for rid in outputs if outputs[rid] != cross_reference[rid]]
                summary.update(
                    context=args.context,
                    prompt_length=prompt_length,
                    concurrency=concurrency,
                    policy=policy,
                    variant=variant,
                    buffer_mode=buffer_mode,
                    repeat=repeat + 1,
                    output_disagreements=differences,
                    cross_backend_disagreements=cross_differences,
                    attention=cache.attention_info,
                    attention_backend=attention_backend,
                    execution=asdict(engine.execution_config),
                )
                name = f"{policy}-c{concurrency}-{variant}-r{repeat + 1}"
                if args.compare_attention_backends:
                    name = f"{attention_backend}-{name}"
                if buffer_mode != "dynamic":
                    name = f"{buffer_mode}-{name}"
                (args.output / f"{name}.outputs.json").write_text(json.dumps(outputs))
                summaries.append(summary)
                (args.output / "results.json").write_text(json.dumps(summaries, indent=2))
                print(json.dumps(summary), flush=True)
    for rid in list(engine.cache_manager._allocations):
        engine.cache_manager.free(rid)
    assert engine.cache_manager.num_used_blocks == 0
    if any(row["output_disagreements"] for row in summaries):
        raise RuntimeError(
            "Output disagreements recorded; inspect results before accepting timings"
        )


if __name__ == "__main__":
    main()
