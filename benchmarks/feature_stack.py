"""Real-gate feature stack: independent prefixes and per-request latency logs."""

import argparse
import gc
import hashlib
import json
import math
import os
import random
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import torch

from benchmarks.context_sweep import synchronize
from benchmarks.runtime_baseline import percentile
from vllm_lt import CacheConfig, ExecutionConfig, ExitConfig, SamplingParams, SchedulerConfig
from vllm_lt.core.scheduler import Scheduler
from vllm_lt.engine.llm_engine import LLMEngine
from vllm_lt.kernels.flash_attention import FlashPagedAttention
from vllm_lt.kernels.paged_attention import triton_paged_attention
from vllm_lt.models import OuroForCausalLM
from vllm_lt.request import Request, Stage
from vllm_lt.worker.model_runner import ModelRunner

STAGES = {
    "P0": ("triton", "ouro", False, False, False, False, False),
    "P1": ("flash_attn", "ouro", False, False, False, False, False),
    "P2": ("flash_attn", "ouro", False, False, False, False, True),
    "P3": ("flash_attn", "ouro_delayed", False, False, False, False, True),
    "P4": ("flash_attn", "ouro_delayed", True, False, False, False, True),
    "P5": ("flash_attn", "ouro_delayed", True, True, False, False, True),
    "P6": ("flash_attn", "ouro_delayed", True, True, True, False, True),
    "P7": ("flash_attn", "ouro_delayed", True, True, True, True, True),
}


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temporary.replace(path)


def quantiles(values):
    return {f"p{q}": percentile(values, q / 100) for q in (50, 95, 99)}


def latency_metrics(records, scope):
    """Quantile populations: requests for TTFT/TPOT/E2E, intervals for ITL."""
    ttft, tpot, itl, e2e, queue = [], [], [], [], []
    for row in records:
        stamps = row["token_times_seconds"]
        if len(stamps) > 1:
            tpot.append((stamps[-1] - stamps[0]) / (len(stamps) - 1))
            itl.extend(b - a for a, b in zip(stamps, stamps[1:]))
        if scope == "e2e":
            ttft.append(stamps[0] - row["start_seconds"])
            e2e.append(row["end_seconds"] - row["start_seconds"])
            queue.append(row["admitted_seconds"] - row["start_seconds"])
    return dict(
        ttft_seconds=quantiles(ttft) if scope == "e2e" else None,
        tpot_seconds=quantiles(tpot),
        itl_seconds=quantiles(itl),
        e2e_seconds=quantiles(e2e) if scope == "e2e" else None,
        queue_seconds=quantiles(queue) if scope == "e2e" else None,
        latency_request_samples=len(records),
        itl_samples=len(itl),
        tail_latency_caveat="Request percentiles have few samples; not an SLO estimate."
        if len(records) < 100
        else None,
    )


def configure(engine, stage, concurrency, batch_tokens, attention_options, *, reuse=False):
    engine.model_runner.synchronize()
    backend, mode, asynchronous, multi, static, padding, _ = STAGES[stage]
    engine.exit_config = ExitConfig(mode)
    engine.execution_config = ExecutionConfig(
        async_scheduling=asynchronous,
        multi_stream=multi,
        static_buffers=static,
        pad_to_power_of_two=padding,
    )
    config = SchedulerConfig(
        max_num_seqs=concurrency,
        max_num_batched_tokens=batch_tokens,
        prefill_chunk_size=batch_tokens,
    )
    engine.scheduler = Scheduler(config, engine.cache_manager)
    engine._pending_coda.clear()
    engine._signals.clear()
    engine._inflight.clear()
    engine._overlap_boundary = False
    engine.last_schedule = None
    cache = engine.cache_manager
    cache.backend = backend
    cache.attention, cache.attention_info = attention_options[backend]
    if not reuse:
        engine.model_runner = ModelRunner(
            engine.model,
            cache,
            exit_config=engine.exit_config,
            execution_config=engine.execution_config,
            scheduler_config=config,
        )
    else:
        assert engine.model_runner.execution_config == engine.execution_config
        assert engine.model_runner.exit_config == engine.exit_config


def params_for(stage, output_length, threshold):
    return SamplingParams(
        max_tokens=output_length,
        min_loops=2,
        max_loops=4,
        exit_threshold=threshold if STAGES[stage][-1] else 1.0,
        temperature=0,
        ignore_eos=True,
    )


@torch.inference_mode()
def build_independent_prefixes(engine, prompts, output_length, progress=None):
    """Execute actual prefill for every prompt, without sampling or KV replication."""
    for i, prompt in enumerate(prompts):
        engine.add_request(str(i), prompt, params_for("P0", output_length, 1))
    while engine.scheduler.queues[Stage.WAITING]:
        before = len(engine.scheduler.queues[Stage.WAITING])
        engine.scheduler._admit()
        if len(engine.scheduler.queues[Stage.WAITING]) == before:
            raise RuntimeError("prefix requests cannot be admitted within KV capacity")
    start, last = time.perf_counter(), time.perf_counter()
    while engine.scheduler.queues[Stage.PREFILL]:
        batch = engine.scheduler._take(Stage.PREFILL)
        result = engine.model_runner.execute(batch)
        assert not engine._update(batch, result)
        if progress and time.perf_counter() - last >= 20:
            progress(
                dict(
                    phase="prefix",
                    seconds=time.perf_counter() - start,
                    prefilled=sum(
                        r.num_prefilled_tokens for r in engine.scheduler.requests.values()
                    ),
                )
            )
            last = time.perf_counter()
    synchronize(engine.cache_manager.device)
    hidden = {}
    for i, prompt in enumerate(prompts):
        request = engine.scheduler.requests[str(i)]
        assert request.num_prefilled_tokens == len(prompt) and request.stage == Stage.CODA
        hidden[str(i)] = request.hidden_state.clone()
    return dict(
        prompts=prompts,
        hidden=hidden,
        allocations=dict(engine.cache_manager._allocations),
        free_blocks=list(engine.cache_manager._free_blocks),
        seconds=time.perf_counter() - start,
    )


def restore_independent_prefixes(engine, checkpoint, concurrency, params):
    cache = engine.cache_manager
    cache._allocations = dict(checkpoint["allocations"])
    cache._free_blocks = list(checkpoint["free_blocks"])
    for i in range(concurrency):
        rid = str(i)
        prompt = checkpoint["prompts"][i]
        for plane in cache._allocations[rid].written:
            for written in plane:
                written.prefix = len(prompt)
                written.pending.clear()
        request = Request(
            rid,
            list(prompt),
            params,
            num_prefilled_tokens=len(prompt),
            loops_done=4,
            hidden_state=checkpoint["hidden"][rid].clone(),
        )
        engine.scheduler.requests[rid] = request
        engine.scheduler.enqueue(request, Stage.CODA)
    synchronize(cache.device)


def run_trial(engine, prompts, concurrency, params, scope, requests, progress=None):
    runner, cache = engine.model_runner, engine.cache_manager
    original_execute, original_prepare, original_allocate = (
        runner._execute,
        runner.prepare,
        cache.allocate,
    )
    batches, overlap = Counter(), Counter()
    records, active = [], {}
    issued = 0
    start = None

    def execute(batch, prepared=None):
        batches[(batch.stage.value, len(batch.items))] += 1
        if batch.stage == Stage.PREFILL:
            now = time.perf_counter() - start
            for item in batch.items:
                row = active[item.request.request_id]
                row.setdefault("prefill_started_seconds", now)
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

    def allocate(rid, capacity):
        result = original_allocate(rid, capacity)
        if result and rid in active:
            active[rid].setdefault("admitted_seconds", time.perf_counter() - start)
        return result

    def submit():
        nonlocal issued
        rid = str(issued)
        active[rid] = dict(
            request_id=rid,
            prompt_index=issued,
            start_seconds=time.perf_counter() - start,
            token_times_seconds=[],
        )
        engine.add_request(rid, prompts[issued], params)
        issued += 1

    runner._execute, runner.prepare, cache.allocate = execute, prepare, allocate
    runner.synchronize()
    if runner.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(runner.device)
    start, last = time.perf_counter(), time.perf_counter()
    try:
        if scope == "e2e":
            for _ in range(min(concurrency, requests)):
                submit()
        else:
            issued = requests = concurrency
            active = {
                str(i): dict(
                    request_id=str(i), prompt_index=i, start_seconds=None, token_times_seconds=[]
                )
                for i in range(concurrency)
            }
        while engine.has_unfinished_requests():
            outputs = engine.step()
            now = time.perf_counter() - start
            for out in outputs:
                row = active[out.request_id]
                assert len(out.token_ids) == len(row["token_times_seconds"]) + 1
                row["token_times_seconds"].append(now)
                if out.finished:
                    assert len(out.token_ids) == params.max_tokens
                    assert out.finish_reason == "length"
                    assert out.exit_depths[0] == 4
                    assert all(2 <= d <= 4 for d in out.exit_depths[1:])
                    row.update(
                        end_seconds=now,
                        token_ids=out.token_ids,
                        exit_depths=out.exit_depths,
                        finish_reason=out.finish_reason,
                    )
                    records.append(row)
                    del active[out.request_id]
                    if scope == "e2e" and issued < requests:
                        submit()
            if progress and time.perf_counter() - last >= 20:
                progress(
                    dict(
                        phase="generating",
                        seconds=now,
                        issued=issued,
                        completed=len(records),
                        active=len(active),
                        prefilled=sum(
                            r.num_prefilled_tokens for r in engine.scheduler.requests.values()
                        ),
                    )
                )
                last = time.perf_counter()
        runner.synchronize()
        elapsed = time.perf_counter() - start
    except Exception as error:
        error.partial_records = dict(completed=records, active=list(active.values()))
        raise
    finally:
        runner._execute, runner.prepare, cache.allocate = (
            original_execute,
            original_prepare,
            original_allocate,
        )
    assert not active and len(records) == requests
    assert not runner.state_slots and not engine._pending_coda
    assert all(str(i) not in cache._allocations for i in range(requests))
    depths = [d for row in records for d in row["exit_depths"][1:]]
    counts = Counter(depths)
    core_calls = sum(n for (stage, _), n in batches.items() if stage == "recurrent")
    core_rows = sum(n * size for (stage, size), n in batches.items() if stage == "recurrent")
    assert core_rows == sum(depths)
    summary = dict(
        scope=scope,
        elapsed_seconds=elapsed,
        requests=requests,
        output_tokens=requests * params.max_tokens,
        output_tokens_per_second=requests * params.max_tokens / elapsed,
        requests_per_second=requests / elapsed,
        completion_rate=1.0,
        error_count=0,
        mean_decode_depth=sum(depths) / len(depths),
        exit_depth_counts=dict(counts),
        recurrent_rows=core_rows,
        recurrent_calls=core_calls,
        mean_recurrent_batch=core_rows / core_calls,
        overlap=dict(overlap),
        batches=[dict(stage=s, rows=size, calls=n) for (s, size), n in sorted(batches.items())],
        **latency_metrics(records, scope),
    )
    if runner.device.type == "cuda":
        summary.update(
            peak_allocated_bytes=torch.cuda.max_memory_allocated(runner.device),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(runner.device),
        )
    return records, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scope", choices=["decode", "e2e"], required=True)
    parser.add_argument("--concurrencies", nargs="+", type=int, required=True)
    parser.add_argument("--stages", nargs="+", choices=list(STAGES), default=list(STAGES))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output-length", type=int, default=128)
    parser.add_argument("--batch-tokens", type=int, default=2048)
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument("--waves", type=int, default=2)
    args = parser.parse_args()
    if (
        min(args.repeats, args.waves, args.batch_tokens, *args.concurrencies) < 1
        or args.output_length < 2
    ):
        parser.error("positive counts and output-length >= 2 required")
    if not 0 <= args.threshold < 1 or max(args.concurrencies) > args.batch_tokens:
        parser.error("invalid threshold or insufficient token budget")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    workload = json.loads(args.workload.read_text())
    prompts = [r["token_ids"] for r in workload["requests"]]
    maximum = max(args.concurrencies)
    assert len(prompts) >= maximum * (args.waves if args.scope == "e2e" else 1)
    model = OuroForCausalLM.from_pretrained(args.model, device="cuda", dtype=torch.bfloat16)
    config = model.config
    assert max(map(len, prompts)) + args.output_length - 1 <= config.max_position_embeddings
    blocks = (
        math.ceil((max(map(len, prompts)) + args.output_length - 1) / 16)
        * config.total_ut_steps
        * maximum
    )
    block_bytes = (
        2 * config.num_hidden_layers * 16 * config.num_key_value_heads * config.head_dim * 2
    )
    free, total = torch.cuda.mem_get_info()
    if blocks * block_bytes + 8 * 2**30 > min(free, 0.9 * total):
        raise RuntimeError("KV pool plus 8 GiB headroom does not fit")
    engine = LLMEngine(
        model,
        cache_config=CacheConfig(num_blocks=blocks, block_size=16),
        scheduler_config=SchedulerConfig(
            max_num_seqs=maximum,
            max_num_batched_tokens=args.batch_tokens,
            prefill_chunk_size=args.batch_tokens,
        ),
        exit_config=ExitConfig("ouro"),
        attention_backend="triton",
    )
    flash = FlashPagedAttention(engine.cache_manager.device, torch.bfloat16, config.head_dim, 16)
    attention_options = {
        "triton": (triton_paged_attention, {"backend": "triton"}),
        "flash_attn": (flash, flash.info),
    }
    manifest = dict(
        arguments={**vars(args), "workload": str(args.workload), "output": str(args.output)},
        workload_sha256=hashlib.sha256(args.workload.read_bytes()).hexdigest(),
        context=workload["context"],
        actual_prompt_lengths=sorted(set(map(len, prompts))),
        torch_version=torch.__version__,
        num_blocks=blocks,
        kv_bytes=blocks * block_bytes,
        attention=flash.info,
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        cpu_affinity=sorted(os.sched_getaffinity(0)),
        timing="engine host delivery; pretokenized; no HTTP; no per-token CUDA sync",
        request_count="C for decode, waves*C for e2e",
        gate_threshold=args.threshold,
    )
    atomic_json(args.output / "manifest.json", manifest)

    def progress(value):
        atomic_json(args.output / "progress.json", value)
        print(json.dumps(value), flush=True)

    checkpoint = None
    if args.scope == "decode":
        progress(dict(phase="prefix", context=workload["context"]))
        checkpoint = build_independent_prefixes(
            engine, prompts[:maximum], args.output_length, progress
        )
        manifest["prefix_seconds"] = checkpoint["seconds"]
        manifest["prefix_backend"] = "triton"
        atomic_json(args.output / "manifest.json", manifest)
    summaries = []
    for concurrency in args.concurrencies:
        references = {}
        for repeat in range(args.repeats):
            order = list(args.stages)
            random.Random(20260915 + repeat).shuffle(order)
            for stage in order:
                name = f"c{concurrency}-{stage}-r{repeat + 1}"
                policy = "fixed4" if not STAGES[stage][-1] else STAGES[stage][1]
                params = params_for(stage, args.output_length, args.threshold)
                for warm in (True, False):
                    configure(
                        engine,
                        stage,
                        concurrency,
                        args.batch_tokens,
                        attention_options,
                        reuse=not warm,
                    )
                    if checkpoint is not None:
                        restore_independent_prefixes(engine, checkpoint, concurrency, params)
                    else:
                        assert not engine.cache_manager._allocations
                        engine.cache_manager._free_blocks = list(reversed(range(blocks)))
                    gc.collect()
                    progress(
                        dict(
                            phase="warmup" if warm else "measurement",
                            scope=args.scope,
                            context=workload["context"],
                            concurrency=concurrency,
                            stage=stage,
                            repeat=repeat + 1,
                        )
                    )
                    try:
                        records, summary = run_trial(
                            engine,
                            prompts,
                            concurrency,
                            params,
                            args.scope,
                            concurrency * args.waves,
                            progress,
                        )
                    except Exception as error:
                        atomic_json(
                            args.output / f"{name}.failure.json",
                            dict(
                                error=repr(error),
                                partial=getattr(error, "partial_records", None),
                                warmup=warm,
                            ),
                        )
                        raise
                    if warm:
                        atomic_json(args.output / f"{name}.warmup.json", summary)
                        continue
                    outputs = {
                        r["request_id"]: dict(
                            token_ids=r["token_ids"], exit_depths=r["exit_depths"]
                        )
                        for r in records
                    }
                    references.setdefault(policy, outputs)
                    differences = [
                        rid for rid, out in outputs.items() if out != references[policy][rid]
                    ]
                    summary.update(
                        context=workload["context"],
                        concurrency=concurrency,
                        stage=stage,
                        repeat=repeat + 1,
                        policy=policy,
                        threshold=params.exit_threshold,
                        attention=engine.cache_manager.attention_info,
                        execution=asdict(engine.execution_config),
                        output_disagreements=differences,
                    )
                    atomic_json(args.output / f"{name}.requests.json", records)
                    summaries.append(summary)
                    atomic_json(args.output / "results.json", summaries)
                    print(json.dumps(summary), flush=True)
    engine.model_runner.synchronize()
    for rid in list(engine.cache_manager._allocations):
        engine.cache_manager.free(rid)
    assert engine.cache_manager.num_used_blocks == 0
    atomic_json(
        args.output / "complete.json",
        dict(
            measurements=len(summaries),
            expected=len(args.concurrencies) * len(args.stages) * args.repeats,
            disagreement_trials=sum(bool(r["output_disagreements"]) for r in summaries),
        ),
    )


if __name__ == "__main__":
    main()
