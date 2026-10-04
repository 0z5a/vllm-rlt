"""Paired Tiny Huginn serving measurements; CPU storage is not GPU peak memory."""

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import torch

from vllm_rlt import CacheConfig, SamplingParams, SchedulerConfig
from vllm_rlt.engine.llm_engine import LLMEngine
from vllm_rlt.models.huginn import HuginnConfig, HuginnForCausalLM


def make_engine(model, loops: int, concurrency: int, grouped: bool) -> LLMEngine:
    return LLMEngine(
        model,
        cache_config=CacheConfig(
            num_blocks=loops * concurrency * 5,
            block_size=2,
            stage_aware=grouped,
            incremental_allocation=True,
        ),
        scheduler_config=SchedulerConfig(
            max_num_seqs=concurrency, max_num_batched_tokens=64, prefill_chunk_size=2
        ),
    )


@torch.inference_mode()
def serve(engine: LLMEngine, loops: int, concurrency: int, seed: int) -> tuple:
    """Time admission through final score export for a burst of 2C requests."""
    started = time.perf_counter_ns()
    count = 2 * concurrency
    for index in range(count):
        length = 2 + index % 5
        engine.add_request(
            str(index),
            [(index + position) % 13 for position in range(length)],
            SamplingParams(
                max_tokens=4,
                min_loops=loops,
                max_loops=loops,
                ignore_eos=True,
                temperature=0.8,
                seed=seed * 1000 + index,
                latent_seed=seed * 2000 + index,
                logprobs=0,
                logprobs_mode="processed",
            ),
        )
    completed = {}
    steps = 0
    peak_live, peak_blocks = 0, 0
    latencies = []
    for steps in range(1, 10001):
        if not engine.has_unfinished_requests():
            break
        for output in engine.step(final_only=True):
            assert output.finished and output.request_id not in completed
            completed[output.request_id] = {
                "tokens": output.token_ids,
                "depths": output.exit_depths,
                "scores": output.log_probs,
            }
            latencies.append((time.perf_counter_ns() - started) / 1e9)
        peak_live = max(peak_live, len(engine.cache_manager._allocations))
        peak_blocks = max(peak_blocks, engine.cache_manager.num_used_blocks)
    elapsed = (time.perf_counter_ns() - started) / 1e9
    assert not engine.has_unfinished_requests(), "finite scheduler budget exceeded"
    assert len(completed) == count
    assert all(len(row["tokens"]) == 4 for row in completed.values())
    assert engine.cache_manager.num_used_blocks == 0
    latencies.sort()
    return (
        elapsed,
        completed,
        {
            "scheduler_steps": steps - 1,
            "peak_live_requests": peak_live,
            "peak_used_blocks": peak_blocks,
            "completion_latency_seconds": latencies,
        },
    )


def run(args) -> None:
    torch.set_num_threads(1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        for loops, concurrency in ((3, 1), (3, 16), (3, 64), (32, 1), (32, 16)):
            torch.manual_seed(args.seed)
            model = HuginnForCausalLM(
                HuginnConfig(
                    n_embd=16,
                    n_heads=2,
                    n_layers=8,
                    n_layers_in_prelude=2,
                    n_layers_in_recurrent_block=4,
                    n_layers_in_coda=2,
                    intermediate_size=32,
                    mean_recurrence=loops,
                    block_size=64,
                    vocab_size=13,
                    padded_vocab_size=13,
                    bos_token_id=0,
                    eos_token_id=1,
                    pad_token_id=0,
                )
            ).eval()
            engines = {
                arm: make_engine(model, loops, concurrency, arm == "stage")
                for arm in ("rectangular", "stage")
            }
            _, expected, _ = serve(engines["rectangular"], loops, concurrency, args.seed)
            _, warm, _ = serve(engines["stage"], loops, concurrency, args.seed)
            assert warm == expected
            encoded = json.dumps(expected, sort_keys=True).encode()
            digest = hashlib.sha256(encoded).hexdigest()
            args.output.with_name(f"outputs-s{args.seed}-r{loops}-c{concurrency}.json").write_bytes(
                encoded + b"\n"
            )
            for block in range(args.blocks):
                for order, arm in enumerate(("rectangular", "stage", "stage", "rectangular")):
                    engine = engines[arm]
                    elapsed, actual, metrics = serve(engine, loops, concurrency, args.seed)
                    assert actual == expected, (args.seed, loops, concurrency, block, arm)
                    storage = sum(
                        keys.numel() + values.numel()
                        for keys, values, _ in engine.cache_manager.storage_pages([])
                    )
                    row = {
                        "seed": args.seed,
                        "loops": loops,
                        "concurrency": concurrency,
                        "requests": 2 * concurrency,
                        "response_tokens": 8 * concurrency,
                        "block": block,
                        "order": order,
                        "arm": arm,
                        "wall_seconds": elapsed,
                        "tokens_per_second": 8 * concurrency / elapsed,
                        "kv_tensor_bytes": storage * next(model.parameters()).element_size(),
                        **metrics,
                        "outputs_sha256": digest,
                        "exact_outputs": actual == expected,
                        "used_blocks_after": 0,
                        "torch": torch.__version__,
                        "python": platform.python_version(),
                        "device": "cpu",
                        "threads": 1,
                    }
                    stream.write(json.dumps(row, sort_keys=True) + "\n")
                    stream.flush()
            for engine in engines.values():
                engine.close()
            print(
                f"seed={args.seed} R={loops} C={concurrency} exact outputs; blocks={args.blocks}",
                flush=True,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--blocks", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
