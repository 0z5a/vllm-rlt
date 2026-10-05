"""Measure existing prefix reuse before adding group admission or fork logic.

Tiny CPU native generation only. Warm-cache time excludes a separately recorded
primer; neither result is an official-model or full-RL performance claim.
"""

import argparse
import json
import platform
import time
from pathlib import Path

import torch

from vllm_rlt import CacheConfig, SamplingParams, SchedulerConfig
from vllm_rlt.engine.llm_engine import LLMEngine
from vllm_rlt.models import NanbeigeConfig, NanbeigeForCausalLM, OuroConfig, OuroForCausalLM
from vllm_rlt.request import RequestOutput, Stage


def make_model(family: str, seed: int):
    torch.manual_seed(seed)
    common = dict(
        vocab_size=17,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=128,
        bos_token_id=0,
        eos_token_id=1,
    )
    if family == "ouro":
        return OuroForCausalLM(OuroConfig(**common, total_ut_steps=4)).eval()
    return NanbeigeForCausalLM(NanbeigeConfig(**common, num_loops=2, total_ut_steps=2)).eval()


def make_engine(model, concurrency: int, enabled: bool) -> LLMEngine:
    return LLMEngine(
        model,
        cache_config=CacheConfig(256, 8, enable_prefix_caching=enabled),
        scheduler_config=SchedulerConfig(
            max_num_seqs=concurrency, max_num_batched_tokens=128, prefill_chunk_size=8
        ),
    )


@torch.inference_mode()
def generate(engine: LLMEngine, prompt: list[int], count: int, seed: int) -> tuple:
    cache = engine.cache_manager
    queries, hits = cache.prefix_queries, cache.prefix_hits
    start = time.perf_counter_ns()
    for index in range(count):
        engine.add_request(
            str(index),
            prompt,
            SamplingParams(
                max_tokens=4,
                min_loops=engine.model.config.total_ut_steps,
                max_loops=engine.model.config.total_ut_steps,
                temperature=0.8,
                seed=seed * 1000 + index,
                ignore_eos=True,
                logprobs=0,
                logprobs_mode="processed",
            ),
        )
    outputs: dict[str, RequestOutput] = {}
    first_prefill, prefill_tokens = {}, 0
    peak_live, peak_used, peak_shared = 0, 0, 0
    latencies = []
    for steps in range(1, 10001):
        if not engine.has_unfinished_requests():
            break
        for output in engine.step(final_only=True):
            assert output.finished and output.request_id not in outputs
            outputs[output.request_id] = output
            latencies.append((time.perf_counter_ns() - start) / 1e9)
        batch = engine.last_schedule
        if batch.stage == Stage.PREFILL:
            prefill_tokens += batch.num_tokens
            for item in batch.items:
                first_prefill.setdefault(item.request.request_id, item.token_start)
        live_refs: dict[int, int] = {}
        for allocation in cache._allocations.values():
            for table in allocation.block_tables:
                for page in table:
                    live_refs[page] = live_refs.get(page, 0) + 1
        peak_shared = max(peak_shared, sum(n > 1 for n in live_refs.values()))
        peak_live = max(peak_live, len(cache._allocations))
        peak_used = max(peak_used, cache.num_used_blocks)
    elapsed = (time.perf_counter_ns() - start) / 1e9
    assert not engine.has_unfinished_requests(), "finite scheduler budget exceeded"
    assert len(outputs) == count and all(len(output.token_ids) == 4 for output in outputs.values())
    assert len(first_prefill) == count and cache.num_used_blocks == 0
    reused = cache.prefix_hits - hits
    assert prefill_tokens + reused == count * len(prompt)
    rows = [outputs[str(index)] for index in range(count)]
    return rows, {
        "wall_seconds": elapsed,
        "prefill_tokens": prefill_tokens,
        "prefix_hit_tokens": reused,
        "prefix_queries": cache.prefix_queries - queries,
        "first_prefill_positions": [first_prefill[str(index)] for index in range(count)],
        "peak_live_requests": peak_live,
        "peak_live_shared_physical_pages": peak_shared,
        "peak_used_physical_pages": peak_used,
        "scheduler_steps": steps - 1,
        "completion_latency_seconds": latencies,
        "used_pages_after": cache.num_used_blocks,
    }


def compare(actual: list[RequestOutput], expected: list[RequestOutput]) -> float:
    largest = 0.0
    for observed, reference in zip(actual, expected, strict=True):
        assert observed.request_id == reference.request_id
        assert observed.token_ids == reference.token_ids
        assert observed.exit_depths == reference.exit_depths
        assert observed.finish_reason == reference.finish_reason
        assert observed.weight_version == reference.weight_version
        assert observed.log_probs is not None and reference.log_probs is not None
        left, right = torch.tensor(observed.log_probs), torch.tensor(reference.log_probs)
        torch.testing.assert_close(left, right, atol=1e-5, rtol=3e-5)
        largest = max(largest, float((left - right).abs().max()))
    return largest


def run(args) -> None:
    torch.set_num_threads(1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        for family in ("ouro", "nanbeige"):
            model = make_model(family, args.seed)
            for length in (15, 16, 17):
                prompt = [(index * 3 + 2) % 17 for index in range(length)]
                for concurrency in (1, 4, 8):
                    reference_engine = make_engine(model, concurrency, False)
                    reference, _ = generate(reference_engine, prompt, 8, args.seed)
                    reference_engine.close()
                    for order, arm in enumerate(
                        ("disabled", "cold", "warm", "warm", "cold", "disabled")
                    ):
                        engine = make_engine(model, concurrency, arm != "disabled")
                        primer_seconds = 0.0
                        if arm == "warm":
                            _, primer = generate(engine, prompt, 1, args.seed)
                            primer_seconds = primer["wall_seconds"]
                        outputs, metrics = generate(engine, prompt, 8, args.seed)
                        score_error = compare(outputs, reference)
                        engine.reset_prefix_cache()
                        assert engine.cache_manager.num_free_blocks == 256
                        assert not any(engine.cache_manager._refs)
                        engine.close()
                        row = {
                            "family": family,
                            "seed": args.seed,
                            "prompt_tokens": length,
                            "group_size": 8,
                            "concurrency": concurrency,
                            "arm": arm,
                            "order": order,
                            "primer_seconds": primer_seconds,
                            "primer_plus_group_seconds": primer_seconds + metrics["wall_seconds"],
                            "max_score_error": score_error,
                            "outputs": [
                                {
                                    "request_id": output.request_id,
                                    "tokens": output.token_ids,
                                    "depths": output.exit_depths,
                                    "scores": output.log_probs,
                                    "weight_version": output.weight_version,
                                    "finish_reason": output.finish_reason,
                                }
                                for output in outputs
                            ],
                            **metrics,
                            "device": "cpu",
                            "dtype": "float32",
                            "python": platform.python_version(),
                            "torch": torch.__version__,
                            "threads": 1,
                        }
                        stream.write(json.dumps(row, sort_keys=True) + "\n")
                        stream.flush()
                    print(
                        f"{family} seed={args.seed} P={length} G=8 C={concurrency}: passed",
                        flush=True,
                    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
