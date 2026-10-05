"""Cold group-prefill comparisons with retained outputs and separate CPU oracles."""

import argparse
import hashlib
import inspect
import json
import os
import time
from pathlib import Path

import torch

from vllm_rlt import CacheConfig, SamplingParams, SchedulerConfig
from vllm_rlt.engine.llm_engine import LLMEngine
from vllm_rlt.models import NanbeigeConfig, NanbeigeForCausalLM, OuroConfig, OuroForCausalLM
from vllm_rlt.request import Stage


def make_model(family, seed):
    torch.manual_seed(seed)
    values = dict(
        vocab_size=17,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=512,
        bos_token_id=0,
        eos_token_id=1,
    )
    if family == "ouro":
        return OuroForCausalLM(OuroConfig(**values, total_ut_steps=4)).eval()
    return NanbeigeForCausalLM(NanbeigeConfig(**values, total_ut_steps=2, num_loops=2)).eval()


def make_engine(model, concurrency, fork, *, prefix=True):
    options = dict(max_num_seqs=concurrency, max_num_batched_tokens=128, prefill_chunk_size=8)
    if fork:
        options["group_prefill_fork"] = True
    return LLMEngine(
        model,
        cache_config=CacheConfig(1024, 8, enable_prefix_caching=prefix),
        scheduler_config=SchedulerConfig(**options),
    )


@torch.inference_mode()
def run(engine, length, count, seed, fork):
    prompt = [2 + i % 13 for i in range(length)]
    cache = engine.cache_manager
    results, prefill, peak_active, peak_pages, peak_shared = {}, 0, 0, 0, 0
    latencies = []
    start = time.perf_counter_ns()
    for i in range(count):
        engine.add_request(
            str(i),
            prompt,
            SamplingParams(
                max_tokens=8,
                min_loops=engine.model.config.total_ut_steps,
                temperature=0.8,
                seed=seed * 1000 + i,
                ignore_eos=True,
                logprobs=0,
                logprobs_mode="processed",
            ),
        )
    for step in range(10000):
        if not engine.has_unfinished_requests():
            break
        for output in engine.step(final_only=True):
            assert output.finished and output.request_id not in results
            results[output.request_id] = output
            latencies.append((time.perf_counter_ns() - start) / 1e9)
        batch = engine.last_schedule
        if batch is not None and batch.stage == Stage.PREFILL:
            prefill += batch.num_tokens
        peak_active = max(
            peak_active,
            sum(
                r.stage not in (Stage.WAITING, Stage.RECEIVING)
                for r in engine.scheduler.requests.values()
            ),
        )
        peak_pages = max(peak_pages, cache.num_used_blocks)
        peak_shared = max(peak_shared, sum(n > 1 for n in cache._refs))
    elapsed = (time.perf_counter_ns() - start) / 1e9
    assert not engine.has_unfinished_requests() and len(results) == count
    assert not cache._allocations
    reused = engine.scheduler.group_forks.reused_tokens if fork else 0
    assert prefill + cache.prefix_hits + reused == count * length
    assert all(len(out.token_ids) == 8 for out in results.values())
    if fork:
        assert (
            not engine.scheduler.group_forks.groups and not engine.scheduler.group_forks.by_request
        )
    engine.reset_prefix_cache()
    assert not any(cache._refs) and cache.num_free_blocks == 1024
    outputs = [results[str(i)] for i in range(count)]
    return outputs, {
        "wall_seconds": elapsed,
        "prefill_tokens": prefill,
        "ordinary_prefix_hit_tokens": cache.prefix_hits,
        "group_reused_tokens": reused,
        "group_forks": engine.scheduler.group_forks.forked_requests if fork else 0,
        "capacity_fallbacks": engine.scheduler.group_forks.capacity_fallbacks if fork else 0,
        "peak_active_requests": peak_active,
        "peak_used_pages_including_group_snapshot": peak_pages,
        "peak_shared_pages_including_cache_owners": peak_shared,
        "scheduler_steps": step,
        "request_completion_seconds": latencies,
        "outputs": [
            {
                "id": out.request_id,
                "tokens": out.token_ids,
                "scores": out.log_probs,
                "depths": out.exit_depths,
                "stop": out.finish_reason,
                "version": out.weight_version,
            }
            for out in outputs
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--fork", action="store_true")
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    source_root = Path(inspect.getfile(LLMEngine)).resolve().parents[2]
    source_files = {
        str(p.relative_to(source_root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted((source_root / "vllm_rlt").rglob("*.py"))
    }
    options.output.parent.mkdir(parents=True, exist_ok=True)
    workloads = [(p, c, 8) for p in (7, 8, 9, 17, 128, 129) for c in (1, 4, 8)] + [
        (7, 1, 1),
        (129, 1, 1),
    ]
    with options.output.open("x") as stream:
        for family in ("ouro", "nanbeige"):
            model = make_model(family, options.seed)
            for length, concurrency, count in workloads:
                reference = make_engine(model, 1, False, prefix=False)
                expected, _ = run(reference, length, count, options.seed, False)
                candidate = make_engine(model, concurrency, options.fork)
                actual, observation = run(candidate, length, count, options.seed, options.fork)
                error = 0.0
                for out, ref in zip(actual, expected, strict=True):
                    assert out.token_ids == ref.token_ids and out.exit_depths == ref.exit_depths
                    assert (
                        out.finish_reason == ref.finish_reason
                        and out.weight_version == ref.weight_version
                    )
                    a, b = torch.tensor(out.log_probs), torch.tensor(ref.log_probs)
                    torch.testing.assert_close(a, b, atol=1e-5, rtol=3e-5)
                    error = max(error, float((a - b).abs().max()))
                observation.update(
                    family=family,
                    prompt_length=length,
                    concurrency=concurrency,
                    group_size=count,
                    seed=options.seed,
                    label=options.label,
                    fork=options.fork,
                    source_root=str(source_root),
                    source_sha256=source_files,
                    max_reference_score_error=error,
                    load_average=os.getloadavg(),
                    torch_version=torch.__version__,
                    pid=os.getpid(),
                    threads=torch.get_num_threads(),
                )
                stream.write(json.dumps(observation, separators=(",", ":")) + "\n")
                stream.flush()
                print(
                    f"{options.label} s{options.seed} {family} P{length} C{concurrency} G{count}: "
                    f"{observation['wall_seconds']:.4f}s prefill={observation['prefill_tokens']}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
