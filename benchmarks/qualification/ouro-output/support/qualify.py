"""Finite native Ouro output-path qualification; no performance or quality claim."""

import argparse
import gc
import hashlib
import json
import math
import platform
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch
from native_checkpoint import load_ouro

from vllm_rlt import CacheConfig, ExecutionConfig, ExitConfig, SamplingParams, SchedulerConfig
from vllm_rlt.engine.llm_engine import LLMEngine

CONCURRENCY = (1, 4, 16, 64)
MODES = ("sync", "async-single", "async-multi", "graph-sync")
ALLOCATOR_LIMIT = 12 * 1024**3
SCORE_ATOL = 0.02


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def make_engine(model, concurrency: int, mode: str, backend: str) -> LLMEngine:
    return LLMEngine(
        model,
        cache_config=CacheConfig(num_blocks=1536, block_size=4),
        scheduler_config=SchedulerConfig(
            max_num_seqs=concurrency, max_num_batched_tokens=256, prefill_chunk_size=16
        ),
        execution_config=ExecutionConfig(
            async_scheduling=mode.startswith("async"),
            multi_stream=mode == "async-multi",
            cuda_graphs=mode == "graph-sync",
            cuda_graph_max_batch_size=64,
            cuda_graph_max_graphs=8,
        ),
        exit_config=ExitConfig("ouro_delayed"),
        attention_backend=backend,
    )


@torch.inference_mode()
def serve(engine: LLMEngine, concurrency: int, final_only: bool) -> dict:
    """Same IDs and seeds in both arms also exercise released-slot reuse."""
    count = 2 * concurrency
    started = time.perf_counter()
    for index in range(count):
        length = 2 + index % 8
        engine.add_request(
            str(index),
            [
                2 + (index * 7 + pos * 3) % (engine.model.config.vocab_size - 2)
                for pos in range(length)
            ],
            SamplingParams(
                max_tokens=8,
                min_loops=4,
                max_loops=4,
                ignore_eos=True,
                temperature=0.8,
                seed=31800 + index,
                logprobs=0,
                logprobs_mode="processed",
            ),
        )
    completed, emitted_lengths = {}, {str(index): [] for index in range(count)}
    peak_live = peak_blocks = 0
    batch_counts: dict[str, int] = {}
    for steps in range(8192):
        if not engine.has_unfinished_requests():
            break
        outputs = engine.step(final_only=final_only)
        batch = engine.last_schedule
        if batch is not None:
            key = f"{batch.stage.value}:{len(batch.items)}"
            batch_counts[key] = batch_counts.get(key, 0) + 1
        for output in outputs:
            assert output.finished or not final_only
            emitted_lengths[output.request_id].append(len(output.token_ids))
            if output.finished:
                assert output.request_id not in completed
                completed[output.request_id] = asdict(output)
        peak_live = max(peak_live, len(engine.cache_manager._allocations))
        peak_blocks = max(peak_blocks, engine.cache_manager.num_used_blocks)
    else:
        raise RuntimeError("finite scheduler budget exceeded")
    assert len(completed) == count
    for request_id, output in completed.items():
        assert len(output["token_ids"]) == 8 and output["exit_depths"] == [4] * 8
        assert output["weight_version"] == 0 and output["finish_reason"] == "length"
        assert all(math.isfinite(score) for score in output["log_probs"])
        assert emitted_lengths[request_id] == ([8] if final_only else list(range(1, 9)))
    assert engine.cache_manager.num_used_blocks == 0
    assert not engine.cache_manager._allocations
    assert not engine._pending_coda and not engine._pending_exit_signals
    assert not engine._inflight and all(not queue for queue in engine.scheduler.queues.values())
    graphs = engine.model_runner.graphs
    return {
        "outputs": completed,
        "emitted_lengths": emitted_lengths,
        "scheduler_steps": steps,
        "batch_counts": batch_counts,
        "peak_live_requests": peak_live,
        "peak_used_blocks": peak_blocks,
        "used_blocks_after": engine.cache_manager.num_used_blocks,
        "observational_wall_seconds": time.perf_counter() - started,
        "graph_captures": graphs.captures if graphs is not None else 0,
        "graph_replays": graphs.replays if graphs is not None else 0,
        "graph_fallbacks": graphs.fallbacks if graphs is not None else 0,
    }


def compare_outputs(left: dict, right: dict) -> dict:
    assert left.keys() == right.keys()
    max_error = 0.0
    for request_id, a in left.items():
        b = right[request_id]
        assert a.keys() == b.keys()
        for field in a.keys() - {"log_probs"}:
            assert a[field] == b[field], (request_id, field, a[field], b[field])
        assert len(a["log_probs"]) == len(b["log_probs"]) == 8
        max_error = max(max_error, *(abs(x - y) for x, y in zip(a["log_probs"], b["log_probs"])))
    return {
        "tokens_depths_metadata_exact": True,
        "max_score_absolute_error": max_error,
        "scores_exact": max_error == 0,
        "scores_within_frozen_bound": max_error <= SCORE_ATOL,
    }


def run(folder: Path, output: Path, bundle: Path) -> None:
    manifest = json.loads((bundle / "manifest.json").read_text())
    for item in manifest["files"]:
        assert sha256(bundle / item["path"]) == item["sha256"], item["path"]
    for item in manifest["model_files"]:
        path = folder / item["path"]
        assert path.stat().st_size == item["bytes"] and sha256(path) == item["sha256"]
    import vllm_rlt

    assert Path(vllm_rlt.__file__).resolve() == bundle / "runtime/vllm_rlt/__init__.py"
    assert torch.cuda.device_count() == 1
    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(ALLOCATOR_LIMIT / total, 0)
    torch.manual_seed(81)
    torch.cuda.manual_seed_all(81)
    output.mkdir(parents=True, exist_ok=False)
    environment = {
        "source_commit": manifest["source_commit"],
        "manifest_sha256": sha256(bundle / "manifest.json"),
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0),
        "total_device_bytes": total,
        "torch_allocator_limit_bytes": ALLOCATOR_LIMIT,
        "limit_excludes_cuda_context_and_non_torch_allocators": True,
        "score_atol": SCORE_ATOL,
        "timing_scope": "overlapped qualification; not performance",
    }
    (output / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")
    model = load_ouro(folder, device="cuda", dtype=torch.bfloat16)
    with (output / "arms.jsonl").open("x") as stream:
        for mode in MODES:
            for concurrency in CONCURRENCY:
                engine = make_engine(model, concurrency, mode, "triton")
                paired = []
                for final_only in (False, True):
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    row = serve(engine, concurrency, final_only)
                    torch.cuda.synchronize()
                    row.update(
                        mode=mode,
                        concurrency=concurrency,
                        final_only=final_only,
                        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                        peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                    )
                    stream.write(json.dumps(row, sort_keys=True) + "\n")
                    stream.flush()
                    paired.append(row)
                result = compare_outputs(paired[0]["outputs"], paired[1]["outputs"])
                (output / f"comparison-{mode}-c{concurrency}.json").write_text(
                    json.dumps(result, indent=2) + "\n"
                )
                assert result["scores_within_frozen_bound"], result
                assert all(row["peak_live_requests"] == concurrency for row in paired)
                if mode == "graph-sync":
                    assert paired[1]["graph_captures"] > 0
                    assert paired[1]["graph_replays"] > paired[1]["graph_captures"]
                engine.close()
                del engine, paired
                gc.collect()
                torch.cuda.empty_cache()
                print(f"PASS {mode} C{concurrency}: {result}", flush=True)
    torch.cuda.synchronize()
    (output / "success.json").write_text(
        json.dumps({"cases": 16, "arms": 32, "status": "PASS"}) + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    args = parser.parse_args()
    run(args.model, args.output, args.bundle.resolve())
