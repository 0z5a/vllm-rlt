"""Reproduce PR82 full-tensor parity, paired coda ablation and complete-token profiles.

Each invocation is a fresh process. The core_only arm disables only coda_graphs.
Use timing --start 0 and --start 1 for the two reversed batch/arm orders.
"""

import argparse
import gc
import hashlib
import json
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARMS = ("core_only", "core_coda")
MODEL_REVISION = "bb6621b65e90b6a4b9b29ef88dc83866d450470c"


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".writing")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"size": Path(path).stat().st_size, "sha256": digest.hexdigest()}


PROMPTS = (
    "The capital of France is",
    "Explain why water freezes at low temperatures in one sentence.",
    "Write a short Python function that adds two integers and returns the result.",
    "Question: A box contains 12 apples. We add 8 and remove 5. How many remain? Answer:",
)


def graph_stats(engine):
    result = {}
    for name, bank in (
        ("core", engine.model_runner.graphs),
        ("coda", engine.model_runner.coda_graphs),
    ):
        result[name] = {
            "captures": bank.captures if bank else 0,
            "replays": bank.replays if bank else 0,
            "fallbacks": bank.fallbacks if bank else 0,
            "keys": sorted(repr(key) for key in bank.entries) if bank else [],
        }
    return result


def make_engine(model, arm):
    from vllm_rlt import CacheConfig, ExecutionConfig, SchedulerConfig
    from vllm_rlt.engine.llm_engine import LLMEngine

    engine = LLMEngine(
        model,
        cache_config=CacheConfig(num_blocks=1024, block_size=16, layout="last_exited"),
        scheduler_config=SchedulerConfig(
            max_num_seqs=4, max_num_batched_tokens=32, prefill_chunk_size=8
        ),
        execution_config=ExecutionConfig(cuda_graphs=True),
        attention_backend="triton",
    )
    if arm == "core_only":
        engine.model_runner.coda_graphs = None
    assert engine.model_runner.graphs is not None
    assert (engine.model_runner.coda_graphs is not None) == (arm == "core_coda")
    return engine


def sample_params(tokens, seed):
    from vllm_rlt import SamplingParams

    return SamplingParams(max_tokens=tokens, min_loops=32, max_loops=32, ignore_eos=True, seed=seed)


def burst(engine, prompts, tokens=16, seed=123):
    import torch

    torch.manual_seed(seed)
    before = graph_stats(engine)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    sampling = sample_params(tokens, seed)
    observations = {str(i): [] for i in range(len(prompts))}
    finished, stages = {}, {}
    started = time.perf_counter()
    for index, prompt in enumerate(prompts):
        engine.add_request(str(index), prompt, sampling)
    steps = 0
    while engine.has_unfinished_requests():
        outputs = engine.step()
        now = time.perf_counter() - started
        batch = engine.last_schedule
        if batch is not None:
            key = batch.stage.value
            entry = stages.setdefault(key, {"steps": 0, "request_rows": 0, "token_rows": 0})
            entry["steps"] += 1
            entry["request_rows"] += len(batch.items)
            entry["token_rows"] += sum(item.token_count for item in batch.items)
        for output in outputs:
            rid = output.request_id
            assert len(output.token_ids) == len(observations[rid]) + 1
            observations[rid].append(now)
            if output.finished:
                finished[rid] = {
                    "token_ids": output.token_ids,
                    "exit_depths": output.exit_depths,
                    "finish_reason": output.finish_reason,
                    "prompt_token_ids": output.prompt_token_ids,
                }
        steps += 1
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    after = graph_stats(engine)
    assert len(finished) == len(prompts) and engine.cache_manager.num_used_blocks == 0
    for rid, value in finished.items():
        assert len(value["token_ids"]) == tokens and value["exit_depths"] == [32] * tokens
        assert value["finish_reason"] == "length"
    request_metrics = {
        rid: {
            "delivery_seconds": times,
            "ttft_seconds": times[0],
            "e2e_seconds": times[-1],
            "tpot_seconds": (times[-1] - times[0]) / (tokens - 1),
            "inter_token_seconds": [b - a for a, b in zip(times, times[1:])],
        }
        for rid, times in observations.items()
    }
    return {
        "seconds": seconds,
        "tokens_per_second": len(prompts) * tokens / seconds,
        "requests_per_second": len(prompts) / seconds,
        "outputs": finished,
        "request_metrics": request_metrics,
        "steps": steps,
        "stages": stages,
        "kv_used_blocks_after": engine.cache_manager.num_used_blocks,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "graph_delta": {
            name: {
                key: after[name][key] - before[name][key]
                for key in ("captures", "replays", "fallbacks")
            }
            for name in before
        },
        "graph_banks_after": after,
    }


def retain_tensor(folder, name, tensor):
    import torch

    value = tensor.detach().contiguous()
    data = value.view(torch.uint8).cpu().numpy().tobytes()
    path = folder / (name + ".bin")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(data)
    return {
        "path": path.name,
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "size": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def traced_burst(engine, prompts, folder):
    from vllm_rlt.request import Stage

    folder.mkdir(parents=True, exist_ok=False)
    runner = engine.model_runner
    execute, sampler = runner.execute, runner._sample_tensor
    trace = {}

    def save(name, tensor):
        assert name not in trace
        trace[name] = retain_tensor(folder, name, tensor)
        write_json(folder / "tensor-index.json", trace)

    def sample(logits, request):
        save(f"r{request.request_id}-t{request.num_scheduled_outputs}-logits", logits)
        return sampler(logits, request)

    def execute_trace(batch):
        requests = [item.request for item in batch.items]
        if batch.stage == Stage.CODA:
            for request in requests:
                save(
                    f"r{request.request_id}-t{request.num_scheduled_outputs}-hidden",
                    request.hidden_state,
                )
        result = execute(batch)
        if batch.stage == Stage.CODA:
            for request in requests:
                for layer in range(engine.model.config.num_hidden_layers):
                    depth = (
                        request.loops_done - 1 if layer in engine.model.recurrent_kv_layers else 0
                    )
                    for kind, tensor in zip(
                        ("k", "v"), engine.cache_manager.read(layer, request.request_id, depth)
                    ):
                        save(
                            f"r{request.request_id}-t{request.num_scheduled_outputs}-l{layer}-{kind}",
                            tensor,
                        )
        return result

    runner.execute, runner._sample_tensor = execute_trace, sample
    try:
        result = burst(engine, prompts, tokens=3)
        result["diagnostic_only_not_performance"] = True
        result["tensor_index"] = str(folder / "tensor-index.json")
        return result
    finally:
        runner.execute, runner._sample_tensor = execute, sampler


def parity(model, prompts, output):
    engines = {arm: make_engine(model, arm) for arm in ARMS}
    cases = []
    try:
        for count in (1, 4):
            for wave in range(2):
                record = {"batch": count, "reuse_wave": wave, "arms": {}}
                for arm in ARMS:
                    record["arms"][arm] = traced_burst(
                        engines[arm], prompts[:count], output / f"b{count}-w{wave}-{arm}"
                    )
                    write_json(output / "parity-progress.json", {"cases": cases, "current": record})
                assert record["arms"][ARMS[0]]["outputs"] == record["arms"][ARMS[1]]["outputs"]
                cases.append(record)
        return {"cases": cases, "diagnostic_only_not_performance": True}
    finally:
        for engine in engines.values():
            engine.model_runner.synchronize()
            engine.close()


def timing(model, prompts, output, start):
    import torch

    cases = []
    for count in (1, 4) if start == 0 else (4, 1):
        engines = {arm: make_engine(model, arm) for arm in ARMS}
        case = {"batch": count, "warmups": [], "pairs": []}
        try:
            for trial in range(10):
                order = ARMS if (trial + start) % 2 == 0 else ARMS[::-1]
                seed = 123 + trial if trial < 5 else 123
                record = {
                    "trial": trial if trial < 5 else trial - 5,
                    "order": list(order),
                    "seed": seed,
                    "arms": {},
                }
                for arm in order:
                    record["arms"][arm] = burst(engines[arm], prompts[:count], seed=seed)
                    write_json(
                        output / "timing-progress.json",
                        {"cases": cases, "current_case": case, "current_trial": record},
                    )
                assert record["arms"][ARMS[0]]["outputs"] == record["arms"][ARMS[1]]["outputs"]
                if trial >= 5:
                    for arm in ARMS:
                        assert record["arms"][arm]["graph_delta"]["core"]["captures"] == 0
                        assert record["arms"][arm]["graph_delta"]["coda"]["captures"] == 0
                    case["pairs"].append(record)
                else:
                    case["warmups"].append(record)
            case["memory_plan"] = {arm: engines[arm].memory_plan for arm in ARMS}
            cases.append(case)
            write_json(output / "timing-progress.json", {"cases": cases})
        finally:
            for engine in engines.values():
                engine.model_runner.synchronize()
                engine.close()
            del engine, engines
            gc.collect()
            torch.cuda.empty_cache()
    return {
        "start": start,
        "cases": cases,
        "unprofiled": True,
        "simultaneously_resident_engines": 2,
    }


def profile(model, prompts, output, arm, level):
    import torch

    engine = make_engine(model, arm)
    full = level == "full"
    try:
        warmups = [burst(engine, prompts) for _ in range(5)]
        for index, prompt in enumerate(prompts):
            engine.add_request(str(index), prompt, sample_params(16, 123))
        first = set()
        while len(first) != len(prompts):
            for value in engine.step():
                assert len(value.token_ids) == 1
                first.add(value.request_id)
        before = graph_stats(engine)
        second = set()
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=full,
            profile_memory=full,
            with_stack=full,
        ) as profiler:
            with torch.profiler.record_function("pr82_complete_decode_token_2"):
                while len(second) != len(prompts):
                    for value in engine.step():
                        assert len(value.token_ids) == 2
                        second.add(value.request_id)
                torch.cuda.synchronize()
        profiler.export_chrome_trace(str(output / "trace.json"))
        events = profiler.events()
        event_counts = {}
        for event in events:
            key = str(event.device_type)
            event_counts[key] = event_counts.get(key, 0) + 1
        write_json(output / "profile-event-counts.json", event_counts)
        (output / "operator-table.txt").write_text(
            profiler.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1)
        )
        after = graph_stats(engine)
        assert after["core"]["replays"] - before["core"]["replays"] == 32
        assert after["coda"]["replays"] - before["coda"]["replays"] == (
            1 if arm == "core_coda" else 0
        )
        assert all(after[name]["captures"] == before[name]["captures"] for name in before)
        final = {}
        while engine.has_unfinished_requests():
            for value in engine.step():
                if value.finished:
                    final[value.request_id] = {
                        "token_ids": value.token_ids,
                        "exit_depths": value.exit_depths,
                        "finish_reason": value.finish_reason,
                    }
        assert engine.cache_manager.num_used_blocks == 0 and len(final) == len(prompts)
        return {
            "arm": arm,
            "level": level,
            "record_shapes": full,
            "profile_memory": full,
            "with_stack": full,
            "activities": ["CPU", "CUDA"],
            "complete_decode_token_index": 2,
            "batch": len(prompts),
            "warmups": warmups,
            "graph_before": before,
            "graph_after": after,
            "outputs": final,
            "event_counts": event_counts,
            "trace": sha_file(output / "trace.json"),
            "diagnostic_only_not_performance": True,
            "kv_used_blocks_after": 0,
        }
    finally:
        engine.model_runner.synchronize()
        engine.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("parity", "timing", "profile"), required=True)
    parser.add_argument("--start", type=int, choices=(0, 1), default=0)
    parser.add_argument("--arm", choices=ARMS, default=ARMS[0])
    parser.add_argument("--level", choices=("ops", "full"), default="ops")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--runtime-manifest", type=Path)
    parser.add_argument("--source", type=Path, default=ROOT)
    parser.add_argument(
        "--model-manifest",
        type=Path,
        default=ROOT / "benchmarks/fixtures/huginn-0125-manifest.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-uuid")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    model_manifest = read_json(args.model_manifest)
    assert model_manifest["revision"] == MODEL_REVISION
    for name, expected in model_manifest["files"].items():
        actual = sha_file(args.model / name)
        assert all(actual[key] == expected[key] for key in ("size", "sha256")), name
    source = args.source.resolve()
    head = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = bool(subprocess.check_output(["git", "-C", str(source), "status", "--porcelain"]))
    sys.path.insert(0, str(source))
    import numpy
    import safetensors
    import tokenizers
    import torch
    import transformers
    import triton
    from transformers import AutoTokenizer

    from vllm_rlt.models import AutoModelForCausalLM

    if args.runtime_manifest:
        runtime = read_json(args.runtime_manifest)
        manifest_paths = {str(Path(row["path"]).resolve()) for row in runtime["files"]}
        for module in (numpy, safetensors, tokenizers, torch, torch._C, transformers, triton):
            assert str(Path(module.__file__).resolve()) in manifest_paths, module.__name__
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.cuda.set_device(0)
    properties = torch.cuda.get_device_properties(0)
    if args.gpu_uuid:
        assert str(properties.uuid).removeprefix("GPU-") == args.gpu_uuid.removeprefix("GPU-")
    assert torch.cuda.device_count() == 1, "Owner must assign exactly one visible GPU"
    assert properties.total_memory >= 32 * 1024**3
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=True, trust_remote_code=False
    )
    prompts = [tokenizer.encode(text, add_special_tokens=False) for text in PROMPTS]
    write_json(
        args.output / "frozen-prompts.json",
        {"texts": PROMPTS, "token_ids": prompts, "model_revision": MODEL_REVISION},
    )
    begin = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(args.model, device="cuda", dtype=torch.bfloat16)
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - begin
    assert model.config.n_embd == 5280 and model.config.n_heads == 55
    assert model.config.total_ut_steps == 32 and model.config.head_dim == 96
    assert model.lm_head.weight is model.transformer.wte.weight
    header = {
        "head": head,
        "source_dirty": dirty,
        "model_revision": MODEL_REVISION,
        "mode": args.mode,
        "runtime": {
            module.__name__: module.__version__
            for module in (numpy, safetensors, tokenizers, torch, transformers, triton)
        },
        "python": sys.version,
        "gpu": properties.name,
        "gpu_uuid": str(properties.uuid),
        "gpu_total_memory_bytes": properties.total_memory,
        "capability": [properties.major, properties.minor],
        "model_load_seconds_excluded": load_seconds,
        "timing_boundary": (
            "CUDA-synchronized request admission through returned CPU tokens and KV reclamation; "
            "native synchronous engine, no network or text decode"
        ),
        "initial_state_rng": (
            "new request-owned persistent generator, SamplingParams.seed reset identically per arm"
        ),
        "sampling": asdict(sample_params(16, 123)),
        "prompt_token_ids": prompts,
    }
    write_json(args.output / "run-header.json", header)
    with torch.inference_mode():
        if args.mode == "parity":
            result = parity(model, prompts, args.output)
        elif args.mode == "timing":
            result = timing(model, prompts, args.output, args.start)
        else:
            result = profile(model, prompts[:1], args.output, args.arm, args.level)
    write_json(args.output / "result.json", {"producer_complete": True, **header, **result})
    print(json.dumps({"mode": args.mode, "producer_complete": True}), flush=True)


if __name__ == "__main__":
    main()
