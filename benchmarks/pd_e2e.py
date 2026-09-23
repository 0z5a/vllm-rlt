"""Equal-card Ouro E2E: two independent replicas versus one prefill/decode pair."""

import argparse
import json
import multiprocessing as mp
import statistics
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from vllm_rlt.config import CacheConfig, ExecutionConfig, ExitConfig, SchedulerConfig
from vllm_rlt.engine.llm_engine import LLMEngine
from vllm_rlt.models import OuroForCausalLM
from vllm_rlt.pd import PDConfig, PDEngine
from vllm_rlt.sampling_params import SamplingParams


def settings(max_tokens):
    return dict(
        cache_config=CacheConfig(num_blocks=1536),
        scheduler_config=SchedulerConfig(
            max_num_seqs=4, max_num_batched_tokens=256, prefill_chunk_size=256
        ),
        exit_config=ExitConfig("trace", depths_by_request={"fixed": [4] * max_tokens}),
        execution_config=ExecutionConfig(async_scheduling=True),
        attention_backend="triton",
    )


def serve(engine, prompts, params):
    for request_id, tokens in prompts.items():
        engine.add_request(request_id, tokens, params, trace_id="fixed")
    outputs = {}
    while engine.has_unfinished_requests():
        for output in engine.step():
            if output.finished:
                outputs[output.request_id] = dict(
                    token_ids=output.token_ids, exit_depths=output.exit_depths
                )
    if len(outputs) != len(prompts):
        raise RuntimeError("E2E request count mismatch")
    return outputs


def replica(device, model_path, prompts, max_tokens, channel):
    model = OuroForCausalLM.from_pretrained(
        model_path, device=f"cuda:{device}", dtype=torch.bfloat16
    )
    options = settings(max_tokens)
    engine = LLMEngine(
        model,
        cache_config=options["cache_config"],
        scheduler_config=options["scheduler_config"],
        exit_config=options["exit_config"],
        execution_config=options["execution_config"],
        attention_backend=options["attention_backend"],
    )
    warmup = SamplingParams(max_tokens=4, min_loops=4, max_loops=4, ignore_eos=True)
    serve(engine, {"warmup": next(iter(prompts.values()))[:32]}, warmup)
    channel.send("ready")
    channel.recv()
    params = SamplingParams(
        max_tokens=max_tokens, min_loops=4, max_loops=4, ignore_eos=True
    )
    outputs = serve(engine, prompts, params)
    torch.cuda.synchronize(device)
    channel.send(outputs)
    channel.close()


def run_replicas(model_path, prompts, max_tokens):
    context = mp.get_context("spawn")
    workers = []
    for device in (0, 1):
        parent, child = context.Pipe()
        assigned = {key: value for key, value in prompts.items() if int(key) % 2 == device}
        process = context.Process(
            target=replica, args=(device, model_path, assigned, max_tokens, child)
        )
        process.start()
        child.close()
        workers.append((process, parent))
    for _, channel in workers:
        if channel.recv() != "ready":
            raise RuntimeError("replica startup failed")
    start = time.perf_counter()
    for _, channel in workers:
        channel.send("start")
    outputs = {}
    for _, channel in workers:
        outputs.update(channel.recv())
    seconds = time.perf_counter() - start
    for process, channel in workers:
        process.join()
        channel.close()
        if process.exitcode != 0:
            raise RuntimeError(f"replica exited {process.exitcode}")
    return outputs, seconds


def run_pd(model_path, prompts, max_tokens):
    options = settings(max_tokens)
    scheduler = options["scheduler_config"]
    with PDEngine(
        model_path,
        pd_config=PDConfig(request_timeout=900, startup_timeout=900),
        prefill_cache_config=options["cache_config"],
        decode_cache_config=options["cache_config"],
        prefill_scheduler_config=scheduler,
        decode_scheduler_config=SchedulerConfig(max_num_seqs=4, max_num_batched_tokens=4),
        exit_config=options["exit_config"],
        execution_config=options["execution_config"],
        attention_backend=options["attention_backend"],
    ) as engine:
        warmup = SamplingParams(max_tokens=4, min_loops=4, max_loops=4, ignore_eos=True)
        serve(engine, {"warmup": next(iter(prompts.values()))[:32]}, warmup)
        params = SamplingParams(
            max_tokens=max_tokens, min_loops=4, max_loops=4, ignore_eos=True
        )
        start = time.perf_counter()
        outputs = serve(engine, prompts, params)
        seconds = time.perf_counter() - start
    return outputs, seconds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt-tokens", type=int, default=256)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--requests", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if min(args.prompt_tokens, args.max_tokens, args.repeats) < 1 or args.requests < 2:
        parser.error("positive lengths/repeats and at least two requests are required")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    texts = (
        "The quick brown fox jumps over the lazy dog. ",
        "Explain matrix multiplication with a numerical example. ",
        "Describe how a compiler translates source code. ",
        "Summarize the steps of a scientific experiment. ",
    )
    prompts = {
        str(index): tokenizer.encode(texts[index % len(texts)] * (args.prompt_tokens + 1))[
            : args.prompt_tokens
        ]
        for index in range(args.requests)
    }
    rows = []
    raw = dict(
        model=str(args.model),
        device=torch.cuda.get_device_name(0),
        prompt_tokens=args.prompt_tokens,
        output_tokens=args.max_tokens,
        prompts=prompts,
        rows=rows,
    )
    for trial in range(args.repeats):
        for name, runner in (
            (("replicas", run_replicas), ("pd", run_pd))
            if trial % 2 == 0
            else (("pd", run_pd), ("replicas", run_replicas))
        ):
            outputs, seconds = runner(str(args.model), prompts, args.max_tokens)
            rows.append(dict(trial=trial, arm=name, seconds=seconds, outputs=outputs))
        pair = rows[-2:]
        if pair[0]["outputs"] != pair[1]["outputs"]:
            args.output.mkdir(parents=True, exist_ok=True)
            (args.output / "raw.json").write_text(json.dumps(raw, indent=2))
            raise AssertionError(f"trial {trial}: PD output differs from equal-card replicas")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "raw.json").write_text(json.dumps(raw, indent=2))
    baseline = statistics.median(row["seconds"] for row in rows if row["arm"] == "replicas")
    pd = statistics.median(row["seconds"] for row in rows if row["arm"] == "pd")
    (args.output / "speedup.md").write_text(
        "# Ouro-1.4B equal-card E2E\n\n"
        f"Device: {torch.cuda.get_device_name(0)}; 2 replicas versus 1P1D. "
        f"{args.requests} requests × {args.prompt_tokens} prompt / "
        f"{args.max_tokens} output tokens; "
        "BF16, Triton, LAST_EXITED, fixed four loops. "
        "Model startup and warmup excluded; prefill, NIXL transfer, and decode included. "
        f"{args.repeats} alternating trials with strict token and exit-depth equality.\n\n"
        "| GPUs | Replicas median s | PD median s | PD speedup | Status |\n"
        "|---:|---:|---:|---:|---|\n"
        f"| 2 | {baseline:.3f} | {pd:.3f} | {baseline / pd:.3f}× | PASS |\n"
    )


if __name__ == "__main__":
    main()
