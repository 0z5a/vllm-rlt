"""Equal-card speculative E2E: two replicas versus one prefill/decode pair."""

import argparse
import json
import multiprocessing as mp
import statistics
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from vllm_rlt import CacheConfig, SamplingParams, SchedulerConfig, SpeculativeConfig
from vllm_rlt.engine.llm_engine import LLMEngine
from vllm_rlt.models import OuroForCausalLM
from vllm_rlt.pd import PDConfig, PDEngine

SPEC = SpeculativeConfig(4)


def serve(engine, prompts, params):
    for request_id, tokens in prompts.items():
        engine.add_request(request_id, tokens, params)
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
    engine = LLMEngine(
        model,
        cache_config=CacheConfig(num_blocks=1024),
        scheduler_config=SchedulerConfig(max_num_seqs=4, max_num_batched_tokens=128),
        attention_backend="triton",
        speculative_config=SPEC,
    )
    warmup = SamplingParams(max_tokens=4, min_loops=4, max_loops=4, ignore_eos=True)
    serve(engine, {"warmup": next(iter(prompts.values()))[:32]}, warmup)
    channel.send("ready")
    channel.recv()
    params = SamplingParams(max_tokens=max_tokens, min_loops=4, max_loops=4, ignore_eos=True)
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
    scheduler = SchedulerConfig(max_num_seqs=4, max_num_batched_tokens=128)
    with PDEngine(
        model_path,
        pd_config=PDConfig(
            prefill_devices=(0,), decode_devices=(1,), request_timeout=900, startup_timeout=900
        ),
        prefill_cache_config=CacheConfig(num_blocks=1024),
        decode_cache_config=CacheConfig(num_blocks=1024),
        prefill_scheduler_config=scheduler,
        decode_scheduler_config=scheduler,
        attention_backend="triton",
        speculative_config=SPEC,
    ) as engine:
        warmup = SamplingParams(max_tokens=4, min_loops=4, max_loops=4, ignore_eos=True)
        serve(engine, {"warmup": next(iter(prompts.values()))[:32]}, warmup)
        params = SamplingParams(max_tokens=max_tokens, min_loops=4, max_loops=4, ignore_eos=True)
        start = time.perf_counter()
        outputs = serve(engine, prompts, params)
        seconds = time.perf_counter() - start
    return outputs, seconds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt-tokens", type=int, default=128)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--requests", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=5)
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
        model="ByteDance/Ouro-1.4B",
        device=torch.cuda.get_device_name(0),
        dtype="bfloat16",
        backend="triton",
        speculative=dict(draft_loops=2, target_loops=4, k=4),
        prompt_tokens=args.prompt_tokens,
        output_tokens=args.max_tokens,
        prompts=prompts,
        rows=rows,
    )
    for trial in range(args.repeats):
        runners = (("replicas", run_replicas), ("pd", run_pd))
        for name, runner in runners if trial % 2 == 0 else reversed(runners):
            outputs, seconds = runner(str(args.model), prompts, args.max_tokens)
            rows.append(dict(trial=trial, arm=name, seconds=seconds, outputs=outputs))
        if rows[-2]["outputs"] != rows[-1]["outputs"]:
            args.output.mkdir(parents=True, exist_ok=True)
            (args.output / "raw.json").write_text(json.dumps(raw, indent=2))
            raise AssertionError(f"trial {trial}: PD output differs from equal-card replicas")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "raw.json").write_text(json.dumps(raw, indent=2))
    baseline = statistics.median(row["seconds"] for row in rows if row["arm"] == "replicas")
    pd = statistics.median(row["seconds"] for row in rows if row["arm"] == "pd")
    (args.output / "speedup.md").write_text(
        "# Ouro-1.4B speculative PD E2E\n\n"
        f"{torch.cuda.get_device_name(0)}; two replicas versus 1P1D. "
        f"{args.requests} requests × {args.prompt_tokens} prompt / "
        f"{args.max_tokens} output tokens; BF16, Triton, d=2/D=4, K=4. "
        "Startup and warmup excluded; prefill, NIXL transfer, speculative decode "
        f"and KV commit included. {args.repeats} alternating paired trials; "
        "all output tokens and exit depths matched.\n\n"
        "| GPUs | Replicas median s | PD median s | PD speedup | Output |\n"
        "|---:|---:|---:|---:|---|\n"
        f"| 2 | {baseline:.3f} | {pd:.3f} | {baseline / pd:.3f}× | exact match |\n"
    )


if __name__ == "__main__":
    main()
