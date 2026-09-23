"""Paired, model-resident Ouro greedy E2E comparison on one CUDA device."""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from vllm_rlt.core.kv_cache_manager import KVCacheManager
from vllm_rlt.models import OuroForCausalLM
from vllm_rlt.models.config import OURO_REVISION
from vllm_rlt.speculative import FixedDepthGreedy

PROMPTS = {
    "repeated": "The quick brown fox jumps over the lazy dog. " * 12,
    "prose": (
        "Explain how a matrix multiplication works, including the shape of the inputs, "
        "the shape of the output, and one numerical example."
    ),
}


def run(decoder, prompt, max_tokens, gamma):
    torch.cuda.synchronize()
    start = time.perf_counter()
    result = decoder.generate(prompt, max_tokens, gamma=gamma, ignore_eos=True)
    torch.cuda.synchronize()
    return result, time.perf_counter() - start


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.max_tokens < 2 or args.repeats < 1:
        parser.error("max-tokens must be at least 2 and repeats must be positive")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    prompts = {name: tokenizer.encode(text) for name, text in PROMPTS.items()}
    model = OuroForCausalLM.from_pretrained(args.model, device="cuda:0", dtype=torch.bfloat16)
    config = model.config
    maximum = max(len(ids) for ids in prompts.values()) + args.max_tokens - 1
    blocks = ((maximum + 15) // 16 + 4) * config.total_ut_steps
    cache = KVCacheManager(
        config.num_hidden_layers,
        config.num_key_value_heads,
        config.head_dim,
        blocks,
        16,
        config.total_ut_steps,
        device="cuda:0",
        dtype=torch.bfloat16,
        backend="triton",
    )
    decoder = FixedDepthGreedy(model, cache)
    rows = []
    for name, prompt in prompts.items():
        reference, _ = run(decoder, prompt, args.max_tokens, 0)
        for gamma in (0, 1, 2, 4):
            run(decoder, prompt, args.max_tokens, gamma)
        for trial in range(args.repeats):
            order = (0, 1, 2, 4) if trial % 2 == 0 else (4, 2, 1, 0)
            for gamma in order:
                result, seconds = run(decoder, prompt, args.max_tokens, gamma)
                if result.token_ids != reference.token_ids:
                    raise AssertionError(f"{name} gamma={gamma} output mismatch")
                rows.append(
                    dict(
                        workload=name,
                        gamma=gamma,
                        trial=trial,
                        seconds=seconds,
                        output_tokens=len(result.token_ids),
                        drafted_tokens=result.drafted_tokens,
                        accepted_tokens=sum(result.accepted_lengths),
                        target_cycles=result.target_cycles,
                    )
                )

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "raw.json").write_text(
        json.dumps(
            dict(
                model="ByteDance/Ouro-1.4B",
                revision=OURO_REVISION,
                dtype="bfloat16",
                backend="triton",
                device=torch.cuda.get_device_name(0),
                prompt_token_ids=prompts,
                output_budget=args.max_tokens,
                rows=rows,
            ),
            indent=2,
        )
    )
    table = [
        "| Workload | γ | Baseline E2E s | Spec E2E s | Speedup | Accepted / drafted |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in prompts:
        baseline = statistics.median(
            row["seconds"] for row in rows if row["workload"] == name and row["gamma"] == 0
        )
        for gamma in (1, 2, 4):
            arm = [row for row in rows if row["workload"] == name and row["gamma"] == gamma]
            measured = statistics.median(row["seconds"] for row in arm)
            accepted = sum(row["accepted_tokens"] for row in arm)
            drafted = sum(row["drafted_tokens"] for row in arm)
            table.append(
                f"| {name} | {gamma} | {baseline:.3f} | {measured:.3f} | "
                f"{baseline / measured:.3f}× | {accepted} / {drafted} |"
            )
    (args.output / "speedup.md").write_text(
        "# Ouro-1.4B fixed-depth greedy E2E\n\n"
        "Model remains resident. Timings include prefill, draft, verification, and KV commit. "
        "Each arm has one warmup and paired inputs with alternating order; "
        "medians use three trials by default.\n\n"
        + "\n".join(table)
        + "\n"
    )


if __name__ == "__main__":
    main()
