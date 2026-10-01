"""Alternating-order warmed rollout timings, including returned log probabilities."""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from vllm_rlt import (
    LLM,
    CacheConfig,
    ExecutionConfig,
    SamplingParams,
    SchedulerConfig,
    SpeculativeConfig,
)
from vllm_rlt.models import OuroForCausalLM


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    prompts = [tokenizer.encode(text) for text in ("Compute 17 + 25.\n", "Compute 9 * 8.\n")]
    native = OuroForCausalLM.from_pretrained(args.model, device="cuda", dtype=torch.bfloat16)
    report = {}
    for speculative in (False, True):
        arms = {
            mode: LLM(
                native,
                attention_backend="triton",
                cache_config=CacheConfig(num_blocks=512),
                scheduler_config=SchedulerConfig(max_num_seqs=8, max_num_batched_tokens=128),
                execution_config=ExecutionConfig(cuda_graphs=mode == "graph"),
                speculative_config=SpeculativeConfig(3) if speculative else None,
            )
            for mode in ("eager", "graph")
        }
        for loops in (4,) if speculative else (2, 3, 4):
            params = [
                SamplingParams(
                    max_tokens=32,
                    max_loops=loops,
                    temperature=0.9,
                    seed=31 + i,
                    logprobs=0,
                    ignore_eos=True,
                )
                for i in range(len(prompts))
            ]
            for arm in arms.values():
                arm.generate(prompts, params)
                arm.start_weight_update()
                arm.update_weights(native.named_parameters())
                arm.finish_weight_update()
            version = arms["eager"].get_weight_version()
            times = {mode: [] for mode in arms}
            for index in range(args.repeats):
                outputs = {}
                for mode in ("eager", "graph") if index % 2 == 0 else ("graph", "eager"):
                    torch.cuda.synchronize()
                    started = time.monotonic()
                    outputs[mode] = arms[mode].generate(prompts, params)
                    torch.cuda.synchronize()
                    times[mode].append(time.monotonic() - started)
                for eager, graph in zip(outputs["eager"], outputs["graph"], strict=True):
                    assert eager.token_ids == graph.token_ids
                    assert eager.exit_depths == graph.exit_depths
                    assert eager.log_probs == graph.log_probs
                    assert eager.weight_version == graph.weight_version == version
            medians = {mode: statistics.median(values) for mode, values in times.items()}
            name = f"speculative-{loops}" if speculative else f"native-{loops}"
            report[name] = {
                "seconds": times,
                "medians": medians,
                "speedup": medians["eager"] / medians["graph"],
                "tokens_and_logprobs_exact": True,
            }
            print(name, report[name], flush=True)
        for arm in arms.values():
            arm.close()
        del arms
    args.output.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
