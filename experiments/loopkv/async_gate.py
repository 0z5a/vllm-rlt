"""Finite CUDA routing/lifetime gate for the native and rectangular alias paths."""

import argparse
import json
from pathlib import Path

import torch

from vllm_rlt import CacheConfig, ExecutionConfig, ExitConfig, SamplingParams, SchedulerConfig
from vllm_rlt.engine.llm_engine import LLMEngine
from vllm_rlt.models import OuroConfig, OuroForCausalLM


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(19)
    model = (
        OuroForCausalLM(
            OuroConfig(
                vocab_size=64,
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=64,
                max_position_embeddings=128,
                total_ut_steps=4,
            )
        )
        .to(device="cuda", dtype=torch.bfloat16)
        .eval()
    )
    cases = []
    for batch in (1, 4, 16, 32, 64, 128):
        for use_uva in (False, True):
            baseline = None
            for mode in ("sync", "native_async", "alias_async", "native_graph"):
                engine = LLMEngine(
                    model,
                    attention_backend="triton",
                    cache_config=CacheConfig(
                        batch * 32, 2, alias_last_exited=mode == "alias_async"
                    ),
                    scheduler_config=SchedulerConfig(
                        max_num_seqs=batch, max_num_batched_tokens=max(4, batch)
                    ),
                    exit_config=ExitConfig(
                        "trace", depths_by_request={"frozen": [4, 1, 3, 2, 4, 1]}
                    ),
                    execution_config=ExecutionConfig(
                        async_scheduling=mode != "sync",
                        cuda_graphs=mode == "native_graph",
                        static_buffers=mode == "native_graph",
                        pad_to_power_of_two=mode == "native_graph",
                    ),
                )
                if engine.model_runner.async_state is not None:
                    engine.model_runner.async_state.use_uva = use_uva
                rounds = []
                for round_index in range(2):
                    for i in range(2 * batch):
                        engine.add_request(
                            str(i),
                            [2 + i % 5, 9 + round_index],
                            SamplingParams(
                                max_tokens=6,
                                min_loops=1,
                                ignore_eos=True,
                            ),
                            trace_id="frozen",
                        )
                    if round_index:
                        engine.step()
                        engine.abort_request("0")
                        engine.add_request(
                            "0",
                            [7, 8, 9],
                            SamplingParams(
                                max_tokens=6,
                                min_loops=1,
                                ignore_eos=True,
                            ),
                            trace_id="frozen",
                        )
                    completed = {}
                    for _ in range(100_000):
                        if not engine.has_unfinished_requests():
                            break
                        for output in engine.step():
                            if output.finished:
                                completed[output.request_id] = (
                                    output.token_ids,
                                    output.exit_depths,
                                )
                    engine.model_runner.synchronize()
                    assert len(completed) == 2 * batch
                    assert engine.cache_manager.num_free_blocks == batch * 32
                    assert not engine.model_runner.state_slots
                    rounds.append(completed)
                if baseline is None:
                    baseline = rounds
                assert rounds == baseline, (batch, use_uva, mode)
                row = {
                    "batch": batch,
                    "requests_per_round": 2 * batch,
                    "use_uva": use_uva,
                    "mode": mode,
                    "exact_two_rounds": True,
                }
                cases.append(row)
                print(json.dumps(row), flush=True)
                engine.close()
    args.out.write_text(
        json.dumps(
            {
                "scope": "tiny_cuda_routing_lifetime_not_official_or_performance",
                "torch": torch.__version__,
                "gpu": torch.cuda.get_device_name(),
                "cases": cases,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
