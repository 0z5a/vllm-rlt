"""Finite CUDA text-stop cancellation and ID-reuse gate for resident execution."""

import argparse
import gc
import json
from pathlib import Path

import torch

from experiments.loopkv.capture import drive, dump
from vllm_rlt import CacheConfig, ExecutionConfig, ExitConfig, SamplingParams, SchedulerConfig
from vllm_rlt.engine.llm_engine import LLMEngine
from vllm_rlt.models import OuroConfig, OuroForCausalLM


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(23)
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
    pieces = [b"x"] * 64
    pieces[10:14] = [b"a", b"Q", b":", b"later"]
    stops = {"token_bytes_hex": [p.hex() for p in pieces], "stops": ["Q:"]}
    cases = []
    for batch in (1, 32, 128):
        for alias, graphs in ((False, False), (True, False), (False, True), (True, True)):
            engine = LLMEngine(
                model,
                attention_backend="triton",
                cache_config=CacheConfig(batch * 32, 16, alias_last_exited=alias),
                scheduler_config=SchedulerConfig(max_num_seqs=batch, max_num_batched_tokens=128),
                exit_config=ExitConfig("ouro_delayed"),
                execution_config=ExecutionConfig(
                    async_scheduling=True, static_buffers=graphs, cuda_graphs=graphs
                ),
            )
            engine.model_runner._sample_tensor = lambda logits, request: torch.tensor(
                10 + request.num_scheduled_outputs, device="cuda"
            )
            for _ in range(2):
                completed, _, _ = drive(
                    engine,
                    [[2, 3] * 17 if i % 2 else [4] * 33 for i in range(batch * 2)],
                    SamplingParams(max_tokens=8, exit_threshold=0.5, ignore_eos=True),
                    trace=False,
                    stop_spec=stops,
                )
                assert len(completed) == batch * 2
                assert all(
                    r["token_ids"] == [10, 11, 12]
                    and r["text_stop"] == {"text": "Q:", "byte_offset": 1}
                    for r in completed.values()
                )
                assert engine.cache_manager.num_free_blocks == batch * 32
                assert not engine.model_runner.state_slots
            row = {
                "batch": batch,
                "alias": alias,
                "graphs": graphs,
                "rounds": 2,
                "requests_per_round": batch * 2,
                "exact_stop_tokens_and_offsets": True,
            }
            if graphs:
                graph = engine.model_runner.graphs
                row["graph_counts"] = {
                    "captures": graph.captures,
                    "replays": graph.replays,
                    "fallbacks": graph.fallbacks,
                }
                assert graph.replays > graph.captures > 0
                del graph
            cases.append(row)
            print(json.dumps(row), flush=True)
            engine.close()
            del engine
            gc.collect()
            torch.cuda.empty_cache()
    dump(args.out, {"scope": "tiny_cuda_cancellation_not_model_quality", "cases": cases})


if __name__ == "__main__":
    main()
