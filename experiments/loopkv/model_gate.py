"""Finite tiny-model cache qualification, separate from official-weight results."""

import argparse
import json
from pathlib import Path

import torch

from vllm_rlt import CacheConfig, SamplingParams, SchedulerConfig
from vllm_rlt.core.compact_kv_cache import CompactKVCacheManager
from vllm_rlt.engine.llm_engine import LLMEngine
from vllm_rlt.models import HuginnConfig, HuginnForCausalLM, NanbeigeConfig, NanbeigeForCausalLM


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 4, 16, 32, 64, 128])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(29)
    models = (
        NanbeigeForCausalLM(
            NanbeigeConfig(
                vocab_size=64,
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=8,
                max_position_embeddings=128,
                bos_token_id=0,
                eos_token_id=1,
            )
        ),
        HuginnForCausalLM(
            HuginnConfig(
                n_embd=32,
                n_heads=4,
                n_layers=4,
                n_layers_in_prelude=1,
                n_layers_in_recurrent_block=2,
                n_layers_in_coda=1,
                intermediate_size=64,
                mean_recurrence=3,
                block_size=128,
                vocab_size=64,
                padded_vocab_size=64,
                bos_token_id=0,
                eos_token_id=1,
                pad_token_id=0,
            )
        ),
    )
    result = {
        "scope": "tiny_model_correctness_not_official_or_performance",
        "device": args.device,
        "torch": torch.__version__,
        "cases": [],
    }
    for model in models:
        model.to(
            device=args.device, dtype=torch.bfloat16 if args.device == "cuda" else torch.float32
        ).eval()
        for loops in (1, model.config.total_ut_steps):
            for batch in args.batches:
                baseline = None
                for mode in ("native", "alias", "compact", "credits"):
                    torch.manual_seed(71)
                    blocks = batch * 8 * model.config.total_ut_steps
                    engine = LLMEngine(
                        model,
                        attention_backend="triton" if args.device == "cuda" else "torch",
                        cache_config=CacheConfig(
                            blocks,
                            2,
                            alias_last_exited=mode == "alias",
                            compact_last_exited=mode in ("compact", "credits"),
                            reclaim_skipped_credits=mode == "credits",
                        ),
                        scheduler_config=SchedulerConfig(
                            max_num_seqs=batch, max_num_batched_tokens=max(4, batch)
                        ),
                    )
                    for i in range(2 * batch):
                        engine.add_request(
                            str(i),
                            [2, 3 + i % 4, 7],
                            SamplingParams(
                                max_tokens=8,
                                min_loops=loops,
                                max_loops=loops,
                                ignore_eos=True,
                            ),
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
                    assert len(completed) == 2 * batch
                    assert engine.cache_manager.num_free_blocks == blocks
                    if isinstance(engine.cache_manager, CompactKVCacheManager):
                        assert (
                            engine.cache_manager.live_records
                            == engine.cache_manager._reserved_records
                            == 0
                        )
                    if baseline is None:
                        baseline = completed
                    assert baseline == completed, (type(model).__name__, loops, batch, mode)
                    engine.model_runner.synchronize()
                    engine.close()
                row = {
                    "model": type(model).__name__,
                    "loops": loops,
                    "batch": batch,
                    "requests": 2 * batch,
                    "exact_four_modes": True,
                }
                result["cases"].append(row)
                print(json.dumps(row), flush=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
