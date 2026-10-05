"""Finite LoopFormer CUDA causal oracle and resident/cache lifetime qualification."""

import argparse
import gc
from dataclasses import asdict
from pathlib import Path

import torch

from experiments.loopkv.capture import dump
from tests.reference.loopformer import dense_loopformer_reference
from vllm_rlt import CacheConfig, ExecutionConfig, ExitConfig, SamplingParams, SchedulerConfig
from vllm_rlt.core.kv_cache_manager import KVCacheManager
from vllm_rlt.engine.llm_engine import LLMEngine
from vllm_rlt.models import LoopFormerConfig, LoopFormerForCausalLM


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(71)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    config = LoopFormerConfig(
        vocab_size=64,
        n_embd=32,
        n_head=4,
        n_layer=2,
        intermediate_dim=64,
        block_size=64,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    model = LoopFormerForCausalLM(config).eval()
    tokens = torch.tensor([3, 7, 4, 2, 6, 8, 9])
    states, logits, kv = dense_loopformer_reference(model, tokens)
    model.cuda()
    cache = KVCacheManager(
        num_layers=config.num_hidden_layers,
        num_kv_heads=4,
        head_dim=8,
        num_blocks=64,
        block_size=2,
        max_loops=8,
        device="cuda",
        dtype=torch.float32,
        backend="triton",
    )
    assert cache.allocate("oracle", 7)
    for positions in [list(range(5)), [5], [6]]:
        hidden = model.prelude(tokens[positions].cuda())
        for depth in range(8):
            batch = cache._prepare_batch(
                ["oracle"] * len(positions),
                [depth] * len(positions),
                positions,
            )
            hidden, _ = model.recurrent_prepared(hidden, batch, cache)
            torch.testing.assert_close(hidden.cpu(), states[depth][positions], atol=3e-6, rtol=3e-5)
        torch.testing.assert_close(
            model.coda(hidden).cpu(), logits[positions], atol=3e-6, rtol=3e-5
        )
        for position in positions:
            cache.finalize_token("oracle", position, 7)
    for (depth, layer), expected in kv.items():
        for result, value in zip(cache.read(layer, "oracle", depth, 7), expected):
            torch.testing.assert_close(result.cpu(), value, atol=3e-6, rtol=3e-5)
    cache.free("oracle")
    del cache, states, logits, kv
    model.bfloat16()
    rows = []
    for batch_limit in (1, 4, 16, 32, 64, 128):
        for mode in ("sync", "async", "graph"):
            baseline = None
            for storage in ("native", "alias", "compact", "credits"):
                blocks = batch_limit * 64
                engine = LLMEngine(
                    model,
                    attention_backend="triton",
                    cache_config=CacheConfig(
                        blocks,
                        2,
                        alias_last_exited=storage == "alias",
                        compact_last_exited=storage in ("compact", "credits"),
                        reclaim_skipped_credits=storage == "credits",
                    ),
                    scheduler_config=SchedulerConfig(
                        max_num_seqs=batch_limit,
                        max_num_batched_tokens=max(64, batch_limit),
                        prefill_chunk_size=1,
                    ),
                    execution_config=ExecutionConfig(
                        async_scheduling=mode != "sync",
                        static_buffers=mode == "graph",
                        cuda_graphs=mode == "graph",
                    ),
                    exit_config=ExitConfig("ouro" if mode == "sync" else "ouro_delayed"),
                )
                completed_rounds = []
                for _ in range(2):
                    for index in range(2 * batch_limit):
                        engine.add_request(
                            str(index),
                            [3, 7, 4, 2, 6][: 3 + index % 3],
                            SamplingParams(
                                max_tokens=6,
                                min_loops=8,
                                max_loops=8,
                                exit_threshold=1,
                                ignore_eos=True,
                                seed=17 + index,
                            ),
                        )
                    completed = {}
                    while engine.has_unfinished_requests():
                        for output in engine.step():
                            if output.finished:
                                completed[output.request_id] = asdict(output)
                    assert len(completed) == 2 * batch_limit
                    assert all(r["exit_depths"] == [8] * 6 for r in completed.values())
                    assert engine.cache_manager.num_free_blocks == blocks
                    completed_rounds.append(completed)
                assert completed_rounds[0] == completed_rounds[1]
                if baseline is None:
                    baseline = completed_rounds[0]
                assert baseline == completed_rounds[0], (batch_limit, mode, storage)
                graph = engine.model_runner.graphs
                graph_stats = None
                if graph:
                    assert graph.captures > 0 and graph.replays > 0
                    graph_stats = dict(
                        captures=graph.captures, replays=graph.replays, fallbacks=graph.fallbacks
                    )
                rows.append(
                    dict(
                        batch=batch_limit,
                        concurrency=2 * batch_limit,
                        mode=mode,
                        storage=storage,
                        two_lifetimes_exact=True,
                        graphs=graph_stats,
                    )
                )
                print(rows[-1], flush=True)
                engine.close()
                del graph, engine
                gc.collect()
                torch.cuda.empty_cache()
    dump(
        args.out,
        dict(
            scope="tiny_LoopFormer_not_official_or_performance",
            torch=torch.__version__,
            gpu=torch.cuda.get_device_name(),
            dense_FP32_oracle=True,
            atol=3e-6,
            rtol=3e-5,
            cases=rows,
        ),
    )


if __name__ == "__main__":
    main()
