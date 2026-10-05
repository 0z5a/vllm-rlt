"""Finite Parcae CUDA oracle and same-backend storage comparison; no speed claim."""

import argparse
import gc
from dataclasses import asdict
from pathlib import Path

import torch

from experiments.loopkv.capture import dump
from tests.reference.parcae import dense_parcae_reference
from vllm_rlt import CacheConfig, SamplingParams, SchedulerConfig
from vllm_rlt.core.kv_cache_manager import KVCacheManager
from vllm_rlt.engine.llm_engine import LLMEngine
from vllm_rlt.models import ParcaeConfig, ParcaeForCausalLM


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(41)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    config = ParcaeConfig(
        n_embd=32,
        intermediate_size=64,
        num_attention_heads=4,
        num_key_value_heads=4,
        n_layers_in_prelude=2,
        n_layers_in_recurrent_block=2,
        n_layers_in_coda=2,
        mean_recurrence=3,
        block_size=64,
        vocab_size=1024,
    )
    model = ParcaeForCausalLM(config).eval()
    tokens = torch.tensor([257, 511, 9, 768, 1023, 256, 255])
    initial = torch.linspace(-0.1, 0.1, len(tokens) * config.n_embd).reshape(len(tokens), -1)
    states, logits, kv = dense_parcae_reference(model, tokens, initial)
    model.cuda()
    cache = KVCacheManager(
        config.num_hidden_layers,
        4,
        8,
        128,
        2,
        max_loops=3,
        device="cuda",
        dtype=torch.float32,
        backend="triton",
        recurrent_layers=model.recurrent_kv_layers,
    )
    cache.allocate("oracle", 7)
    for positions in [list(range(5)), [5], [6]]:
        boundary = cache._prepare_batch(
            ["oracle"] * len(positions), [0] * len(positions), positions
        )
        hidden = model.prelude_prepared(tokens[positions].cuda(), boundary, cache)
        hidden = torch.cat((initial[positions].cuda(), hidden[:, config.n_embd :]), -1)
        for depth in range(3):
            core = cache._prepare_batch(
                ["oracle"] * len(positions), [depth] * len(positions), positions
            )
            hidden, _ = model.recurrent_prepared(hidden, core, cache)
            torch.testing.assert_close(
                hidden[:, : config.n_embd].cpu(), states[depth][positions], atol=3e-6, rtol=3e-5
            )
        for position in positions:
            cache.finalize_token("oracle", position, 2)
        observed = model.coda_prepared(hidden, boundary, cache)
        torch.testing.assert_close(observed.cpu(), logits[positions], atol=3e-6, rtol=3e-5)
    for (layer, depth), expected in kv.items():
        for value, reference in zip(cache.read(layer, "oracle", depth, 7), expected):
            torch.testing.assert_close(value.cpu(), reference, atol=3e-6, rtol=3e-5)
    cache.free("oracle")
    del cache, states, logits, kv
    model.bfloat16()
    rows = []
    for batch_limit in (1, 4, 16, 32, 64, 128, 256, 512):
        baseline = None
        for storage in ("native", "alias", "compact", "credits"):
            blocks = batch_limit * 32
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
            )
            rounds = []
            for _ in range(2):
                torch.manual_seed(83)
                for index in range(2 * batch_limit):
                    engine.add_request(
                        str(index),
                        [257, 511, 9, 768, 1023][: 3 + index % 3],
                        SamplingParams(
                            max_tokens=6,
                            min_loops=3,
                            max_loops=3,
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
                assert all(row["exit_depths"] == [3] * 6 for row in completed.values())
                assert engine.cache_manager.num_free_blocks == blocks
                rounds.append(completed)
            assert rounds[0] == rounds[1]
            if baseline is None:
                baseline = rounds[0]
            assert baseline == rounds[0], (batch_limit, storage)
            rows.append(
                dict(
                    batch=batch_limit,
                    concurrency=2 * batch_limit,
                    storage=storage,
                    mode="sync",
                    two_lifetimes_exact=True,
                )
            )
            print(rows[-1], flush=True)
            engine.close()
            del engine
            gc.collect()
            torch.cuda.empty_cache()
    dump(
        args.out,
        dict(
            scope="tiny Parcae same-backend storage, not BF16 author qualification or speed",
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
