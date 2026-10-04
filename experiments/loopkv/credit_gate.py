"""Finite CUDA pressure gate; synthetic depths, no performance claim."""

import argparse
import json
import subprocess
from pathlib import Path

import torch

from vllm_rlt import CacheConfig, SamplingParams, SchedulerConfig
from vllm_rlt.core.compact_kv_cache import CompactKVCacheManager
from vllm_rlt.engine.llm_engine import LLMEngine
from vllm_rlt.models import OuroConfig, OuroForCausalLM

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
torch.set_num_threads(2)
torch.manual_seed(32)
cache = CompactKVCacheManager(
    2,
    2,
    32,
    48,
    2,
    max_loops=4,
    dtype=torch.bfloat16,
    device="cuda",
    backend="triton",
    reclaim_skipped_credits=True,
)
assert cache.allocate("a", 13) and not cache.allocate("b", 13)
for rid, positions, loops in [("a", range(6), 1), ("a", range(6, 13), 4), ("b", range(13), 4)]:
    if rid == "a" and positions.start == 6:
        assert cache._reserved_records == 38
        assert cache.allocate("b", 13)
    for position in positions:
        for depth in range(loops):
            for layer in range(2):
                value = torch.full(
                    (1, 2, 32),
                    position * 100.0 + depth * 10 + layer,
                    device="cuda",
                    dtype=cache.dtype,
                )
                cache.write(layer, [rid], [depth], [position], value, -value)
                assert cache.live_records <= cache._reserved_records <= 96
        cache.finalize_token(rid, position, loops - 1)
assert cache.live_records == 86 and cache._reserved_records == 94
for rid in ("a", "b"):
    for depth in range(4):
        keys, values = cache.read(0, rid, depth, 13)
        expected = torch.tensor(
            [p * 100.0 + (0 if rid == "a" and p < 6 else depth) * 10 for p in range(13)],
            device="cuda",
            dtype=cache.dtype,
        )[:, None, None].expand_as(keys)
        assert torch.equal(keys, expected) and torch.equal(values, -expected)
    cache.free(rid)
assert cache.num_free_blocks == 48 and cache.live_records == cache._reserved_records == 0
model = (
    OuroForCausalLM(
        OuroConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            max_position_embeddings=128,
            total_ut_steps=4,
        )
    )
    .to(device="cuda", dtype=torch.bfloat16)
    .eval()
)
outputs, peaks = [], []
for compact, reclaim in ((False, False), (True, False), (True, True)):
    engine = LLMEngine(
        model,
        attention_backend="triton",
        cache_config=CacheConfig(
            96, 2, compact_last_exited=compact, reclaim_skipped_credits=reclaim
        ),
        scheduler_config=SchedulerConfig(max_num_seqs=8, max_num_batched_tokens=16),
    )
    for index in range(16):
        loops = 2 if index < 8 else 4
        engine.add_request(
            str(index),
            [2, 3, 4, 5],
            SamplingParams(max_tokens=16, min_loops=loops, max_loops=loops, ignore_eos=True),
        )
    completed, peak = {}, 0
    while engine.has_unfinished_requests():
        for output in engine.step():
            if output.finished:
                completed[output.request_id] = (output.token_ids, output.exit_depths)
        peak = max(peak, len(engine.cache_manager._allocations))
        if compact:
            assert (
                engine.cache_manager.live_records <= engine.cache_manager._reserved_records <= 192
            )
    assert len(completed) == 16 and engine.cache_manager.num_free_blocks == 96
    outputs.append(completed)
    peaks.append(peak)
    engine.close()
assert outputs[0] == outputs[1] == outputs[2]
assert peaks[0] == peaks[1] == 2 and peaks[2] > 2
torch.cuda.synchronize()
result = {
    "scope": "synthetic_tiny_cuda_pressure_not_official_model_or_performance",
    "source_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "torch": torch.__version__,
    "gpu": torch.cuda.get_device_name(),
    "future_full_depth_after_six_r1_positions": True,
    "exact_16_request_outputs": True,
    "peak_residents_native_fixed_credits": peaks,
    "all_records_and_credits_returned": True,
}
args.out.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result), flush=True)
