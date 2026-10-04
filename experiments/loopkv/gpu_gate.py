import argparse
import json
import time
from pathlib import Path

import torch

from vllm_rlt import LLM, CacheConfig, SamplingParams, SchedulerConfig
from vllm_rlt.core.alias_kv_cache import AliasKVCacheManager
from vllm_rlt.core.compact_kv_cache import CompactKVCacheManager
from vllm_rlt.core.kv_cache_manager import KVCacheManager
from vllm_rlt.models import OuroConfig, OuroForCausalLM

parser = argparse.ArgumentParser()
parser.add_argument("--storage", choices=("alias", "compact"), default="alias")
parser.add_argument("--out", type=Path, default=Path("evidence/gpu-alias-gate-v1.json"))
args = parser.parse_args()
out = args.out
candidate_type = CompactKVCacheManager if args.storage == "compact" else AliasKVCacheManager
torch.set_num_threads(2)
torch.manual_seed(321)
torch.backends.cuda.matmul.allow_tf32 = False
caches = [
    cls(
        num_layers=2,
        num_kv_heads=2,
        head_dim=32,
        num_blocks=48,
        block_size=2,
        max_loops=4,
        dtype=torch.bfloat16,
        device="cuda",
        backend="triton",
    )
    for cls in (KVCacheManager, candidate_type)
]
for cache in caches:
    cache.key_cache.fill_(float("nan"))
    cache.value_cache.fill_(float("nan"))
    assert cache.allocate("a", 9)
identity = attention = poison = 0
for position, count in enumerate((4, 4, 4, 1, 4, 2, 1, 3)):
    for depth in range(count):
        for layer in range(2):
            k = torch.randn(1, 2, 32, device="cuda", dtype=torch.bfloat16)
            v = torch.randn_like(k)
            q = torch.randn(1, 4, 32, device="cuda", dtype=torch.bfloat16)
            results = []
            for cache in caches:
                cache.write(layer, ["a"], [depth], [position], k, v)
                results.append(cache.attend(layer, ["a"], [depth], [position], q))
            assert torch.equal(*results), (position, depth, layer)
            attention += 1
    for cache in caches:
        cache.finalize_token("a", position, count - 1)
    for depth in range(4):
        for layer in range(2):
            ref = caches[0].read(layer, "a", depth, position + 1)
            candidate = caches[1].read(layer, "a", depth, position + 1)
            assert all(torch.equal(a, b) for a, b in zip(ref, candidate))
            identity += 1
            if depth >= count:
                cache = caches[1]
                allocation = cache._get_allocation("a")
                if args.storage == "compact":
                    assert (depth, position) not in cache._records[id(allocation)]
                    assert cache._maps[id(allocation)][depth, position].item() == -1
                else:
                    block = allocation.block_tables[depth][position // 2]
                    assert torch.isnan(cache.key_cache[block, layer, position % 2]).all()
                assert position not in allocation.written[depth][layer]
                poison += 1
if args.storage == "compact":
    assert caches[1].live_records == 23
    assert caches[1].peak_live_records == 23
for cache in caches:
    cache.free("a")
rows = []
config = OuroConfig(
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
model = OuroForCausalLM(config).to(device="cuda", dtype=torch.bfloat16).eval()
for batch in (1, 4, 16, 32, 64, 128):
    prompts = [[2, 3, 4, 5][: 1 + i % 4] for i in range(2 * batch)]
    parameters = [
        SamplingParams(max_tokens=5, ignore_eos=True, exit_threshold=(0.0, 1.0, 0.5, 0.8)[i % 4])
        for i in range(len(prompts))
    ]
    arms = []
    for alias in (False, True):
        engine = LLM(
            model,
            attention_backend="triton",
            cache_config=CacheConfig(
                batch * 24,
                2,
                alias_last_exited=alias and args.storage == "alias",
                compact_last_exited=alias and args.storage == "compact",
            ),
            scheduler_config=SchedulerConfig(
                max_num_seqs=batch, max_num_batched_tokens=2 * batch, prefill_chunk_size=2
            ),
        )
        outputs = engine.generate(prompts, parameters)
        arms.append([(o.token_ids, o.exit_depths) for o in outputs])
        assert engine.engine.cache_manager.num_used_blocks == 0
        engine.close()
    assert arms[0] == arms[1], batch
    row = {"batch": batch, "requests": 2 * batch, "token_exit_parity": True}
    rows.append(row)
    print(json.dumps(row), flush=True)
torch.cuda.synchronize()
result = {
    "scope": "tiny_weight_cuda_correctness_not_official_e2e_performance",
    "storage": args.storage,
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "gpu": torch.cuda.get_device_name(),
    "capability": torch.cuda.get_device_capability(),
    "attention_bitwise_checks": attention,
    "payload_identity_checks": identity,
    "skipped_poison_checks": poison,
    "engine_cases": rows,
    "finished_at_unix": time.time(),
}
out.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result), flush=True)
