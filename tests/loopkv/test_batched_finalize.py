import pytest
import torch

from tests.loopkv.test_alias_runtime import make_cache
from vllm_rlt.core.alias_kv_cache import AliasKVCacheManager
from vllm_rlt.core.compact_kv_cache import CompactKVCacheManager


@pytest.mark.parametrize("cache_type", [AliasKVCacheManager, CompactKVCacheManager])
def test_batched_mixed_exits_preserve_each_version_and_full_depth(cache_type):
    cache = make_cache(cache_type)
    for rid in ("a", "b"):
        assert cache.allocate(rid, 4)
    for position in range(3):
        counts = {"a": (4, 1, 3)[position], "b": (2, 4, 1)[position]}
        for rid, count in counts.items():
            for depth in range(count):
                value = torch.full((1, 2, 32), position * 10.0 + depth + (100 if rid == "b" else 0))
                for layer in range(2):
                    cache.write(layer, [rid], [depth], [position], value, -value)
        cache.finalize_tokens((rid, position, count - 1) for rid, count in counts.items())
        for rid, count in counts.items():
            for depth in range(4):
                expected = position * 10.0 + min(depth, count - 1) + (100 if rid == "b" else 0)
                key, value = cache.read(0, rid, depth, position + 1)
                assert torch.equal(key[-1], torch.full_like(key[-1], expected))
                assert torch.equal(value[-1], -key[-1])
    with pytest.raises(ValueError, match="finalized"):
        cache.finalize_tokens([("a", 2, 2)])
    for rid in ("a", "b"):
        cache.free(rid)
    assert cache.num_free_blocks == 48
