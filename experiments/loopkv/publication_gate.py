"""Finite CUDA gate for mixed exits around publication-kernel tile boundaries."""

import argparse
import json
from pathlib import Path

import torch

from vllm_rlt.core.alias_kv_cache import AliasKVCacheManager
from vllm_rlt.core.compact_kv_cache import CompactKVCacheManager


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    cases = []
    for cache_type in (AliasKVCacheManager, CompactKVCacheManager):
        for count in (1, 127, 128, 129, 257):
            cache = cache_type(
                2,
                2,
                32,
                count * 4,
                2,
                max_loops=4,
                dtype=torch.bfloat16,
                device="cuda",
                backend="triton",
            )
            cache.key_cache.fill_(-11)
            cache.value_cache.fill_(-12)
            ids = [str(i) for i in range(count)]
            exits = [i % 4 for i in range(count)]
            for rid in ids:
                assert cache.allocate(rid, 1)
            for depth in range(4):
                selected = [rid for rid, exit_depth in zip(ids, exits) if depth <= exit_depth]
                if not selected:
                    continue
                for layer in range(2):
                    values = (
                        torch.tensor(
                            [int(rid) + depth * 4 + layer for rid in selected],
                            device="cuda",
                            dtype=cache.dtype,
                        )[:, None, None]
                        .expand(-1, 2, 32)
                        .contiguous()
                    )
                    cache.write(
                        layer,
                        selected,
                        [depth] * len(selected),
                        [0] * len(selected),
                        values,
                        -values,
                    )
            before_k, before_v = cache.key_cache.clone(), cache.value_cache.clone()
            cache.finalize_tokens(zip(ids, [0] * count, exits))
            assert torch.equal(cache.key_cache, before_k)
            assert torch.equal(cache.value_cache, before_v)
            for rid, exit_depth in zip(ids, exits):
                for depth in range(4):
                    for layer in range(2):
                        key, value = cache.read(layer, rid, depth, 1)
                        expected = torch.full_like(
                            key, int(rid) + min(depth, exit_depth) * 4 + layer
                        )
                        assert torch.equal(key, expected) and torch.equal(value, -expected)
                cache.free(rid)
            assert cache.num_free_blocks == count * 4
            row = {
                "storage": cache_type.__name__,
                "requests": count,
                "exact_versions": True,
                "payload_unchanged": True,
            }
            cases.append(row)
            print(json.dumps(row), flush=True)
    args.out.write_text(
        json.dumps(
            {
                "scope": "cuda_publication_correctness_not_performance",
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
