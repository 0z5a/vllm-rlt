"""Verify a two-GPU NIXL KV WRITE without starting worker processes."""

import json
import statistics
import time

import torch

from vllm_rlt.core.kv_cache_manager import KVCacheManager
from vllm_rlt.pd.config import PDConfig
from vllm_rlt.pd.transport import NixlConnector, kv_segments, partition_segments


def cache(device):
    result = KVCacheManager(
        num_layers=2,
        num_kv_heads=2,
        head_dim=64,
        num_blocks=256,
        block_size=16,
        max_loops=4,
        device=f"cuda:{device}",
        dtype=torch.bfloat16,
        backend="triton",
    )
    assert result.allocate("probe", 1024)
    return result


def main():
    source_cache, target_cache = cache(0), cache(1)
    for depth in range(4):
        for block in source_cache.get_block_table("probe", depth):
            source_cache.key_cache[block].fill_(depth + 1)
            source_cache.value_cache[block].fill_(depth + 11)
    target_cache.key_cache.zero_()
    target_cache.value_cache.zero_()
    source_hidden = torch.ones((1, 64), device="cuda:0", dtype=torch.bfloat16)
    target_hidden = torch.zeros_like(source_hidden, device="cuda:1")
    torch.cuda.synchronize(0)
    torch.cuda.synchronize(1)

    config = PDConfig(max_transfer_descriptors=1024)
    source = NixlConnector("probe-source", source_cache, source_hidden, config)
    target = NixlConnector("probe-target", target_cache, target_hidden, config)
    source.connect(target.info)
    target.connect(source.info)
    source_tables = source_cache._get_allocation("probe").block_tables
    target_tables = target_cache._get_allocation("probe").block_tables
    local = list(kv_segments(source.info, source_tables, 0, 1024))
    remote = list(kv_segments(target.info, target_tables, 0, 1024))
    local.append((source.info["hidden_ptr"], source.info["hidden_bytes"], 0))
    remote.append((target.info["hidden_ptr"], target.info["hidden_bytes"], 1))
    chunks = list(partition_segments(local, remote, config.transfer_chunk_bytes, 1024))
    assert len(chunks) == 1
    left, right, size = chunks[0]
    samples = []
    for sequence in range(7):
        before = source.transfer_seconds
        assert source.submit("probe", sequence, target.info["agent"], left, right, size)
        notifications = []
        deadline = time.monotonic() + 60
        while source.pending or not notifications:
            source.poll()
            _, received = target.poll()
            notifications.extend(received)
            assert time.monotonic() < deadline, "NIXL transfer timed out"
            time.sleep(0.001)
        assert notifications == [(source.info["agent"], "probe", sequence)]
        samples.append(source.transfer_seconds - before)
    target_cache.mark_imported_prefix("probe", 1024)
    for depth in range(4):
        for source_block, target_block in zip(
            source_tables[depth], target_tables[depth], strict=True
        ):
            assert torch.equal(
                source_cache.key_cache[source_block],
                target_cache.key_cache[target_block].to("cuda:0"),
            )
            assert torch.equal(
                source_cache.value_cache[source_block],
                target_cache.value_cache[target_block].to("cuda:0"),
            )
    assert torch.equal(source_hidden, target_hidden.to("cuda:0"))
    print(
        json.dumps(
            {
                "bytes": size,
                "transfer_seconds_median": statistics.median(samples[1:]),
                "throughput_gib_s": size / statistics.median(samples[1:]) / 1024**3,
                "samples_s": samples,
                "verified_tokens": 1024,
                "depths": 4,
            }
        )
    )
    source.close()
    target.close()


if __name__ == "__main__":
    main()
