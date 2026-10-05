import torch

from tests.helpers import tiny_ouro_config
from vllm_rlt.core.kv_cache_manager import KVCacheManager
from vllm_rlt.worker.buffers import Workspace


def test_device_loop_metadata_tracks_reordering_padding_and_request_reuse():
    config = tiny_ouro_config()
    cache = KVCacheManager(
        config.num_hidden_layers,
        config.num_key_value_heads,
        config.head_dim,
        num_blocks=32,
        block_size=2,
        max_loops=4,
    )
    assert cache.allocate("a", 4) and cache.allocate("b", 4)
    batches = cache._prepare_batches(["b", "a"], [[3, 0], [1, 2]], [1, 0])
    assert [batch.loop_ids.tolist() for batch in batches] == [[3, 0], [1, 2]]
    workspace = Workspace(4, 2, config.hidden_size, torch.device("cpu"), torch.float32)
    prepared = workspace.prepare(cache, ["a", "b"], [2, 1], [0, 1], 4)
    assert prepared.loop_ids.tolist() == [2, 1, -1, -1]
    cache.free("a")
    assert cache.allocate("a", 4)
    prepared = workspace.prepare(cache, ["a"], [0], [0], 4)
    assert prepared.loop_ids.tolist() == [0, -1, -1, -1]
