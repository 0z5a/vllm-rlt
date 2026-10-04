import pytest
import torch

from tests.helpers import tiny_ouro_config
from vllm_rlt import LLM, CacheConfig, ExecutionConfig, SamplingParams, SchedulerConfig
from vllm_rlt.core.alias_kv_cache import AliasKVCacheManager
from vllm_rlt.core.kv_cache_manager import KVCacheManager
from vllm_rlt.models import OuroForCausalLM


def make_cache(cls, device="cpu"):
    return cls(
        num_layers=2,
        num_kv_heads=2,
        head_dim=32,
        num_blocks=48,
        block_size=2,
        max_loops=4,
        dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device=device,
        backend="triton" if device == "cuda" else "torch",
    )


def check_payload_attention(device):
    reference = make_cache(KVCacheManager, device)
    alias = make_cache(AliasKVCacheManager, device)
    for cache in (reference, alias):
        cache.key_cache.fill_(float("nan"))
        cache.value_cache.fill_(float("nan"))
        assert cache.allocate("a", 9)
    generator = torch.Generator().manual_seed(321)
    # Full prompt at 0..2, then heterogeneous exits spanning page boundaries.
    for position, count in enumerate((4, 4, 4, 1, 4, 2, 1, 3)):
        for depth in range(count):
            for layer in range(2):
                keys = torch.randn(1, 2, 32, generator=generator).to(device, alias.dtype)
                values = torch.randn(1, 2, 32, generator=generator).to(device, alias.dtype)
                query = torch.randn(1, 4, 32, generator=generator).to(device, alias.dtype)
                outputs = []
                for cache in (reference, alias):
                    cache.write(layer, ["a"], [depth], [position], keys, values)
                    outputs.append(cache.attend(layer, ["a"], [depth], [position], query))
                torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
        for cache in (reference, alias):
            cache.finalize_token("a", position, count - 1)
        for depth in range(4):
            for layer in range(2):
                a = reference.read(layer, "a", depth, position + 1)
                b = alias.read(layer, "a", depth, position + 1)
                for expected, actual in zip(a, b):
                    assert torch.equal(expected, actual)
                if depth >= count:
                    allocation = alias._get_allocation("a")
                    block = allocation.block_tables[depth][position // 2]
                    assert torch.isnan(alias.key_cache[block, layer, position % 2]).all()
                    assert position not in allocation.written[depth][layer]
    assert alias.promotion_copy_bytes == 0
    # Test materialization is counted separately from GPU attention.
    if device == "cuda":
        before = alias.materialization_bytes
        alias.attend(0, ["a"], [3], [7], torch.zeros(1, 4, 32, device=device, dtype=alias.dtype))
        assert alias.materialization_bytes == before


def test_cpu_payload_poison_and_attention():
    check_payload_attention("cpu")


@pytest.mark.gpu
def test_cuda_payload_poison_and_bitwise_attention():
    check_payload_attention("cuda")


def test_reuse_rejects_stale_descriptor_and_resets_metadata():
    cache = make_cache(AliasKVCacheManager)
    cache.allocate("a", 3)
    batch = cache._prepare_batch(["a"], [0], [0])
    for layer in range(2):
        cache._write_prepared(layer, batch, torch.ones(1, 2, 32), torch.ones(1, 2, 32))
    cache.finalize_token("a", 0, 0)
    cache.free("a")
    cache.free("a")
    assert cache.num_used_blocks == 0
    cache.allocate("a", 3)
    with pytest.raises(RuntimeError, match="stale"):
        cache._attend_prepared(0, batch, torch.zeros(1, 4, 32))
    for block in cache.get_block_table("a", 0):
        assert (cache.source_depths[block] == -1).all()
    with pytest.raises(RuntimeError, match="uninitialized"):
        cache.read(0, "a", 3, 1)


@pytest.mark.parametrize("batch", [1, 2, 4, 8])
@pytest.mark.parametrize("mode", ["refill", "no_refill"])
def test_tiny_model_token_and_exit_parity(batch, mode):
    torch.manual_seed(27)
    model = OuroForCausalLM(tiny_ouro_config()).eval()
    prompts = [[2, 3, 4, 5][: 1 + i % 4] for i in range(batch * 2)]
    parameters = [
        SamplingParams(max_tokens=5, exit_threshold=t, ignore_eos=True)
        for t in ([0.0, 1.0, 0.5, 0.8] * (len(prompts) // 4 + 1))[: len(prompts)]
    ]
    outputs = []
    for use_alias in (False, True):
        llm = LLM(
            model,
            cache_config=CacheConfig(256, 2, alias_last_exited=use_alias),
            scheduler_config=SchedulerConfig(
                max_num_seqs=batch,
                max_num_batched_tokens=batch * 2,
                prefill_chunk_size=2,
                mode=mode,
            ),
        )
        outputs.append(llm.generate(prompts, parameters))
        assert llm.engine.cache_manager.num_used_blocks == 0
    assert [(o.token_ids, o.exit_depths) for o in outputs[0]] == [
        (o.token_ids, o.exit_depths) for o in outputs[1]
    ]


def test_unsupported_graph_rejected_before_cache_allocation():
    model = OuroForCausalLM(tiny_ouro_config()).eval()
    with pytest.raises(ValueError, match="alias storage"):
        LLM(
            model,
            cache_config=CacheConfig(64, alias_last_exited=True),
            execution_config=ExecutionConfig(static_buffers=True),
        )


def test_finalized_source_is_immutable():
    cache = make_cache(AliasKVCacheManager)
    cache.allocate("a", 3)
    payload = torch.ones(1, 2, 32)
    for layer in range(2):
        cache.write(layer, ["a"], [0], [0], payload, payload)
    cache.finalize_token("a", 0, 0)
    with pytest.raises(ValueError, match="overwrite"):
        cache.write(0, ["a"], [0], [0], payload * 2, payload)
    assert torch.equal(cache.read(0, "a", 3, 1)[0], payload)


def test_pd_alias_rejected_before_worker_start():
    from vllm_rlt.pd.engine import PDEngine

    with pytest.raises(ValueError, match="alias KV transfer"):
        PDEngine(tiny_ouro_config(), decode_cache_config=CacheConfig(alias_last_exited=True))


def test_direct_depths_and_first_alias_metadata():
    cache = make_cache(AliasKVCacheManager)
    assert cache.allocate("a", 6)
    payload = torch.ones(1, 2, 32)
    for position, loops in enumerate((4, 2, 1)):
        for depth in range(loops):
            for layer in range(2):
                cache.write(layer, ["a"], [depth], [position], payload, payload)
        cache.finalize_token("a", position, loops - 1)
    batches = cache._prepare_batches(["a"], [[0], [1], [2], [3]], [2], for_write=False)
    assert batches[0].depth_block_tables is None
    for batch, expected in zip(batches[1:], (2, 1, 1)):
        assert batch.alias_starts.tolist() == [expected]
        assert batch.query_depths.tolist() == [batch.rows[0][1]]
        # All alias metadata in the traversal shares one staged buffer.
        assert (
            batch.alias_starts.untyped_storage().data_ptr()
            == batches[1].query_depths.untyped_storage().data_ptr()
        )
    allocation = cache._get_allocation("a")
    block = allocation.block_tables[0][0]
    assert cache.source_depths[block, 0] == -1  # Full-depth needs no publication.
    cache.free("a")
    assert cache.allocate("a", 6)
    assert cache._prepare_batch(["a"], [3], [2]).alias_starts is None
