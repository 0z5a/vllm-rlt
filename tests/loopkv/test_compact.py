import pytest
import torch

from tests.helpers import tiny_ouro_config
from tests.loopkv.test_alias_runtime import make_cache
from vllm_rlt import LLM, CacheConfig, ExecutionConfig, SamplingParams, SchedulerConfig
from vllm_rlt.core.compact_kv_cache import CompactKVCacheManager
from vllm_rlt.core.kv_cache_manager import KVCacheManager
from vllm_rlt.models import OuroForCausalLM


def compact_payload_check(device):
    reference = make_cache(KVCacheManager, device)
    compact = make_cache(CompactKVCacheManager, device)
    generator = torch.Generator().manual_seed(71)
    counts = [4, 4, 4, 1, 4, 2, 1, 3]
    for cache in (reference, compact):
        assert cache.allocate("a", 9)
    compact.key_cache.fill_(float("nan"))
    compact.value_cache.fill_(float("nan"))
    assert compact.live_records == 0  # Admission creates credits, not payload records.
    for position, count in enumerate(counts):
        for depth in range(count):
            for layer in range(2):
                k, v, q = [
                    torch.randn(1, h, 32, generator=generator).to(device, compact.dtype)
                    for h in (2, 2, 4)
                ]
                results = []
                for cache in (reference, compact):
                    cache.write(layer, ["a"], [depth], [position], k, v)
                    results.append(cache.attend(layer, ["a"], [depth], [position], q))
                torch.testing.assert_close(results[0], results[1], rtol=0, atol=0)
        for cache in (reference, compact):
            cache.finalize_token("a", position, count - 1)
        allocation = compact._get_allocation("a")
        for depth in range(4):
            for layer in range(2):
                for expected, actual in zip(
                    reference.read(layer, "a", depth, position + 1),
                    compact.read(layer, "a", depth, position + 1),
                ):
                    assert torch.equal(expected, actual)
            if depth >= count:
                assert (depth, position) not in compact._records[id(allocation)]
                assert compact._maps[id(allocation)][depth, position].item() == -1
        assert compact.live_records == sum(counts[: position + 1])
    assert compact.peak_live_records == 23
    unused = torch.tensor(compact._free_records, device=device, dtype=torch.int64)
    for payload in (compact.key_cache, compact.value_cache):
        assert torch.isnan(
            payload[unused // compact.block_size, :, unused % compact.block_size]
        ).all()
    assert compact.promotion_copy_bytes == 0
    descriptor = compact._prepare_batch(["a"], [3], [7], for_write=False)
    compact.free("a")
    assert compact.live_records == 0 and compact.num_free_blocks == 48
    assert compact.allocate("a", 9)
    with pytest.raises(RuntimeError, match="stale"):
        compact._attend_prepared(
            0, descriptor, torch.zeros(1, 4, 32, device=device, dtype=compact.dtype)
        )
    assert (compact._maps[id(compact._get_allocation("a"))] == -1).all()


def test_compact_cpu_payload_and_reuse():
    compact_payload_check("cpu")


@pytest.mark.gpu
def test_compact_cuda_payload_and_reuse():
    compact_payload_check("cuda")


@pytest.mark.parametrize("batch", [1, 4, 16])
def test_compact_full_engine_and_conservative_admission(batch):
    torch.manual_seed(27)
    model = OuroForCausalLM(tiny_ouro_config()).eval()
    outputs = []
    for compact in (False, True):
        llm = LLM(
            model,
            cache_config=CacheConfig(128, 2, compact_last_exited=compact),
            scheduler_config=SchedulerConfig(
                max_num_seqs=batch, max_num_batched_tokens=max(batch, 8)
            ),
        )
        prompts = [[2, 3, 4][: 1 + i % 3] for i in range(batch * 2)]
        params = [
            SamplingParams(max_tokens=7, exit_threshold=(i % 4) / 3, ignore_eos=True)
            for i in range(len(prompts))
        ]
        outputs.append(llm.generate(prompts, params))
        assert llm.engine.cache_manager.num_free_blocks == 128
        if compact:
            assert llm.engine.cache_manager.live_records == 0
    assert [(o.token_ids, o.exit_depths) for o in outputs[0]] == [
        (o.token_ids, o.exit_depths) for o in outputs[1]
    ]


def test_compact_reserves_future_before_admitting():
    cache = make_cache(CompactKVCacheManager)
    assert cache.allocate("a", 13)  # 28 blocks of worst-future credits.
    assert cache.num_free_blocks == 20
    assert cache.live_records == 0
    assert not cache.allocate("b", 13)
    cache.free("a")
    assert cache.allocate("b", 13)


def test_compact_graph_and_transfer_rejected():
    from vllm_rlt.pd.engine import PDEngine

    config = CacheConfig(128, compact_last_exited=True)
    with pytest.raises(ValueError, match="alias storage"):
        LLM(
            OuroForCausalLM(tiny_ouro_config()),
            cache_config=config,
            execution_config=ExecutionConfig(static_buffers=True),
        )
    with pytest.raises(ValueError, match="alias KV transfer"):
        PDEngine(tiny_ouro_config(), decode_cache_config=config)


@pytest.mark.parametrize("cache_type", [KVCacheManager, CompactKVCacheManager])
def test_fixed_reservation_admits_the_same_initial_batch(cache_type):
    from vllm_rlt.core.scheduler import Scheduler
    from vllm_rlt.request import Request

    cache = make_cache(cache_type)
    scheduler = Scheduler(SchedulerConfig(max_num_seqs=4, max_num_batched_tokens=16), cache)
    for index in range(4):
        scheduler.add_request(Request(str(index), [2, 3, 4, 5], SamplingParams(max_tokens=3)))
    scheduler._admit()
    assert len(cache._allocations) == 4
    assert cache.num_free_blocks == 0
