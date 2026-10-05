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


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
@pytest.mark.parametrize("prefix_lm", [False, True])
def test_batched_depth_metadata_matches_separate_preparation(monkeypatch, device, prefix_lm):
    candidate, reference = [make_cache(CompactKVCacheManager, device) for _ in range(2)]
    for cache in (candidate, reference):
        assert cache.allocate("a", 5) and cache.allocate("b", 3)
    ids, positions = ["a"] * 3 + ["b"] * 2, [0, 1, 2, 0, 1]
    read_lengths = [3, 3, 3, 2, 2] if prefix_lm else None
    staged = []
    stage = candidate._stage

    def record(values, dtype):
        staged.append(len(values) * torch.empty((), dtype=dtype).element_size())
        return stage(values, dtype)

    monkeypatch.setattr(candidate, "_stage", record)
    batches = candidate._prepare_batches(
        ids, [[depth] * 5 for depth in range(4)], positions, read_lengths=read_lengths
    )
    assert staged == [5 * (2 + 3 * 4) * 8, 5 * (2 + 4) * 4]
    common = batches[0]
    generator = torch.Generator().manual_seed(613)
    for depth, batch in enumerate(batches):
        single = reference._prepare_batch(ids, [depth] * 5, positions, read_lengths=read_lengths)
        assert batch.position_ids.data_ptr() == common.position_ids.data_ptr()
        assert batch.context_lengths.data_ptr() == common.context_lengths.data_ptr()
        assert batch.record_map_pointers.data_ptr() == common.record_map_pointers.data_ptr()
        assert batch.record_map_widths.data_ptr() == common.record_map_widths.data_ptr()
        assert torch.equal(batch.write_blocks, single.write_blocks)
        assert torch.equal(batch.write_offsets, single.write_offsets)
        for layer in range(2):
            keys, values, query = [
                torch.randn(5, heads, 32, generator=generator).to(device, candidate.dtype)
                for heads in (2, 2, 4)
            ]
            observed = []
            for cache, metadata in ((candidate, batch), (reference, single)):
                cache._write_prepared(layer, metadata, keys, values)
                observed.append(cache._attend_prepared(layer, metadata, query))
            torch.testing.assert_close(observed[0], observed[1], atol=0, rtol=0)
    for cache in (candidate, reference):
        for rid in ("a", "b"):
            cache.free(rid)
        assert cache.live_records == cache._reserved_records == 0
    assert candidate.allocate("a", 5)
    with pytest.raises(RuntimeError, match="stale"):
        candidate._attend_prepared(0, batches[-1], query)
    candidate.free("a")


def test_batched_prefill_rejects_unqualified_execution():
    for options in ({"static_buffers": True}, {"cuda_graphs": True}, {"prefill_uva": True}):
        with pytest.raises(ValueError, match="eager execution"):
            ExecutionConfig(prefill_batch_metadata=True, **options)
    with pytest.raises(ValueError, match="boolean"):
        ExecutionConfig(prefill_batch_metadata=1)
    for layout, backend in (("shared", "torch"), ("last_exited", "fa4")):
        with pytest.raises(ValueError, match="batched prefill metadata requires"):
            LLM(
                OuroForCausalLM(tiny_ouro_config()),
                cache_config=CacheConfig(128, 2, layout=layout),
                execution_config=ExecutionConfig(prefill_batch_metadata=True),
                attention_backend=backend,
            )


@pytest.mark.parametrize("batch", [1, 4, 16])
def test_compact_full_engine_and_conservative_admission(batch):
    torch.manual_seed(27)
    model = OuroForCausalLM(tiny_ouro_config()).eval()
    outputs = []
    for compact, batched in ((False, False), (False, True), (True, False), (True, True)):
        llm = LLM(
            model,
            cache_config=CacheConfig(128, 2, compact_last_exited=compact),
            scheduler_config=SchedulerConfig(
                max_num_seqs=batch, max_num_batched_tokens=max(batch, 8)
            ),
            execution_config=ExecutionConfig(prefill_batch_metadata=batched),
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
    reference = [(o.token_ids, o.exit_depths) for o in outputs[0]]
    assert all([(o.token_ids, o.exit_depths) for o in result] == reference for result in outputs)


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


def test_reclaimed_past_credits_survive_future_full_depth():
    cache = CompactKVCacheManager(
        2,
        2,
        32,
        48,
        2,
        max_loops=4,
        dtype=torch.float32,
        device="cpu",
        backend="torch",
        reclaim_skipped_credits=True,
    )
    assert cache.allocate("a", 13)
    assert not cache.allocate("b", 13)

    def write_position(rid, position, loops):
        for depth in range(loops):
            for layer in range(2):
                payload = torch.full((1, 2, 32), position * 100.0 + depth * 10 + layer)
                cache.write(layer, [rid], [depth], [position], payload, -payload)
                assert cache.live_records <= cache._reserved_records <= 96
        cache.finalize_token(rid, position, loops - 1)
        assert cache.live_records <= cache._reserved_records <= 96

    for position in range(6):
        write_position("a", position, 1)
    assert cache._reserved_records == 38  # 56 initial, 18 provably unused past versions.
    assert cache.allocate("b", 13)
    assert cache._reserved_records == 94
    # Every future token switches to maximum depth; no new credit is needed.
    for position in range(6, 13):
        write_position("a", position, 4)
    for position in range(13):
        write_position("b", position, 4)
    assert cache.live_records == 86
    for rid in ("a", "b"):
        for depth in range(4):
            keys, values = cache.read(0, rid, depth, 13)
            expected = torch.tensor(
                [p * 100.0 + (0 if rid == "a" and p < 6 else depth) * 10 for p in range(13)]
            )[:, None, None].expand_as(keys)
            assert torch.equal(keys, expected)
            assert torch.equal(values, -expected)
        cache.free(rid)
    assert cache.live_records == cache._reserved_records == 0
    assert cache.num_free_blocks == 48


def test_prepared_unwritten_versions_keep_their_credit():
    cache = CompactKVCacheManager(
        2,
        2,
        32,
        48,
        2,
        max_loops=4,
        dtype=torch.float32,
        device="cpu",
        backend="torch",
        reclaim_skipped_credits=True,
    )
    assert cache.allocate("a", 3)
    batches = cache._prepare_batches(["a"], [[0], [1], [2], [3]], [0])
    payload = torch.ones(1, 2, 32)
    for layer in range(2):
        cache._write_prepared(layer, batches[0], payload, payload)
    before = cache._reserved_records
    cache.finalize_token("a", 0, 0)
    assert cache._reserved_records == before
    with pytest.raises(ValueError, match="overwrite"):
        cache._write_prepared(0, batches[3], payload, payload)
    cache.free("a")
    assert cache.live_records == cache._reserved_records == 0


def test_credit_pressure_engine_completes_after_depth_increase():
    from vllm_rlt.engine.llm_engine import LLMEngine

    torch.manual_seed(32)
    model = OuroForCausalLM(tiny_ouro_config()).eval()
    arms, peaks = [], []
    for compact, reclaim in ((False, False), (True, False), (True, True)):
        engine = LLMEngine(
            model,
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
                    engine.cache_manager.live_records
                    <= engine.cache_manager._reserved_records
                    <= 192
                )
        assert len(completed) == 16
        assert engine.cache_manager.num_free_blocks == 96
        arms.append(completed)
        peaks.append(peak)
        engine.close()
    assert arms[0] == arms[1] == arms[2]
    assert peaks[0] == peaks[1] == 2
    assert peaks[2] > 2
