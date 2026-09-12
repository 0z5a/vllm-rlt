"""Scheduler coverage, allocation accounting and alternating bucket correctness."""

import pytest
import test_recurrent_graph as fixtures
import torch

from vllm_lt.config import CacheConfig, SchedulerConfig
from vllm_lt.engine.llm_engine import LLMEngine
from vllm_lt.worker.decode_buffers import DecodeBucketLayout, allocate_bucket
from vllm_lt.worker.model_runner import ModelRunner

model = fixtures.model
no_cuda = fixtures.no_cuda


@pytest.mark.parametrize(
    "capacity,rows", [(1, (4,)), (4, (4, 8)), (8, (4, 8, 16)), (16, (4, 8, 16, 32))]
)
def test_scheduler_capacity_reaches_both_executors(model, capacity, rows):
    engine = LLMEngine(
        model,
        cache_config=CacheConfig(num_blocks=160),
        scheduler_config=SchedulerConfig(max_num_seqs=capacity),
    )
    runner = engine.model_runner
    assert runner._decode_layout.row_counts == rows
    graph = ModelRunner(model, fixtures.make_cache(model), max_num_seqs=capacity)
    runner._enable_persistent_decode()
    graph._enable_recurrent_graph(use_graphs=False)
    assert tuple(runner._persistent["buckets"]) == tuple(graph._decode_executor.buckets) == rows
    for count in range(1, capacity + 1):
        bucket, reason = runner._decode_layout.select(count, 32)
        assert reason is None and bucket // 2 >= count
        assert bucket == 4 or bucket // 4 < count
    assert runner._decode_layout.select(capacity + 1, 1) == (None, "live_count")
    assert runner._decode_layout.select(1, 33) == (None, "table_width")
    graph._close_recurrent_graph()


@pytest.mark.parametrize("capacity,hidden,width", [(3, 7, 17), (8, 2048, 32), (16, 64, 9)])
def test_payload_budget_matches_allocations(model, capacity, hidden, width):
    layout = DecodeBucketLayout(capacity, width)
    cache = fixtures.make_cache(model)
    buckets = [allocate_bucket(cache, layout, rows, hidden) for rows in layout.row_counts]
    actual_device = sum(
        t.numel() * t.element_size() for b in buckets for t in b["tensors"].values()
    )
    actual_staging = sum(
        t.numel() * t.element_size() for b in buckets for t in b["metadata"].staging.values()
    )
    assert layout.payload_bytes(hidden) == (actual_device, actual_staging)


@pytest.mark.parametrize("mode", ["persistent", "eager", "graph"])
def test_eight_live_requests_alternate_buckets_without_aliasing_outputs(model, monkeypatch, mode):
    cache, reference = fixtures.make_cache(model), fixtures.make_cache(model)
    if mode != "persistent":
        fixtures.fake_runtime(monkeypatch, cache)
    runner = ModelRunner(model, cache, max_num_seqs=8)
    if mode == "persistent":
        runner._enable_persistent_decode()
    else:
        runner._enable_recurrent_graph(use_graphs=mode == "graph")
    for current in (cache, reference):
        for name in "abcdefgh":
            current.allocate(name, 8)
    positions = dict.fromkeys("abcdefgh", 0)
    retained = []
    for ids, rows in [("abcdefgh", 16), ("a", 4), ("abc", 8), ("abcdefgh", 16), ("ab", 4)]:
        ids = list(ids)
        pos = [positions[name] for name in ids]
        hidden = torch.randn(len(ids), model.config.hidden_size)
        actual = runner._recurrent(hidden, ids, [0] * len(ids), pos)
        padded = reference._pad_prepared(
            reference._prepare_batch(ids, [0] * len(ids), pos),
            row_indices=range(1, 2 * len(ids), 2),
            row_count=rows,
            table_width=32,
        )
        physical = hidden.new_zeros(rows, model.config.hidden_size)
        physical[list(padded.live_rows)] = hidden
        full = model._recurrent_prepared(physical, padded, reference)
        assert all(torch.equal(a, b[list(padded.live_rows)]) for a, b in zip(actual, full))
        assert torch.equal(cache.key_cache, reference.key_cache)
        assert torch.equal(cache.value_cache, reference.value_cache)
        if mode == "persistent":
            assert runner._persistent["last_dispatch"]["row_count"] == rows
        else:
            assert runner._decode_executor.last_dispatch["bucket_id"] == rows
        retained.append((actual, tuple(t.clone() for t in actual)))
        for name in ids:
            positions[name] += 1
    assert all(torch.equal(a, b) for output, saved in retained for a, b in zip(output, saved))
    if mode != "persistent":
        runner._close_recurrent_graph()
