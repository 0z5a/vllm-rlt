"""CPU allocator-lifetime model; no claim of measured CUDA allocator behavior."""

from contextlib import contextmanager

import pytest
import test_recurrent_graph as fixtures
import torch

from vllm_lt.worker import capture_resources
from vllm_lt.worker.model_runner import ModelRunner
from vllm_lt.worker.recurrent_graph import _CudaRuntime

model = fixtures.model
pytestmark = pytest.mark.usefixtures("forbid_cuda")


def test_capture_stream_reused_only_after_release_and_never_shared_by_live_runtimes(monkeypatch):
    monkeypatch.setattr(capture_resources, "_IDLE_STREAMS", {})
    monkeypatch.setattr(torch.cuda, "Stream", lambda **kwargs: object())
    a, b, c = (_CudaRuntime("cuda:0") for _ in range(3))
    first, second = a.new_stream(), b.new_stream()
    assert first is not second
    with pytest.raises(RuntimeError, match="already leased"):
        a.new_stream()
    with pytest.raises(RuntimeError, match="does not belong"):
        a.release_stream(second)
    a.release_stream(first)
    assert c.new_stream() is first
    assert b._leased_stream is second
    other_device = _CudaRuntime("cuda:1")
    assert other_device.new_stream() not in (first, second)
    c.release_stream(first)
    b.release_stream(second)


def enable(model, monkeypatch):
    cache = fixtures.make_cache(model)
    runtime = fixtures.fake_runtime(monkeypatch, cache)
    runner = ModelRunner(model, cache)
    runner._enable_recurrent_graph(use_graphs=True)
    return cache, runtime, runner, runner._decode_executor


def released(runtime):
    assert all(ref() is None for ref in runtime.pool_refs)
    assert runtime.pool_reserved == {}
    assert len(runtime.pool_releases) == len(runtime.pool_refs)
    assert all(ready for _, ready in runtime.pool_releases)
    assert runtime.default_reserved == 123 * 1024**2


def test_public_pool_factory_selects_device_and_rejects_unqualified_allocator(monkeypatch):
    events, owner = [], object()

    @contextmanager
    def device(value):
        events.append(("device", value))
        yield
        events.append(("restore", value))

    def pool():
        assert events == [("device", "cuda:0")]
        events.append(("MemPool",))
        return owner

    monkeypatch.setattr(torch.cuda, "get_allocator_backend", lambda: "native")
    monkeypatch.setattr(torch.cuda, "device", device)
    monkeypatch.setattr(torch.cuda, "MemPool", pool)
    monkeypatch.setattr(
        torch.cuda, "empty_cache", lambda: pytest.fail("shared cache release is forbidden")
    )
    runtime = _CudaRuntime("cuda:0")
    assert runtime.new_pool() is owner
    assert events == [("device", "cuda:0"), ("MemPool",), ("restore", "cuda:0")]
    events.clear()
    monkeypatch.setattr(torch.cuda, "get_allocator_backend", lambda: "cudaMallocAsync")
    with pytest.raises(RuntimeError, match="native CUDA allocator"):
        runtime.new_pool()
    assert events == []


def test_shared_owner_releases_after_all_resets_and_captured_outputs(model, monkeypatch):
    _, runtime, runner, executor = enable(model, monkeypatch)
    snapshot = executor.snapshot()
    pool_ids = []
    for key, bucket in snapshot["buckets"].items():
        owner = bucket["pool_owner"]
        assert owner == {
            "kind": "torch.cuda.MemPool",
            "id": bucket["pool_id"],
            "release_policy": "synchronize-reset-drop-captured-outputs-owner-last",
        }
        assert list(executor.buckets[int(key)]["graph"].pool()) == owner["id"]
        pool_ids.append(tuple(owner["id"]))
    assert len(set(pool_ids)) == 1
    assert all(ref() is not None for ref in runtime.pool_refs)
    assert all(ref() is not None for graph in runtime.graphs for ref in graph.output_refs)
    before_close = len(runtime.events)
    runner._close_recurrent_graph()
    events = runtime.events[before_close:]
    assert events[:2] == [("synchronize", "setup"), ("synchronize", "main")]
    resets = [i for i, event in enumerate(events) if event[0] == "reset"]
    frees = [i for i, event in enumerate(events) if event[0] == "pool_release"]
    assert len(resets) == 2 and len(frees) == 1 and max(resets) < min(frees)
    assert events[-1] == ("release_stream", "setup")
    released(runtime)
    assert executor.snapshot()["buckets"] == {}


@pytest.mark.parametrize("failure_site", ["synchronize", "second_reset"])
def test_failed_close_retains_shared_owner_and_outputs_until_confirmed_close(
    model, monkeypatch, failure_site
):
    _, runtime, _, executor = enable(model, monkeypatch)
    pointer = executor.buckets[4]["tensors"]["hidden_in"].data_ptr()
    primary = RuntimeError("close completion/reset failed")
    reset = runtime.graphs[1].reset

    def fail():
        raise primary

    if failure_site == "synchronize":
        runtime.main.failure = primary
    else:
        monkeypatch.setattr(runtime.graphs[1], "reset", fail)
    with pytest.raises(RuntimeError) as caught:
        executor.close()
    assert caught.value is primary
    assert executor.status == "failed"
    assert executor.buckets[4]["tensors"]["hidden_in"].data_ptr() == pointer
    assert executor.failure["primary"]["message"] == str(primary)
    assert all(ref() is not None for ref in runtime.pool_refs)
    assert all(ref() is not None for graph in runtime.graphs for ref in graph.output_refs)
    assert len(runtime.pool_reserved) == 1 and not runtime.pool_releases
    assert len(executor.buckets) == 2 and executor.failure is not None
    assert not any(event[0] == "release_stream" for event in runtime.events)
    runtime.main.failure = None
    monkeypatch.setattr(runtime.graphs[1], "reset", reset)
    executor.close()
    assert executor.status == "closed"
    # The saved primary traceback still holds local graph/bucket references.
    # Explicit owner/output assignments must nevertheless have released pools.
    released(runtime)


@pytest.mark.parametrize("failed_capture", [1, 2])
def test_capture_failure_retains_pool_ownership_then_safe_close_releases(
    model, monkeypatch, failed_capture
):
    cache = fixtures.make_cache(model)
    original = cache.key_cache.clone(), cache.value_cache.clone(), tuple(cache._free_blocks)
    runtime = fixtures.fake_runtime(monkeypatch, cache)
    create = runtime.new_graph
    primary = RuntimeError("capture end failed")

    def graph():
        value = create()
        if value.index == failed_capture:
            value.end_failure = primary
        return value

    monkeypatch.setattr(runtime, "new_graph", graph)
    runner = ModelRunner(model, cache)
    with pytest.raises(RuntimeError) as caught:
        runner._enable_recurrent_graph(use_graphs=True)
    assert caught.value is primary
    executor = runner._decode_executor
    assert executor.failure["completion_confirmed"]
    assert executor.setup_record["scratch"]["restored"]
    assert torch.equal(cache.key_cache, original[0])
    assert torch.equal(cache.value_cache, original[1])
    assert tuple(cache._free_blocks) == original[2]
    assert len(runtime.pool_refs) == 1
    assert all(ref() is not None for ref in runtime.pool_refs)
    runner._close_recurrent_graph()
    released(runtime)


def test_graph_creation_failure_releases_owner_without_a_completed_graph(model, monkeypatch):
    cache = fixtures.make_cache(model)
    runtime = fixtures.fake_runtime(monkeypatch, cache)

    def fail():
        raise RuntimeError("graph creation failed")

    monkeypatch.setattr(runtime, "new_graph", fail)
    runner = ModelRunner(model, cache)
    with pytest.raises(RuntimeError, match="graph creation failed"):
        runner._enable_recurrent_graph(use_graphs=True)
    assert len(runtime.pool_refs) == 1 and runtime.pool_refs[0]() is not None
    assert runtime.graphs == []
    runner._close_recurrent_graph()
    released(runtime)


def test_closed_executor_can_be_replaced_without_retaining_its_pool(model, monkeypatch):
    cache = fixtures.make_cache(model)
    runtime = fixtures.fake_runtime(monkeypatch, cache)
    for repetition in range(2):
        runner = ModelRunner(model, cache)
        runner._enable_recurrent_graph(use_graphs=True)
        assert sum(runtime.pool_reserved.values()) == 19 * 1024**2
        assert runner._graph_snapshot()["setup"]["captures"] == 2
        runner._close_recurrent_graph()
        released(runtime)
        assert len(runtime.pool_releases) == repetition + 1
    assert len(runtime.graphs) == 4 and all(g.reset_done for g in runtime.graphs)
    assert runtime.events.count(("reset_peaks",)) == 2


@pytest.mark.parametrize("backend", ["triton", "torch"])
def test_eager_control_has_null_owner_and_allocates_no_graph_pool(model, monkeypatch, backend):
    cache = fixtures.make_cache(model)
    runtime = fixtures.fake_runtime(monkeypatch, cache)
    cache.backend = backend
    runner = ModelRunner(model, cache)
    runner._enable_recurrent_graph(use_graphs=False)
    assert all(
        bucket["pool_owner"] is None for bucket in runner._graph_snapshot()["buckets"].values()
    )
    assert runtime.pool_refs == [] and runtime.graphs == []
    runner._close_recurrent_graph()
    released(runtime)
