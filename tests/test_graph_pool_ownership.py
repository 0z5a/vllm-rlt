"""CPU allocator-lifetime model; no claim of measured CUDA allocator behavior."""

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
    assert len(runtime.pool_releases) == len(runtime.pool_refs)
    assert all(ready for _, ready in runtime.pool_releases)


def test_pool_rejects_unqualified_allocator(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_allocator_backend", lambda: "cudaMallocAsync")
    with pytest.raises(RuntimeError, match="native CUDA allocator"):
        _CudaRuntime("cuda:0").new_pool()


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
    assert not runtime.pool_releases
    assert len(executor.buckets) == 2 and executor.failure is not None
    assert not any(event[0] == "release_stream" for event in runtime.events)
    runtime.main.failure = None
    monkeypatch.setattr(runtime.graphs[1], "reset", reset)
    executor.close()
    assert executor.status == "closed"
    # The saved primary traceback still holds local graph/bucket references.
    # Explicit owner/output assignments must nevertheless have released pools.
    released(runtime)
