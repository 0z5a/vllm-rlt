"""CPU transaction/lifetime tests; fake capture does not qualify CUDA replay."""

import gc
import sys
import weakref
from contextlib import contextmanager
from dataclasses import asdict
from types import SimpleNamespace

import pytest
import torch

import vllm_lt.core.kv_cache_manager as kv_module
import vllm_lt.worker.recurrent_graph as graph_module
from vllm_lt.config import CacheConfig
from vllm_lt.core.kv_cache_manager import KVCacheManager
from vllm_lt.core.scheduler import ScheduledItem, SchedulerOutput
from vllm_lt.engine.llm_engine import LLMEngine
from vllm_lt.kernels.paged_attention import torch_paged_attention
from vllm_lt.models import OuroConfig, OuroForCausalLM
from vllm_lt.request import Request, Stage
from vllm_lt.sampling_params import SamplingParams
from vllm_lt.worker.model_runner import ModelRunner
from vllm_lt.worker.recurrent_graph import GraphLimits, RecurrentGraphExecutor


@pytest.fixture(autouse=True)
def no_cuda(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("CPU graph tests must not discover or initialize CUDA")

    for name in (
        "is_available",
        "device_count",
        "current_device",
        "init",
        "_lazy_init",
        "synchronize",
        "current_stream",
        "Stream",
        "CUDAGraph",
        "memory_allocated",
        "memory_reserved",
    ):
        monkeypatch.setattr(torch.cuda, name, forbidden)


@pytest.fixture
def model():
    torch.manual_seed(83)
    return OuroForCausalLM(OuroConfig.tiny())


def make_cache(model):
    c = model.config
    cache = KVCacheManager(c.num_hidden_layers, c.num_key_value_heads, c.head_dim, 160, 16, 4)
    cache.key_cache.fill_(71)
    cache.value_cache.fill_(-83)
    return cache


def prefixes(cache):
    return {
        name: [[(p.prefix, sorted(p.pending)) for p in depth] for depth in alloc.written]
        for name, alloc in cache._allocations.items()
    }


class FakeStream:
    def __init__(self, runtime, name):
        self.runtime, self.name = runtime, name
        self.failure = None

    def wait_stream(self, other):
        self.runtime.events.append(("wait", self.name, other.name))

    def synchronize(self):
        self.runtime.events.append(("synchronize", self.name))
        if self.failure is not None:
            raise self.failure


class FakeGraph:
    def __init__(self, runtime, index):
        self.runtime, self.index = runtime, index
        self.body = None
        self.reset_done = False
        self.replay_failure = self.end_failure = None

    def capture_begin(self, *, pool, capture_error_mode):
        assert pool is None and capture_error_mode == "global"
        assert self.runtime.current is self.runtime.side
        self.runtime.capture = self
        self.runtime.events.append(("capture_begin", self.index))

    def capture_end(self):
        self.runtime.capture = None
        self.runtime.events.append(("capture_end", self.index))
        if self.end_failure is not None:
            raise self.end_failure

    def replay(self):
        self.runtime.events.append(("replay", self.index))
        if self.replay_failure is not None:
            raise self.replay_failure
        assert self.body is not None
        self.body()

    def pool(self):
        return (0, self.index)

    def raw_cuda_graph_exec(self):
        return self.index + 100

    def reset(self):
        self.runtime.events.append(("reset", self.index))
        self.reset_done = True
        self.body = None


class FakeRuntime:
    def __init__(self):
        self.events, self.graphs = [], []
        self.main, self.side = FakeStream(self, "main"), FakeStream(self, "setup")
        self.current, self.capture = self.main, None
        self.after_memory = None
        self.memory_calls = 0

    def current_stream(self):
        return self.current

    def new_stream(self):
        return self.side

    @contextmanager
    def stream_context(self, stream):
        old, self.current = self.current, stream
        try:
            yield
        finally:
            self.current = old

    def new_graph(self):
        graph = FakeGraph(self, len(self.graphs) + 1)
        self.graphs.append(graph)
        return graph

    def reset_peaks(self):
        self.events.append(("reset_peaks",))

    def memory(self):
        self.memory_calls += 1
        if self.memory_calls > 1 and self.after_memory is not None:
            return self.after_memory
        return dict.fromkeys(
            ("allocated_bytes", "reserved_bytes", "peak_allocated_bytes", "peak_reserved_bytes"), 0
        )


def fake_runtime(monkeypatch, cache):
    """Replace raw kernels and stream/graph plumbing, retaining real tensor/body/cache code."""

    def scatter(keys, values, blocks, offsets, k, v, active):
        live = active.nonzero().flatten()
        keys[blocks[live], offsets[live]] = k[live]
        values[blocks[live], offsets[live]] = v[live]

    monkeypatch.setitem(
        sys.modules, "vllm_lt.kernels.triton_kv_write", SimpleNamespace(masked_kv_write=scatter)
    )
    monkeypatch.setattr(kv_module, "triton_paged_attention", torch_paged_attention)
    monkeypatch.setattr(cache, "backend", "triton")
    runtime = FakeRuntime()
    monkeypatch.setattr(graph_module, "_make_runtime", lambda device: runtime)
    original = RecurrentGraphExecutor._tensor_body

    def body(executor, bucket):
        if runtime.capture is not None:
            runtime.capture.body = lambda: original(executor, bucket)
        return original(executor, bucket)

    monkeypatch.setattr(RecurrentGraphExecutor, "_tensor_body", body)
    return runtime


@pytest.mark.parametrize("use_graphs", [False, True])
def test_torch_backend_is_compact_and_has_no_cuda_setup(model, use_graphs):
    cache, reference = make_cache(model), make_cache(model)
    runner = ModelRunner(model, cache)
    runner._enable_recurrent_graph(use_graphs=use_graphs)
    setup = runner._graph_snapshot()
    assert setup["setup"]["skip_reason"] == "backend"
    assert setup["device_payload_bytes"] == 96 * model.config.hidden_size + 1980
    assert setup["cpu_staging_bytes"] == 1884
    assert all(b["generation"] == 0 and b["graph_id"] is None for b in setup["buckets"].values())
    for c in (cache, reference):
        c.allocate("a", 3)
    for pos in range(3):
        hidden = model.prelude(torch.tensor([pos + 2]))
        actual = runner._recurrent(hidden, ["a"], [0], [pos])
        expected = model.recurrent(hidden, ["a"], [0], [pos], reference)
        assert all(torch.equal(a, b) for a, b in zip(actual, expected))
        assert torch.equal(cache.key_cache, reference.key_cache)
        assert torch.equal(cache.value_cache, reference.value_cache)
        assert prefixes(cache) == prefixes(reference)
    snapshot = runner._graph_snapshot()
    assert snapshot["fallback_counts"] == {"backend": 3, "live_count": 0, "table_width": 0}
    assert snapshot["counters"]["prepared"] == snapshot["counters"]["replays"] == 0
    assert snapshot["counters"]["completed"] == 3
    runner._close_recurrent_graph()
    assert runner._graph_snapshot()["status"] == "closed"
    assert runner._graph_snapshot()["buckets"] == {}
    with pytest.raises(RuntimeError, match="closed"):
        runner._decode_executor.require_usable()


def test_limits_installation_and_admission_guards(model):
    for invalid in ({}, {**asdict(GraphLimits()), "setup_timeout_s": True}):
        runner = ModelRunner(model, make_cache(model))
        with pytest.raises(ValueError):
            runner._enable_recurrent_graph(use_graphs=False, limits=invalid)
        assert runner._graph_snapshot() == {"enabled": False}
    for first in ("persistent", "graph"):
        runner = ModelRunner(model, make_cache(model))
        if first == "persistent":
            runner._enable_persistent_decode()
        else:
            runner._enable_recurrent_graph(use_graphs=False)
        for enable in (
            runner._enable_persistent_decode,
            lambda: runner._enable_recurrent_graph(use_graphs=True),
        ):
            with pytest.raises(RuntimeError, match="replacement"):
                enable()
    engine = LLMEngine(model, cache_config=CacheConfig(num_blocks=160))
    engine.add_request("queued", [3])
    with pytest.raises(RuntimeError, match="admission"):
        engine._enable_recurrent_graph(use_graphs=False)
    cache = make_cache(model)
    cache.allocate("admitted", 2)
    with pytest.raises(RuntimeError, match="admission"):
        ModelRunner(model, cache)._enable_recurrent_graph(use_graphs=False)


def test_fake_capture_setup_restores_exact_pool_and_has_independent_graphs(model, monkeypatch):
    cache = make_cache(model)
    # Deliberately use a noncanonical allocator order, so ordinary free is insufficient.
    cache._free_blocks[:] = cache._free_blocks[11:] + cache._free_blocks[:11]
    original = cache.key_cache.clone(), cache.value_cache.clone(), tuple(cache._free_blocks)
    runtime = fake_runtime(monkeypatch, cache)
    runner = ModelRunner(model, cache)
    runner._enable_recurrent_graph(use_graphs=True)
    snapshot = runner._graph_snapshot()
    setup = snapshot["setup"]
    assert setup["status"] == "complete" and snapshot["status"] == "ready"
    assert (setup["warmups"], setup["captures"], setup["verification_replays"]) == (6, 2, 2)
    assert setup["scratch"]["saved_cpu_bytes"] == 4 * cache.bytes_per_block
    assert setup["scratch"]["restored"] and not cache._allocations
    assert setup["scratch"]["before_hashes"] == setup["scratch"]["after_hashes"]
    assert torch.equal(cache.key_cache, original[0]) and torch.equal(cache.value_cache, original[1])
    assert tuple(cache._free_blocks) == original[2]
    assert runtime.current is runtime.main
    assert runtime.events.count(("capture_begin", 1)) == 1
    assert runtime.events.count(("capture_begin", 2)) == 1
    assert snapshot["buckets"]["4"]["pool_id"] != snapshot["buckets"]["8"]["pool_id"]
    assert all(b["generation"] == b["setup_generation"] == 1 for b in snapshot["buckets"].values())
    assert all(v == 0 for v in snapshot["counters"].values())
    assert all(b["verification"]["inactive_positive_zero"] for b in snapshot["buckets"].values())
    snapshot["buckets"]["4"]["generation"] = 99
    assert runner._graph_snapshot()["buckets"]["4"]["generation"] == 1
    runner._close_recurrent_graph()
    assert all(g.reset_done for g in runtime.graphs)


@pytest.mark.parametrize("use_graphs", [False, True])
def test_bucket_alternation_commits_only_after_completion_and_owns_publication(
    model, monkeypatch, use_graphs
):
    cache, reference = make_cache(model), make_cache(model)
    fake_runtime(monkeypatch, cache)
    runner = ModelRunner(model, cache)
    runner._enable_recurrent_graph(use_graphs=use_graphs)
    executor = runner._decode_executor
    for c in (cache, reference):
        for name in "abcd":
            c.allocate(name, 3)
    owned = runner._graph_snapshot()["buckets"]
    retained = []
    for ids, positions, rows in [
        (list("abcd"), [0] * 4, 8),
        (["a"], [1], 4),
        (list("bcd"), [1] * 3, 8),
        (["a"], [2], 4),
    ]:
        hidden = torch.randn(len(ids), model.config.hidden_size)
        before = prefixes(cache)
        actual = executor.recurrent(hidden, ids, [0] * len(ids), positions, defer_completion=True)
        assert prefixes(cache) == before and executor.ticket.state == "bound"
        assert executor.buckets[rows]["metadata"].in_use
        padded = reference._pad_prepared(
            reference._prepare_batch(ids, [0] * len(ids), positions),
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
        executor.complete_after_gate()
        assert prefixes(cache) == prefixes(reference)
        assert executor.last_dispatch["ticket_state"] == "committed"
        assert not executor.buckets[rows]["metadata"].in_use
        for value in actual:
            assert all(
                value.untyped_storage().data_ptr() != t.untyped_storage().data_ptr()
                for b in executor.buckets.values()
                for t in b["tensors"].values()
            )
        retained.append((actual, tuple(v.clone() for v in actual)))
    for actual, copied in retained:
        assert all(torch.equal(a, b) for a, b in zip(actual, copied))
    now = executor.snapshot()
    assert all(now["buckets"][key]["tensors"] == owned[key]["tensors"] for key in owned)
    assert now["counters"]["prepared"] == now["counters"]["committed"] == 4
    assert now["counters"]["replays" if use_graphs else "eager"] == 4
    executor.close()


def test_empty_invalid_prefix_and_shape_fallback_do_not_use_bucket(model, monkeypatch):
    cache = make_cache(model)
    fake_runtime(monkeypatch, cache)
    runner = ModelRunner(model, cache)
    runner._enable_recurrent_graph(use_graphs=False)
    executor = runner._decode_executor
    initial = executor.snapshot()["buckets"]
    out = runner._recurrent(torch.empty(0, model.config.hidden_size), [], [], [])
    assert out[0].shape == (0, model.config.hidden_size) and out[1].shape == (0,)
    assert executor.counters["empty"] == 1
    assert executor.last_dispatch["row_count"] == executor.last_dispatch["table_width"] == 0
    assert executor.snapshot()["buckets"] == initial
    cache.allocate("long", 513)
    with pytest.raises(RuntimeError, match="uninitialized"):
        runner._recurrent(torch.zeros(1, model.config.hidden_size), ["long"], [0], [1])
    assert executor.status == "ready" and executor.snapshot()["buckets"] == initial
    for tracker in cache._allocations["long"].written[0]:
        tracker.prefix = 512
    # Seeded storage is finite; this checks routing and prefix completion, not numerical fidelity.
    runner._recurrent(torch.randn(1, model.config.hidden_size), ["long"], [0], [512])
    assert executor.last_dispatch["bucket_id"] is None
    assert executor.fallback_counts["table_width"] == 1
    cache.free("long")
    for name in "abcde":
        cache.allocate(name, 2)
    runner._recurrent(torch.randn(5, model.config.hidden_size), list("abcde"), [0] * 5, [0] * 5)
    assert executor.fallback_counts["live_count"] == 1
    assert executor.counters["prepared"] == 0
    assert executor.snapshot()["buckets"] == initial
    executor.close()


def test_ordinary_dispatch_does_not_collect_rich_tensor_descriptors(model, monkeypatch):
    import vllm_lt.worker.recurrent_graph as implementation

    cache = make_cache(model)
    fake_runtime(monkeypatch, cache)
    runner = ModelRunner(model, cache)
    runner._enable_recurrent_graph(use_graphs=False)
    cache.allocate("live", 2)
    executor = runner._decode_executor
    assert executor.record_dispatch_tensors is False
    with monkeypatch.context() as patch:
        patch.setattr(
            implementation,
            "_description",
            lambda *args: pytest.fail("ordinary dispatch collected rich tensor diagnostics"),
        )
        outputs = runner._recurrent(torch.ones(1, model.config.hidden_size), ["live"], [0], [0])
    assert outputs[0].shape == (1, model.config.hidden_size)
    assert executor.counters["completed"] == 1
    assert executor.last_dispatch["actual_inputs"] is None
    assert executor.last_dispatch["actual_physical_outputs"] is None
    assert executor.last_publication is None
    executor.record_dispatch_tensors = True
    runner._recurrent(torch.ones(1, model.config.hidden_size), ["live"], [0], [1])
    assert executor.last_dispatch["actual_inputs"] is not None
    assert executor.last_publication is not None
    executor.close()


def test_request_publication_waits_for_gate_readback_and_commit(model, monkeypatch):
    cache = make_cache(model)
    fake_runtime(monkeypatch, cache)
    runner = ModelRunner(model, cache)
    runner._enable_recurrent_graph(use_graphs=False)
    cache.allocate("a", 2)
    request = Request(
        "a",
        [3],
        SamplingParams(max_tokens=2, ignore_eos=True),
        stage=Stage.RECURRENT,
        hidden_state=model.prelude(torch.tensor([3]))[0],
    )
    initial = request.hidden_state
    events = []
    tolist = torch.Tensor.tolist
    commit = cache._commit_decode_traversal

    def readback(value):
        if value.dtype == torch.float32 and value.shape == (1,):
            events.append("readback")
        return tolist(value)

    def publish(ticket, *, completion_confirmed):
        assert request.hidden_state is initial
        assert events == ["readback"]
        events.append("commit")
        return commit(ticket, completion_confirmed=completion_confirmed)

    monkeypatch.setattr(torch.Tensor, "tolist", readback)
    monkeypatch.setattr(cache, "_commit_decode_traversal", publish)
    gates = runner.execute(SchedulerOutput(Stage.RECURRENT, [ScheduledItem(request)]))
    assert len(gates) == 1 and events == ["readback", "commit"]
    assert request.hidden_state is not initial and runner._decode_executor.ticket is None
    assert all(p.prefix == 1 for p in cache._allocations["a"].written[0])
    runner._close_recurrent_graph()


@pytest.mark.parametrize("confirmed", [False, True])
def test_partial_replay_failure_retains_primary_and_resources_until_safe_close(
    model, monkeypatch, confirmed
):
    cache = make_cache(model)
    runtime = fake_runtime(monkeypatch, cache)
    runner = ModelRunner(model, cache)
    runner._enable_recurrent_graph(use_graphs=True)
    executor = runner._decode_executor
    cache.allocate("a", 2)
    before = prefixes(cache)
    primary = RuntimeError("replay primary")
    runtime.graphs[0].replay_failure = primary
    if not confirmed:
        runtime.main.failure = RuntimeError("stream secondary")
    graph_ref = weakref.ref(runtime.graphs[0])
    buffer_ref = weakref.ref(executor.buckets[4]["tensors"]["hidden_in"])
    with pytest.raises(RuntimeError) as raised:
        runner._recurrent(torch.ones(1, model.config.hidden_size), ["a"], [0], [0])
    assert raised.value is primary and prefixes(cache) == before
    assert executor.failure["completion_confirmed"] is confirmed
    assert executor.buckets[4]["metadata"].in_use is not confirmed
    assert graph_ref() is not None and buffer_ref() is not None
    with pytest.raises(RuntimeError):
        runner._recurrent(torch.ones(1, model.config.hidden_size), ["a"], [0], [0])
    if not confirmed:
        with pytest.raises(RuntimeError, match="stream secondary"):
            executor.close()
        assert buffer_ref() is not None and not runtime.graphs[0].reset_done
        runtime.main.failure = None
    executor.close()
    assert (
        executor.status == "closed" and executor.failure["primary"]["message"] == "replay primary"
    )
    assert executor.failure["completion_confirmed"]
    # An externally retained exception traceback legitimately keeps the call's
    # local bucket alive; remove that test-owned reference before checking close.
    primary.__traceback__ = None
    del raised
    gc.collect()
    assert buffer_ref() is None


def test_failed_capture_restores_stream_and_scratch_without_masking_primary(model, monkeypatch):
    cache = make_cache(model)
    runtime = fake_runtime(monkeypatch, cache)
    original = cache.key_cache.clone(), cache.value_cache.clone(), tuple(cache._free_blocks)
    create = runtime.new_graph
    primary = ValueError("capture primary")

    def graph():
        value = create()
        value.end_failure = RuntimeError("capture end secondary")
        return value

    monkeypatch.setattr(runtime, "new_graph", graph)
    original_body = RecurrentGraphExecutor._tensor_body

    def body(executor, bucket):
        if runtime.capture is not None:
            raise primary
        return original_body(executor, bucket)

    monkeypatch.setattr(RecurrentGraphExecutor, "_tensor_body", body)
    runner = ModelRunner(model, cache)
    with pytest.raises(ValueError) as raised:
        runner._enable_recurrent_graph(use_graphs=True)
    assert raised.value is primary and runtime.current is runtime.main
    executor = runner._decode_executor
    assert executor.status == "failed" and executor.setup_record["scratch"]["restored"]
    assert executor.setup_record["capture_cleanup_errors"][0]["message"] == "capture end secondary"
    assert not cache._allocations and tuple(cache._free_blocks) == original[2]
    assert torch.equal(cache.key_cache, original[0]) and torch.equal(cache.value_cache, original[1])
    executor.close()


@pytest.mark.parametrize(
    "limit_field, measured",
    [
        ("graph_retained_allocated_bytes", "allocated_bytes"),
        ("graph_retained_reserved_bytes", "reserved_bytes"),
        ("setup_peak_allocated_bytes", "peak_allocated_bytes"),
        ("setup_peak_reserved_bytes", "peak_reserved_bytes"),
    ],
)
def test_setup_memory_caps_decline_after_restoration(model, monkeypatch, limit_field, measured):
    cache = make_cache(model)
    runtime = fake_runtime(monkeypatch, cache)
    runtime.after_memory = {**runtime.memory(), measured: 11}
    runtime.memory_calls = 0
    limits = {**asdict(GraphLimits()), limit_field: 10}
    runner = ModelRunner(model, cache)
    runner._enable_recurrent_graph(use_graphs=True, limits=limits)
    decline = runner._graph_snapshot()["budget_decline"]
    assert decline["attempt_setup"]["scratch"]["restored"] and not cache._allocations
    assert decline["attempt_setup"]["status"] == "failed"
    assert decline["error"]["limit_name"] == limit_field
    assert decline["error"]["observed"] == 11 and decline["error"]["limit"] == 10
    assert runner._decode_executor is None and all(g.reset_done for g in runtime.graphs)


def test_setup_last_timestamp_deadline_cannot_report_success(model, monkeypatch):
    clock = iter([0, 0, 0, 60 * 10**9, 60 * 10**9 + 1])
    monkeypatch.setattr(graph_module.time, "perf_counter_ns", lambda: next(clock))
    runner = ModelRunner(model, make_cache(model))
    runner._enable_recurrent_graph(use_graphs=False)
    decline = runner._graph_snapshot()["budget_decline"]
    assert decline["attempt_setup"]["status"] == "failed"
    assert decline["error"]["limit_name"] == "setup_timeout_s"
    assert decline["error"]["observed"] >= decline["error"]["limit"]
    runner._close_recurrent_graph()


@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt, SystemExit])
@pytest.mark.parametrize("confirmed", [False, True])
def test_engine_update_failure_preserves_commit_and_only_frees_after_completion(
    model, monkeypatch, error_type, confirmed
):
    engine = LLMEngine(model, cache_config=CacheConfig(num_blocks=160))
    cache, runner = engine.cache_manager, engine.model_runner
    runtime = fake_runtime(monkeypatch, cache)
    engine._enable_recurrent_graph(use_graphs=True)
    engine.add_request("a", [3], SamplingParams(max_tokens=3, ignore_eos=True))
    for _ in range(10):
        engine.step()
        if engine.last_schedule.stage == Stage.PRELUDE:
            break
    else:
        pytest.fail("tiny engine never reached decode")
    allocation = cache._allocations["a"]
    assert all(p.prefix == 1 for p in allocation.written[0])
    primary = error_type("update primary")

    def fail_update(batch, result):
        assert batch.stage == Stage.RECURRENT and len(result) == 1
        assert runner._decode_executor.ticket is None
        assert runner._graph_snapshot()["last_dispatch"]["ticket_state"] == "committed"
        assert all(p.prefix == 2 for p in allocation.written[0])
        if not confirmed:
            runtime.main.failure = RuntimeError("update completion secondary")
        raise primary

    monkeypatch.setattr(engine, "_update", fail_update)
    with pytest.raises(error_type) as raised:
        engine.step()
    assert raised.value is primary
    # Successful traversal prefixes are never rolled back by a later route failure.
    assert all(p.prefix == 2 for p in allocation.written[0])
    assert ("a" in cache._allocations) is not confirmed
    assert runner._graph_snapshot()["failure"]["completion_confirmed"] is confirmed
    with pytest.raises(RuntimeError):
        engine.step()
    if not confirmed:
        with pytest.raises(RuntimeError, match="quarantined"):
            engine.abort_request("a")
        runtime.main.failure = None
    runner._close_recurrent_graph()


def test_changed_bucket_storage_is_rejected_before_write_and_cannot_retry(model, monkeypatch):
    cache = make_cache(model)
    fake_runtime(monkeypatch, cache)
    runner = ModelRunner(model, cache)
    runner._enable_recurrent_graph(use_graphs=True)
    cache.allocate("a", 2)
    initial = cache.key_cache.clone(), cache.value_cache.clone(), prefixes(cache)
    executor = runner._decode_executor
    executor.buckets[4]["tensors"]["hidden_in"] = torch.zeros(4, model.config.hidden_size)
    with pytest.raises(RuntimeError, match="signature"):
        runner._recurrent(torch.ones(1, model.config.hidden_size), ["a"], [0], [0])
    assert torch.equal(cache.key_cache, initial[0]) and torch.equal(cache.value_cache, initial[1])
    assert prefixes(cache) == initial[2] and executor.counters["prepared"] == 0
    assert executor.status == "failed"
    executor.close()


def test_close_failure_retains_buffers_and_preserves_first_failure(model, monkeypatch):
    cache = make_cache(model)
    runtime = fake_runtime(monkeypatch, cache)
    runner = ModelRunner(model, cache)
    runner._enable_recurrent_graph(use_graphs=True)
    executor = runner._decode_executor
    pointer = executor.buckets[4]["tensors"]["hidden_in"].data_ptr()
    primary = RuntimeError("reset primary")
    reset = runtime.graphs[0].reset

    def fail_reset():
        raise primary

    monkeypatch.setattr(runtime.graphs[0], "reset", fail_reset)
    with pytest.raises(RuntimeError) as raised:
        executor.close()
    assert raised.value is primary and executor.status == "failed"
    assert executor.buckets[4]["tensors"]["hidden_in"].data_ptr() == pointer
    assert executor.failure["primary"]["message"] == "reset primary"
    monkeypatch.setattr(runtime.graphs[0], "reset", reset)
    executor.close()
    assert executor.status == "closed"
