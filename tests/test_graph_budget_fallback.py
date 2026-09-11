"""Only configured budget declines may safely select ordinary compact decode."""

from dataclasses import asdict

import pytest
import test_recurrent_graph as fixtures
import torch

from vllm_lt.config import CacheConfig
from vllm_lt.engine.llm_engine import LLMEngine
from vllm_lt.sampling_params import SamplingParams
from vllm_lt.worker import recurrent_graph as graph
from vllm_lt.worker.model_runner import ModelRunner

model = fixtures.model
no_cuda = fixtures.no_cuda


def low_limits(field):
    return {**asdict(graph.GraphLimits()), field: 1}


def compact_matches(model, runner, reference):
    for cache in (runner.cache_manager, reference):
        assert cache.allocate("request", 3)
    for position in range(3):
        hidden = model.prelude(torch.tensor([position + 3]))
        actual = runner._recurrent(hidden, ["request"], [0], [position])
        expected = model.recurrent(hidden, ["request"], [0], [position], reference)
        assert all(torch.equal(a, b) for a, b in zip(actual, expected))
    assert torch.equal(runner.cache_manager.key_cache, reference.key_cache)
    assert torch.equal(runner.cache_manager.value_cache, reference.value_cache)
    assert fixtures.prefixes(runner.cache_manager) == fixtures.prefixes(reference)
    assert runner._graph_snapshot()["budget_decline"]["calls"] == 3


@pytest.mark.parametrize("field", ["common_payload_bytes", "cpu_staging_bytes"])
@pytest.mark.parametrize("use_graphs", [False, True])
def test_preallocation_decline_allocates_no_executor_and_runs_actual_compact(
    model, monkeypatch, field, use_graphs
):
    cache, reference = fixtures.make_cache(model), fixtures.make_cache(model)
    runner = ModelRunner(model, cache)
    monkeypatch.setattr(
        cache, "_allocate_metadata_storage", lambda **kw: pytest.fail("allocated declined bundle")
    )
    monkeypatch.setattr(
        graph, "_make_runtime", lambda *a: pytest.fail("initialized declined runtime")
    )
    runner._enable_recurrent_graph(use_graphs=use_graphs, limits=low_limits(field))
    snapshot = runner._graph_snapshot()
    decline = snapshot["budget_decline"]
    assert snapshot["enabled"] is False and snapshot["status"] == "budget_fallback"
    assert decline["attempted_use_graphs"] is use_graphs
    assert decline["attempt_setup"] is decline["attempt_failure"] is None
    assert decline["error"]["limit_name"] == field and decline["completion_confirmed"]
    assert runner._decode_executor is None
    compact_matches(model, runner, reference)
    snapshot["budget_decline"]["calls"] = 999
    assert runner._graph_snapshot()["budget_decline"]["calls"] == 3
    for enable in (
        lambda: runner._enable_recurrent_graph(use_graphs=True),
        runner._enable_persistent_decode,
    ):
        with pytest.raises(RuntimeError, match="replacement"):
            enable()
    runner._close_recurrent_graph()
    assert runner._graph_snapshot()["status"] == "closed"
    with pytest.raises(RuntimeError, match="closed"):
        runner._recurrent(torch.zeros(1, model.config.hidden_size), ["request"], [0], [0])


def test_retained_cap_restores_exact_scratch_order_and_releases_graphs_before_compact(
    model, monkeypatch
):
    cache, reference = fixtures.make_cache(model), fixtures.make_cache(model)
    cache._free_blocks[:] = cache._free_blocks[13:] + cache._free_blocks[:13]
    reference._free_blocks[:] = cache._free_blocks
    before = cache.key_cache.clone(), cache.value_cache.clone(), tuple(cache._free_blocks)
    runtime = fixtures.fake_runtime(monkeypatch, cache)
    runtime.after_memory = {
        "allocated_bytes": 2,
        "reserved_bytes": 2,
        "peak_allocated_bytes": 2,
        "peak_reserved_bytes": 2,
    }
    runner = ModelRunner(model, cache)
    runner._enable_recurrent_graph(
        use_graphs=True, limits=low_limits("graph_retained_allocated_bytes")
    )
    decline = runner._graph_snapshot()["budget_decline"]
    assert decline["attempt_setup"]["captures"] == 2
    assert decline["attempt_setup"]["scratch"]["restored"]
    assert decline["attempt_failure"]["completion_confirmed"]
    assert decline["attempt_failure"]["secondary"] == []
    assert all(g.reset_done and g.body is None for g in runtime.graphs)
    assert not cache._allocations and tuple(cache._free_blocks) == before[2]
    assert torch.equal(cache.key_cache, before[0]) and torch.equal(cache.value_cache, before[1])
    assert runner._decode_executor is None
    compact_matches(model, runner, reference)


@pytest.mark.parametrize(
    "primary",
    [
        MemoryError("allocator failure"),
        TimeoutError("device timeout"),
        RuntimeError("capture failure"),
        ValueError("incorrect output"),
        torch.OutOfMemoryError("CUDA out of memory"),
    ],
)
def test_device_or_correctness_error_is_never_classified_as_budget_decline(
    model, monkeypatch, primary
):
    cache = fixtures.make_cache(model)
    runtime = fixtures.fake_runtime(monkeypatch, cache)
    create = runtime.new_graph

    def failed_graph():
        value = create()

        def fail(**kwargs):
            raise primary

        value.capture_begin = fail
        return value

    monkeypatch.setattr(runtime, "new_graph", failed_graph)
    runner = ModelRunner(model, cache)
    with pytest.raises(type(primary)) as raised:
        runner._enable_recurrent_graph(use_graphs=True)
    assert raised.value is primary
    assert runner._graph_declined is None and runner._decode_executor is not None
    assert runner._decode_executor.status == "failed"
    with pytest.raises(RuntimeError, match="failed"):
        runner._recurrent(torch.zeros(1, model.config.hidden_size), ["new"], [0], [0])
    runner._close_recurrent_graph()


@pytest.mark.parametrize("secondary", ["synchronize", "restore", "close", "prior_secondary"])
def test_budget_decline_refused_after_uncertain_or_failed_cleanup(model, monkeypatch, secondary):
    cache = fixtures.make_cache(model)
    runtime = fixtures.fake_runtime(monkeypatch, cache)
    runtime.after_memory = {
        "allocated_bytes": 2,
        "reserved_bytes": 2,
        "peak_allocated_bytes": 2,
        "peak_reserved_bytes": 2,
    }
    primary = []
    settle = graph.RecurrentGraphExecutor.settle_failure

    def failed_settle(executor, error):
        primary.append(error)
        if secondary in ("synchronize", "prior_secondary"):
            runtime.main.failure = RuntimeError("completion uncertain")
        elif secondary == "restore":

            def broken_restore():
                raise RuntimeError("scratch restoration failed")

            monkeypatch.setattr(executor, "_restore_scratch", broken_restore)
        result = settle(executor, error)
        if secondary == "prior_secondary":
            # A later close could succeed; the recorded earlier uncertainty must
            # still prevent eager fallback and preserve quarantine.
            runtime.main.failure = None
        return result

    monkeypatch.setattr(graph.RecurrentGraphExecutor, "settle_failure", failed_settle)
    if secondary == "close":

        def broken_reset():
            raise RuntimeError("graph reset failed")

        create = runtime.new_graph

        def graph_with_broken_reset():
            value = create()
            value.reset = broken_reset
            return value

        monkeypatch.setattr(runtime, "new_graph", graph_with_broken_reset)
    runner = ModelRunner(model, cache)
    with pytest.raises(graph.CaptureBudgetExceeded) as raised:
        runner._enable_recurrent_graph(
            use_graphs=True, limits=low_limits("graph_retained_allocated_bytes")
        )
    assert raised.value is primary[0]
    assert runner._graph_declined is None and runner._decode_executor is not None
    assert runner._decode_executor.status == "failed"
    assert cache._quarantine_reason is not None
    with pytest.raises(RuntimeError, match="quarantined"):
        cache.allocate("new", 2)
    with pytest.raises(RuntimeError):
        runner._recurrent(torch.zeros(1, model.config.hidden_size), ["new"], [0], [0])


@pytest.mark.parametrize(
    "limits",
    [
        {},
        {**asdict(graph.GraphLimits()), "extra": 1},
        low_limits("setup_timeout_s") | {"setup_timeout_s": 0},
        low_limits("cpu_staging_bytes") | {"cpu_staging_bytes": True},
    ],
)
def test_invalid_configuration_never_declines(model, limits):
    runner = ModelRunner(model, fixtures.make_cache(model))
    with pytest.raises(ValueError):
        runner._enable_recurrent_graph(use_graphs=True, limits=limits)
    assert runner._graph_snapshot() == {"enabled": False}
    assert runner._graph_declined is None


def test_wrong_dtype_is_not_hidden_by_low_budget(model):
    cache = fixtures.make_cache(model)
    model = model.to(dtype=torch.float64)
    with pytest.raises(ValueError, match="float32"):
        ModelRunner(model, cache)._enable_recurrent_graph(
            use_graphs=True, limits=low_limits("common_payload_bytes")
        )


def test_declined_engine_can_run_real_requests_then_closes_without_owned_resources(model):
    engine = LLMEngine(model, cache_config=CacheConfig(num_blocks=160))
    engine._enable_recurrent_graph(use_graphs=True, limits=low_limits("common_payload_bytes"))
    engine.add_request("a", [3], SamplingParams(max_tokens=3, ignore_eos=True))
    outputs = []
    while engine.has_unfinished_requests():
        outputs.extend(engine.step())
    assert outputs[-1].finished and len(outputs[-1].token_ids) == 3
    assert engine.model_runner._graph_snapshot()["budget_decline"]["calls"] > 0
    assert engine.cache_manager.num_used_blocks == 0
    engine.model_runner._close_recurrent_graph()
    with pytest.raises(RuntimeError, match="closed"):
        engine.add_request("b", [3])


def test_setup_time_decline_after_submitted_warmup_restores_then_runs_compact(model, monkeypatch):
    cache, reference = fixtures.make_cache(model), fixtures.make_cache(model)
    original = cache.key_cache.clone(), cache.value_cache.clone(), tuple(cache._free_blocks)
    runtime = fixtures.fake_runtime(monkeypatch, cache)
    clock = [10**12]
    monkeypatch.setattr(graph.time, "perf_counter_ns", lambda: clock[0])
    body = graph.RecurrentGraphExecutor._tensor_body

    def slow_warmup(executor, bucket):
        outputs = body(executor, bucket)
        clock[0] += 2 * 10**9
        return outputs

    monkeypatch.setattr(graph.RecurrentGraphExecutor, "_tensor_body", slow_warmup)
    runner = ModelRunner(model, cache)
    runner._enable_recurrent_graph(use_graphs=True, limits=low_limits("setup_timeout_s"))
    declined = runner._graph_snapshot()["budget_decline"]
    assert declined["error"]["limit_name"] == "setup_timeout_s"
    assert declined["error"]["observed"] == 2
    assert declined["attempt_setup"]["events"][-1]["phase"] == "warmup"
    assert declined["attempt_setup"]["scratch"]["restored"]
    assert declined["attempt_failure"]["completion_confirmed"]
    assert runtime.graphs == [] and runner._decode_executor is None
    assert not cache._allocations and tuple(cache._free_blocks) == original[2]
    assert torch.equal(cache.key_cache, original[0]) and torch.equal(cache.value_cache, original[1])
    compact_matches(model, runner, reference)


@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_budget_refusal_preserves_original_on_python_without_exception_notes(
    model, monkeypatch, cleanup_fails
):
    class LegacyBudgetError(graph.CaptureBudgetExceeded):
        def __getattribute__(self, name):
            if name == "add_note":
                raise AttributeError(name)
            return super().__getattribute__(name)

    primary = LegacyBudgetError(graph.GraphLimits(), "setup_timeout_s", 61)
    secondary = RuntimeError("cleanup failed")

    class FailedExecutor:
        def __init__(self, *args, **kwargs):
            self.failure = {"completion_confirmed": cleanup_fails, "secondary": []}
            self.setup_record = {}

        def setup(self):
            raise primary

        def close(self):
            raise secondary

    monkeypatch.setattr(graph, "CaptureBudgetExceeded", LegacyBudgetError)
    monkeypatch.setattr(graph, "RecurrentGraphExecutor", FailedExecutor)
    runner = ModelRunner(model, fixtures.make_cache(model))
    with pytest.raises(LegacyBudgetError) as raised:
        runner._enable_recurrent_graph(use_graphs=True)
    assert raised.value is primary
    if cleanup_fails:
        assert raised.value.__cause__ is secondary
        with pytest.raises(RuntimeError, match="quarantined"):
            runner.cache_manager._require_usable()


@pytest.mark.parametrize("confirmed", [False, True])
def test_repeated_settlement_keeps_cascaded_failure_without_retrying_device(confirmed):
    executor = object.__new__(graph.RecurrentGraphExecutor)
    primary = {"type": "RuntimeError", "message": "first failure"}
    executor.failure = {"primary": primary, "secondary": [], "completion_confirmed": confirmed}
    result = executor.settle_failure(ValueError("later failure"))
    assert result is confirmed
    assert executor.failure == {
        "primary": primary,
        "secondary": [{"type": "ValueError", "message": "later failure"}],
        "completion_confirmed": confirmed,
    }
