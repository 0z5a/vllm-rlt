import hashlib
import json
import math
from types import SimpleNamespace

import pytest
import torch

from vllm_lt.config import CacheConfig, SchedulerConfig
from vllm_lt.core.kv_cache_manager import KVCacheManager
from vllm_lt.models import OuroConfig, OuroForCausalLM
from vllm_lt.models.serial_oracle import ExitPolicy, SerialOuroOracle
from vllm_lt.sampling_params import SamplingParams
from vllm_lt.validation.native import ValidationEngine, observe_native, snapshot_native


@pytest.fixture(autouse=True)
def no_cuda_discovery(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("native validation CPU tests must not discover or initialize CUDA")

    for name in ("is_available", "device_count", "init"):
        monkeypatch.setattr(torch.cuda, name, forbidden)


def model():
    torch.manual_seed(43)
    return OuroForCausalLM(OuroConfig.tiny())


def fixture(key="one", prompt=None, *, forced=True):
    return {
        "fixture_id": key,
        "prompt_token_ids": prompt or [2, 3, 4, 5, 6],
        "continuation_input_ids": [8, 10, 12, 14, 16, 18, 20, 22],
        "history_policy": "forced" if forced else "fixed",
        "forced_exit_depths": [4, 2, 4, 3, 2, 4, 2, 3, 4] if forced else [4] * 9,
    }


def make_engine(rows, *, mode="refill", history="teacher_forced", native_model=None):
    return ValidationEngine(
        native_model or model(),
        fixtures=rows,
        history_mode=history,
        cache_config=CacheConfig(num_blocks=96, block_size=2),
        scheduler_config=SchedulerConfig(max_num_seqs=2, max_num_batched_tokens=3, mode=mode),
    )


def add_requests(engine, rows, *, threshold=1.0):
    for row in rows:
        engine.add_request(
            row["fixture_id"],
            row["prompt_token_ids"],
            SamplingParams(
                max_tokens=9,
                min_loops=2,
                max_loops=4,
                temperature=0.0,
                ignore_eos=True,
                exit_threshold=threshold,
            ),
        )


def drain(engine):
    outputs = {}
    for _ in range(500):
        if not engine.has_unfinished_requests():
            return outputs
        for output in engine.step():
            if output.finished:
                outputs[output.request_id] = output
    pytest.fail("native validation did not finish within the bounded step budget")


def digest(tokens):
    return hashlib.sha256(json.dumps(tokens, separators=(",", ":")).encode()).hexdigest()


def hook_state(engine):
    cache_methods = ["write", "attend"] + [
        name
        for name in ("_write_prepared", "_attend_prepared")
        if hasattr(engine.cache_manager, name)
    ]
    return (
        {
            (id(obj), name): (name in vars(obj), vars(obj).get(name))
            for obj, name in (
                (engine.model, "recurrent"),
                (engine.model_runner, "execute"),
                *((engine.cache_manager, name) for name in cache_methods),
                (engine.scheduler, "finish"),
            )
        },
        {id(module): dict(module._forward_hooks) for module in engine.model.modules()},
    )


@pytest.mark.parametrize("mode", ["refill", "no_refill"])
def test_packed_forced_histories_match_independent_oracle_and_populated_kv(mode):
    rows = [fixture(), fixture("two", [7, 6, 5])]
    rows[1]["forced_exit_depths"] = [4, 4, 2, 3, 4, 2, 4, 3, 2]
    native_model = model()
    expected_boundaries, expected_traces, expected_kv = {}, {}, {}
    for row in rows:
        prompt, inputs = row["prompt_token_ids"], row["continuation_input_ids"]
        oracle = SerialOuroOracle(
            native_model.config,
            native_model.state_dict(),
            capacity=len(prompt) + len(inputs),
        )

        def record(boundary, values):
            for index, position in enumerate(boundary.positions):
                if position >= len(prompt) - 1:
                    key = (
                        row["fixture_id"],
                        position,
                        boundary.depth,
                        boundary.layer,
                        boundary.operation,
                    )
                    assert key not in expected_boundaries
                    expected_boundaries[key] = values[index].clone()

        traces = [oracle.prefill(prompt, observer=record)]
        traces.extend(
            oracle.advance(token, forced_depth=depth, observer=record)
            for token, depth in zip(inputs, row["forced_exit_depths"][1:])
        )
        expected_traces[row["fixture_id"]] = traces
        expected_kv[row["fixture_id"]] = oracle.snapshot_kv()
        oracle.close()

    engine = make_engine(rows, mode=mode, native_model=native_model)
    add_requests(engine, rows)
    actual_boundaries, completed = {}, {}
    traversals = []
    recurrent = native_model.recurrent

    def recorded_recurrent(hidden, ids, depths, positions, cache):
        traversals.append((tuple(ids), tuple(depths), tuple(positions)))
        return recurrent(hidden, ids, depths, positions, cache)

    native_model.recurrent = recorded_recurrent
    previous = hook_state(engine)

    def sink(metadata, values):
        key = (
            metadata["fixture_id"],
            metadata["positions"][0],
            metadata["depth"],
            metadata["layer"],
            metadata["operation"],
        )
        assert key not in actual_boundaries
        row = engine.fixtures[metadata["fixture_id"]]
        index = metadata["output_index"]
        assert metadata["positions"] == [len(row["prompt_token_ids"]) + index - 1]
        assert metadata["history_sha256"] == digest(
            row["prompt_token_ids"] + row["continuation_input_ids"][:index]
        )
        actual_boundaries[key] = values.clone()

    def on_completed(request_id, cache):
        expected = expected_kv[request_id]
        assert request_id in engine.scheduler.requests
        assert cache.num_used_blocks > 0
        completed[request_id] = []
        for start in range(0, len(expected.positions), 2):
            positions = list(expected.positions[start : start + 2])
            snapshot = snapshot_native(cache, request_id, positions)
            assert snapshot.keys.shape[:3] == (4, 2, len(positions))
            assert snapshot.initialized.device.type == "cpu"
            assert bool(snapshot.initialized.all())
            torch.testing.assert_close(
                snapshot.keys, expected.keys[:, :, positions], atol=5e-6, rtol=1e-4
            )
            torch.testing.assert_close(
                snapshot.values, expected.values[:, :, positions], atol=5e-6, rtol=1e-4
            )
            completed[request_id].extend(snapshot.positions)

    with observe_native(engine, sink, on_completed_kv=on_completed):
        outputs = drain(engine)
    assert hook_state(engine) == previous
    assert set(actual_boundaries) == set(expected_boundaries)
    for key in actual_boundaries:
        torch.testing.assert_close(
            actual_boundaries[key], expected_boundaries[key], atol=5e-6, rtol=1e-4
        )
    for row in rows:
        key = row["fixture_id"]
        assert completed[key] == list(range(len(row["prompt_token_ids"]) + 8))
        assert outputs[key].exit_depths == row["forced_exit_depths"]
        assert outputs[key].token_ids[:8] == row["continuation_input_ids"]
        assert len(outputs[key].token_ids) == len(engine.traces[key]) == 9
        for index, (actual, reference) in enumerate(zip(engine.traces[key], expected_traces[key])):
            assert actual["output_index"] == index
            assert actual["position"] == len(row["prompt_token_ids"]) + index - 1
            assert actual["exit_depth"] == reference.exit_depth
            assert actual["gate_logits"] == pytest.approx(reference.gate_logits, abs=5e-6)
            assert actual["gate_probabilities"] == pytest.approx(
                reference.gate_probabilities, abs=5e-6
            )
            assert actual["cumulative_probabilities"] == pytest.approx(
                reference.cumulative_probabilities, abs=5e-6
            )
            logits = actual_boundaries[
                key, actual["position"], actual["exit_depth"], None, "logits"
            ]
            assert actual["actual_token_id"] == int(logits.argmax())
            assert actual["emitted_token_id"] == outputs[key].token_ids[index]
            assert actual["history_sha256"] == digest(
                row["prompt_token_ids"] + row["continuation_input_ids"][:index]
            )
        assert outputs[key].token_ids[-1] == engine.traces[key][-1]["actual_token_id"]
    if mode == "refill":
        assert any(len(set(depths)) > 1 for _, depths, _ in traversals)
    assert not engine.scheduler.requests and engine.cache_manager.num_used_blocks == 0


@pytest.mark.parametrize("history", ["teacher_forced", "live_gate"])
def test_actual_samples_are_recorded_before_input_override_and_live_gate_uses_real_cdf(history):
    row = fixture(prompt=[2, 3], forced=history == "live_gate")
    native_model = model()
    with torch.no_grad():
        native_model.lm_head.weight.zero_()  # Actual argmax is EOS=0 at every prediction.
        native_model.model.early_exit_gate.weight.zero_()
        native_model.model.early_exit_gate.bias.fill_(math.log(0.4 / 0.6))
    engine = make_engine([row], history=history, native_model=native_model)
    add_requests(engine, [row], threshold=0.7 if history == "live_gate" else 1.0)
    with observe_native(engine, lambda *_: None):
        output = drain(engine)["one"]
    expected_depth = 3 if history == "live_gate" else 4
    assert output.exit_depths == [4] + [expected_depth] * 8
    assert [trace["actual_token_id"] for trace in engine.traces["one"]] == [0] * 9
    assert output.token_ids == (
        [0] * 9 if history == "live_gate" else row["continuation_input_ids"] + [0]
    )
    for index, trace in enumerate(engine.traces["one"]):
        depth = 4 if index == 0 else expected_depth
        assert trace["gate_probabilities"] == pytest.approx([0.4] * depth)
        assert trace["cumulative_probabilities"] == pytest.approx(
            [1 - 0.6**step for step in range(1, depth + 1)]
        )
        assert trace["top_two_margin"] == 0.0
        assert trace["history_sha256"] == digest(row["prompt_token_ids"] + output.token_ids[:index])
        assert trace["gate_usage"] == (
            "full_depth_prefill_diagnostic" if index == 0 else "actual_decode"
        )
    assert engine.cache_manager.num_used_blocks == 0


def test_live_gate_native_and_oracle_follow_identical_actual_predictions():
    row = fixture(prompt=[2, 3, 4])
    native_model = model()
    oracle = SerialOuroOracle(native_model.config, native_model.state_dict(), capacity=11)
    expected = [oracle.prefill(row["prompt_token_ids"])]
    for _ in range(8):
        expected.append(
            oracle.advance(int(expected[-1].logits.argmax()), policy=ExitPolicy(exit_threshold=0.7))
        )
    oracle.close()
    engine = make_engine([row], history="live_gate", native_model=native_model)
    add_requests(engine, [row], threshold=0.7)
    with observe_native(engine, lambda *_: None):
        output = drain(engine)["one"]
    assert output.token_ids == [int(trace.logits.argmax()) for trace in expected]
    assert output.exit_depths == [trace.exit_depth for trace in expected]


def test_fragmented_snapshot_uses_each_depth_page_table_and_owns_selected_storage():
    cache = KVCacheManager(2, 2, 4, 32, 2, 4)
    for key in ("a", "b", "c"):
        assert cache.allocate(key, 4)
    cache.free("a")
    cache.free("c")
    assert cache.allocate("fragmented", 8)
    assert any(
        list(cache.get_block_table("fragmented", depth))
        != list(
            range(
                cache.get_block_table("fragmented", depth)[0],
                cache.get_block_table("fragmented", depth)[0] + 4,
            )
        )
        for depth in range(4)
    )
    positions = [7, 0, 3, 4]
    expected = torch.arange(4 * 2 * 8 * 2 * 4, dtype=torch.float32).reshape(4, 2, 8, 2, 4)
    with pytest.raises(RuntimeError, match="uninitialized"):
        snapshot_native(cache, "fragmented", positions)
    order = [4, 1, 7, 0, 6, 2, 5, 3]
    for depth in range(4):
        for layer in range(2):
            keys = expected[depth, layer, order]
            cache.write(layer, ["fragmented"] * 8, [depth] * 8, order, keys, -keys)
    snapshot = snapshot_native(cache, "fragmented", positions)
    assert snapshot.positions == tuple(positions)
    assert snapshot.keys.shape == (4, 2, 4, 2, 4)
    torch.testing.assert_close(snapshot.keys, expected[:, :, positions], atol=0, rtol=0)
    torch.testing.assert_close(snapshot.values, -expected[:, :, positions], atol=0, rtol=0)
    cache.key_cache.zero_()
    torch.testing.assert_close(snapshot.keys, expected[:, :, positions], atol=0, rtol=0)
    for invalid in ([], [0, 0], [True], [-1], [8], [1.5]):
        with pytest.raises(ValueError):
            snapshot_native(cache, "fragmented", invalid)
    cache.free("fragmented")
    assert cache.allocate("reused", 8)
    with pytest.raises(RuntimeError, match="uninitialized"):
        snapshot_native(cache, "reused", [0])
    cache.free("reused")
    cache.free("b")
    assert cache.num_used_blocks == 0


def test_cancelled_request_pages_are_reused_without_stale_initialized_history():
    cancelled, survivor = fixture("cancelled", [1]), fixture("survivor", [7, 6, 5])
    engine = make_engine([cancelled, survivor])
    oracle = SerialOuroOracle(engine.model.config, engine.model.state_dict(), capacity=11)
    oracle.prefill(survivor["prompt_token_ids"])
    for token, depth in zip(survivor["continuation_input_ids"], survivor["forced_exit_depths"][1:]):
        oracle.advance(token, forced_depth=depth)
    expected = oracle.snapshot_kv()
    oracle.close()
    completed, old_blocks = [], set()

    def on_completed(request_id, cache):
        assert request_id == "survivor"
        new_blocks = {
            block for depth in range(4) for block in cache.get_block_table(request_id, depth)
        }
        assert new_blocks & old_blocks
        snapshot = snapshot_native(cache, request_id, expected.positions)
        torch.testing.assert_close(snapshot.keys, expected.keys, atol=5e-6, rtol=1e-4)
        torch.testing.assert_close(snapshot.values, expected.values, atol=5e-6, rtol=1e-4)
        completed.append(request_id)

    before = hook_state(engine)
    with observe_native(engine, lambda *_: None, on_completed_kv=on_completed):
        add_requests(engine, [cancelled])
        assert engine.step() == []  # Full prefill has written KV but not emitted a token.
        old_blocks.update(
            block
            for depth in range(4)
            for block in engine.cache_manager.get_block_table("cancelled", depth)
        )
        assert engine.abort_request("cancelled").finish_reason == "abort"
        assert engine.cache_manager.num_used_blocks == 0
        add_requests(engine, [survivor])
        assert set(drain(engine)) == {"survivor"}
    assert hook_state(engine) == before
    assert completed == ["survivor"]
    assert engine.traces["cancelled"] == []
    assert not engine.scheduler.requests and engine.cache_manager.num_used_blocks == 0


@pytest.mark.parametrize(
    "operation", ["attention_input", "key", "query", "layer_output", "gate_logits", "logits"]
)
def test_sink_failure_restores_hooks_and_frees_affected_request(operation):
    row = fixture(prompt=[2])
    engine = make_engine([row])
    add_requests(engine, [row])
    before = hook_state(engine)

    def fail(metadata, values):
        if metadata["operation"] == operation:
            raise RuntimeError("injected sink failure")

    with pytest.raises(RuntimeError, match="injected sink"):
        with observe_native(engine, fail):
            drain(engine)
    assert hook_state(engine) == before
    assert not engine.scheduler.requests and engine.cache_manager.num_used_blocks == 0


def test_completion_snapshot_failure_preserves_traces_and_releases_cache():
    row = fixture(prompt=[2])
    engine = make_engine([row])
    add_requests(engine, [row])
    before = hook_state(engine)

    def fail(request_id, cache):
        assert request_id == "one" and cache.num_used_blocks > 0
        assert len(engine.traces[request_id]) == 9
        raise RuntimeError("injected completion failure")

    with pytest.raises(RuntimeError, match="injected completion"):
        with observe_native(engine, lambda *_: None, on_completed_kv=fail):
            drain(engine)
    assert hook_state(engine) == before
    assert not engine.scheduler.requests and engine.cache_manager.num_used_blocks == 0
    assert len(engine.traces["one"]) == 9


def test_partial_hook_installation_restores_preexisting_hooks(monkeypatch):
    engine = make_engine([fixture()])
    existing = engine.model.model.layers[0].input_layernorm.register_forward_hook(lambda *_: None)
    before = hook_state(engine)

    def fail(*args, **kwargs):
        raise RuntimeError("injected registration failure")

    monkeypatch.setattr(engine.model.model.layers[0].self_attn, "register_forward_hook", fail)
    with pytest.raises(RuntimeError, match="registration"):
        with observe_native(engine, lambda *_: None):
            pytest.fail("setup must fail before entering the context")
    assert hook_state(engine) == before
    existing.remove()


def test_caller_exception_restores_hooks_without_changing_another_model_instance():
    engine, other = make_engine([fixture()]), make_engine([fixture("other")])
    before, untouched = hook_state(engine), hook_state(other)
    with pytest.raises(RuntimeError, match="caller"):
        with observe_native(engine, lambda *_: None):
            assert hook_state(other) == untouched
            raise RuntimeError("caller failure")
    assert hook_state(engine) == before
    assert hook_state(other) == untouched


@pytest.mark.parametrize("corruption", [None, "owner", "allocation", "depth", "position"])
def test_prepared_observer_checks_actual_row_identity_before_emitting(corruption):
    row = fixture(prompt=[2])
    engine = make_engine([row])
    add_requests(engine, [row])
    allocation = object()
    events = []
    key = torch.zeros(1, 2, 4)
    value = torch.ones_like(key)
    query = key + 2
    cache = SimpleNamespace(
        _prepare_batch=lambda: None,
        _get_allocation=lambda request_id: allocation,
        _write_prepared=lambda *args: events.append("write"),
        _attend_prepared=lambda *args: events.append("attend"),
    )
    batch = SimpleNamespace(owner=cache, rows=((allocation, 0, 0),))
    if corruption == "owner":
        batch.owner = object()
    if corruption == "allocation":
        batch.rows = ((object(), 0, 0),)
    if corruption == "depth":
        batch.rows = ((allocation, 1, 0),)
    if corruption == "position":
        batch.rows = ((allocation, 0, 1),)

    def recurrent(hidden, request_ids, depths, positions, current_cache):
        current_cache._write_prepared(0, batch, key, value)
        return current_cache._attend_prepared(0, batch, query)

    def sink(metadata, tensor):
        events.append(metadata["operation"])
        assert metadata["fixture_id"] == "one"
        assert metadata["positions"] == [0] and metadata["depth"] == 1
        assert metadata["output_index"] == 0
        assert metadata["history_sha256"] == digest([2])

    engine.cache_manager = cache
    engine.model.recurrent = recurrent
    before = hook_state(engine)
    with observe_native(engine, sink):
        if corruption:
            with pytest.raises(RuntimeError, match="native diagnostic context"):
                engine.model.recurrent(torch.zeros(1, 16), ["one"], [0], [0], cache)
            assert events == []  # Wrong identities cannot publish plausible evidence.
        else:
            engine.model.recurrent(torch.zeros(1, 16), ["one"], [0], [0], cache)
            assert events == ["write", "key", "value", "query", "attend"]
    assert hook_state(engine) == before
