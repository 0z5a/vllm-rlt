import math

import pytest
import torch

from vllm_lt.models import OuroConfig, OuroForCausalLM
from vllm_lt.models.reference import dense_reference
from vllm_lt.models.serial_oracle import ExitPolicy, SerialOuroOracle


def tiny_model(*, dtype=torch.float32):
    torch.manual_seed(29)
    return OuroForCausalLM(OuroConfig.tiny()).to(dtype=dtype)


def oracle_for(model, capacity=16):
    return SerialOuroOracle(model.config, model.state_dict(), capacity=capacity)


def constant_hazard_model(hazard):
    model = tiny_model()
    with torch.no_grad():
        model.model.early_exit_gate.weight.zero_()
        model.model.early_exit_gate.bias.fill_(math.log(hazard / (1 - hazard)))
    return model


def test_full_depth_prefill_and_supplied_inputs_match_functional_dense_reference():
    model = tiny_model()
    prompt, inputs = [2, 4, 6], [8, 10, 12, 14, 16, 18, 20, 22]
    reference = dense_reference(model, torch.tensor(prompt + inputs), 4)
    oracle = oracle_for(model, capacity=len(prompt) + len(inputs))
    observed = {}

    def observer(boundary, values):
        if boundary.operation in ("loop_hidden", "gate_logits"):
            observed[boundary.operation, boundary.depth] = values.clone()

    traces = [oracle.prefill(prompt, observer=observer)]
    for depth, (hidden, gates, _) in enumerate(reference, 1):
        torch.testing.assert_close(
            observed["loop_hidden", depth],
            hidden[: len(prompt)],
            atol=3e-6,
            rtol=3e-5,
        )
        torch.testing.assert_close(
            observed["gate_logits", depth],
            gates[: len(prompt)],
            atol=2e-6,
            rtol=3e-5,
        )
    for token in inputs:
        traces.append(oracle.advance(token))
    assert [trace.output_index for trace in traces] == list(range(9))
    assert [trace.position for trace in traces] == list(range(2, 11))
    assert [trace.exit_depth for trace in traces] == [4] * 9
    for trace in traces:
        torch.testing.assert_close(
            trace.hidden,
            reference[-1][0][trace.position],
            atol=3e-6,
            rtol=3e-5,
        )
        torch.testing.assert_close(
            trace.logits,
            reference[-1][2][trace.position],
            atol=2e-6,
            rtol=3e-5,
        )
    assert oracle.length == 11  # No extra input forward for the ninth prediction.
    assert bool(oracle.snapshot_kv().initialized.all())
    with pytest.raises(ValueError, match="capacity exhausted"):
        oracle.advance(24)
    oracle.close()


def test_oracle_never_calls_native_execution_helpers_or_queries_cuda(monkeypatch):
    model = tiny_model()
    weights = model.state_dict()

    def forbidden(*args, **kwargs):
        pytest.fail("independent oracle called a native execution helper or CUDA discovery")

    for path in (
        "vllm_lt.models.ouro.OuroForCausalLM.prelude",
        "vllm_lt.models.ouro.OuroForCausalLM.recurrent",
        "vllm_lt.models.ouro.OuroForCausalLM.coda",
        "vllm_lt.models.ouro.OuroRMSNorm.forward",
        "vllm_lt.models.ouro.OuroRotaryEmbedding.forward",
        "vllm_lt.models.ouro.OuroAttention.forward",
        "vllm_lt.models.reference.dense_reference",
        "vllm_lt.core.kv_cache_manager.KVCacheManager.__init__",
        "vllm_lt.core.scheduler.Scheduler.schedule",
        "vllm_lt.engine.llm_engine.LLMEngine.__init__",
        "torch.cuda.is_available",
        "torch.cuda.device_count",
        "torch.cuda.init",
    ):
        monkeypatch.setattr(path, forbidden)
    oracle = SerialOuroOracle(model.config, weights, capacity=5)
    oracle.prefill([1, 2, 3])
    assert oracle.advance(4, forced_depth=2).exit_depth == 2
    assert oracle.advance(5, forced_depth=4).exit_depth == 4
    oracle.close()


def test_layer_specific_last_exited_copy_then_deeper_neighbor():
    model = tiny_model()
    oracle = oracle_for(model)
    oracle.prefill([1, 3, 5])
    before = oracle.snapshot_kv()
    computed = {}

    def record(boundary, values):
        if boundary.operation in ("key", "value"):
            computed[boundary.positions[0], boundary.depth, boundary.layer, boundary.operation] = (
                values[0].clone()
            )

    shallow = oracle.advance(7, forced_depth=2, observer=record)
    assert shallow.position == 3
    after = oracle.snapshot_kv(positions=[3])
    for layer in range(model.config.num_hidden_layers):
        for depth in range(4):
            source = min(depth + 1, 2)
            torch.testing.assert_close(
                after.keys[depth, layer, 0], computed[3, source, layer, "key"], atol=0, rtol=0
            )
            torch.testing.assert_close(
                after.values[depth, layer, 0], computed[3, source, layer, "value"], atol=0, rtol=0
            )
    assert not torch.equal(after.keys[1, 0], after.keys[1, 1])
    unchanged = oracle.snapshot_kv(positions=range(3))
    torch.testing.assert_close(unchanged.keys, before.keys, atol=0, rtol=0)
    torch.testing.assert_close(unchanged.values, before.values, atol=0, rtol=0)

    # The next token crosses a conceptual size-four block boundary and computes all depths.
    deep = oracle.advance(9, forced_depth=4, observer=record)
    assert deep.position == 4 and deep.exit_depth == 4
    final = oracle.snapshot_kv(positions=[3, 4])
    torch.testing.assert_close(final.keys[:, :, 0], after.keys[:, :, 0], atol=0, rtol=0)
    for layer in range(model.config.num_hidden_layers):
        for depth in range(4):
            torch.testing.assert_close(
                final.keys[depth, layer, 1], computed[4, depth + 1, layer, "key"], atol=0, rtol=0
            )
            torch.testing.assert_close(
                final.values[depth, layer, 1],
                computed[4, depth + 1, layer, "value"],
                atol=0,
                rtol=0,
            )
    assert bool(final.initialized.all())
    oracle.close()


def test_hole_in_one_layer_is_rejected_and_invalidates_partial_execution():
    oracle = oracle_for(tiny_model())
    oracle.prefill([1, 2])
    oracle.initialized[2, 1, 0] = False
    with pytest.raises(RuntimeError, match="uninitialized.*depth 3, layer 1"):
        oracle.advance(3, forced_depth=4)
    assert oracle.length == 2
    with pytest.raises(RuntimeError, match="invalid after"):
        oracle.advance(3)
    oracle.close()
    assert oracle.key_cache is None and oracle.initialized is None


def test_future_uninitialized_storage_is_never_attention_history():
    oracle = oracle_for(tiny_model())
    with torch.inference_mode():
        oracle.key_cache.fill_(math.nan)
        oracle.value_cache.fill_(math.nan)
    assert bool(oracle.prefill([1, 2]).logits.isfinite().all())
    assert bool(oracle.advance(3, forced_depth=2).logits.isfinite().all())
    assert not bool(oracle.snapshot_kv(positions=[3]).initialized.any())
    oracle.close()


@pytest.mark.parametrize(
    "policy,expected_depth,expected_cdf",
    [
        (ExitPolicy(exit_threshold=0.6), 2, [0.4, 0.64]),
        (ExitPolicy(min_loops=3, exit_threshold=0.5), 3, [0.4, 0.64, 0.784]),
        (ExitPolicy(max_loops=2, exit_threshold=1.0), 2, [0.4, 0.64]),
    ],
)
def test_actual_hazards_accumulate_before_minimum_depth(policy, expected_depth, expected_cdf):
    oracle = oracle_for(constant_hazard_model(0.4))
    assert oracle.prefill([1]).exit_depth == 4
    trace = oracle.advance(2, policy=policy)
    assert trace.exit_depth == expected_depth
    assert trace.gate_probabilities == pytest.approx([0.4] * expected_depth)
    assert trace.cumulative_probabilities == pytest.approx(expected_cdf)
    assert bool(oracle.snapshot_kv().initialized.all())
    oracle.close()


def test_threshold_one_and_forced_depth_do_not_reinterpret_actual_gates():
    model = constant_hazard_model(0.4)
    with torch.no_grad():
        model.model.early_exit_gate.bias.fill_(100)
    oracle = oracle_for(model)
    oracle.prefill([1])
    fixed = oracle.advance(2)
    assert fixed.exit_depth == 4
    assert fixed.cumulative_probabilities == (1.0,) * 4
    forced = oracle.advance(3, forced_depth=4, policy=ExitPolicy(exit_threshold=0.0))
    assert forced.exit_depth == 4 and forced.gate_probabilities == (1.0,) * 4
    oracle.close()


def test_boundary_contract_final_logits_and_owned_snapshots():
    model = tiny_model()
    oracle = oracle_for(model)
    records = []

    def observer(boundary, values):
        records.append((boundary, tuple(values.shape), values.dtype))

    first = oracle.prefill([1, 2], observer=observer)
    assert len([row for row in records if row[0].operation == "logits"]) == 1
    expected = {
        "attention_input",
        "query",
        "key",
        "value",
        "attention_output",
        "layer_output",
        "loop_hidden",
        "gate_logits",
        "logits",
    }
    assert {row[0].operation for row in records} == expected
    assert all(row[0].depth in (1, 2, 3, 4) for row in records)
    assert all(row[1][0] == len(row[0].positions) for row in records)
    assert records[-1][0].positions == (1,) and records[-1][1] == (1, 64)
    saved_hidden, saved_logits = first.hidden.clone(), first.logits.clone()
    snapshot = oracle.snapshot_kv()
    with torch.inference_mode():
        snapshot.keys.zero_()
    assert not torch.equal(snapshot.keys, oracle.snapshot_kv().keys)
    oracle.advance(3, forced_depth=2)
    torch.testing.assert_close(first.hidden, saved_hidden, atol=0, rtol=0)
    torch.testing.assert_close(first.logits, saved_logits, atol=0, rtol=0)
    oracle.close()
    torch.testing.assert_close(first.logits, saved_logits, atol=0, rtol=0)


def test_two_oracles_share_weights_but_not_request_state():
    model = tiny_model()
    weights = model.state_dict()
    original = {name: value.clone() for name, value in weights.items()}
    first, second = oracle_for(model), oracle_for(model)
    first.prefill([1, 2])
    second.prefill([3])
    before = second.snapshot_kv()
    first.advance(4, forced_depth=2)
    first.close()
    torch.testing.assert_close(second.snapshot_kv().keys, before.keys, atol=0, rtol=0)
    for name, value in weights.items():
        torch.testing.assert_close(value, original[name], atol=0, rtol=0)
    assert second.advance(5).position == 1
    second.close()
    second.close()
    with pytest.raises(RuntimeError, match="closed"):
        second.prefill([1])


def test_observer_failure_invalidates_oracle_without_mutating_shared_weights():
    model = tiny_model()
    oracle = oracle_for(model)

    def fail(boundary, values):
        raise RuntimeError("injected observer failure")

    with pytest.raises(RuntimeError, match="injected"):
        oracle.prefill([1, 2], observer=fail)
    with pytest.raises(RuntimeError, match="invalid after"):
        oracle.prefill([1, 2])
    oracle.close()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_loops": True},
        {"max_loops": 0},
        {"min_loops": 3, "max_loops": 2},
        {"exit_threshold": True},
        {"exit_threshold": math.nan},
        {"exit_threshold": 1.1},
    ],
)
def test_invalid_policy_rejected(kwargs):
    with pytest.raises(ValueError):
        ExitPolicy(**kwargs)


@pytest.mark.parametrize("value", [True, 0, -1, 1.5, 129])
def test_invalid_capacity_rejected_before_storage_allocation(value):
    with pytest.raises(ValueError, match="capacity"):
        oracle_for(tiny_model(), capacity=value)


def test_bad_inputs_do_not_consume_positions():
    oracle = oracle_for(tiny_model(), capacity=4)
    with pytest.raises(RuntimeError, match="prefill"):
        oracle.advance(1)
    for prompt in ([], [True], [64], list(range(5))):
        with pytest.raises(ValueError):
            oracle.prefill(prompt)
    oracle.prefill([1, 2])
    with pytest.raises(RuntimeError, match="only once"):
        oracle.prefill([3])
    for arguments in (
        {"forced_depth": True},
        {"forced_depth": 1},
        {"forced_depth": 5},
        {"policy": {}},
        {"policy": ExitPolicy(max_loops=5)},
    ):
        with pytest.raises(ValueError):
            oracle.advance(3, **arguments)
    assert oracle.length == 2
    assert oracle.advance(3).position == 2
    with pytest.raises(ValueError, match="unique"):
        oracle.snapshot_kv(positions=[0, 0])
    with pytest.raises(ValueError, match="capacity"):
        oracle.snapshot_kv(positions=[4])
    oracle.close()


def test_bf16_reference_keeps_model_dtype_and_float32_rotary_constants():
    model = tiny_model(dtype=torch.bfloat16)
    oracle = oracle_for(model)
    first = oracle.prefill([1, 2])
    second = oracle.advance(3, forced_depth=2)
    assert first.logits.dtype == second.logits.dtype == torch.bfloat16
    assert oracle.key_cache.dtype == torch.bfloat16
    assert oracle._inv_freq.dtype == torch.float32
    assert all(math.isfinite(value) for value in second.gate_logits)
    oracle.close()


def test_bf16_gate_uses_published_linear_bias_rounding():
    model = tiny_model(dtype=torch.bfloat16)
    with torch.no_grad():
        model.model.early_exit_gate.bias.fill_(0.11328125)
    oracle = oracle_for(model)
    hidden, gates = {}, {}

    def observer(boundary, values):
        if boundary.operation == "loop_hidden":
            hidden[boundary.depth] = values.clone()
        elif boundary.operation == "gate_logits":
            gates[boundary.depth] = values.clone()

    oracle.prefill([1, 2, 3, 4], observer=observer)
    weight = model.model.early_exit_gate.weight
    bias = model.model.early_exit_gate.bias
    distinct = False
    for depth, values in hidden.items():
        expected = torch.nn.functional.linear(values, weight, bias).squeeze(-1)
        split_bias = (torch.nn.functional.linear(values, weight) + bias).squeeze(-1)
        torch.testing.assert_close(gates[depth], expected, atol=0, rtol=0)
        distinct |= not torch.equal(expected, split_bias)
    assert distinct  # This fixture detects rounding bias as a separate BF16 operation.
    oracle.close()
