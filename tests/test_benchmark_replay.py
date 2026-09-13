import math
from collections import Counter

import pytest
import torch

from benchmarks.replay import ReplayEngine
from vllm_lt import CacheConfig, SamplingParams, SchedulerConfig
from vllm_lt.models import OuroConfig, OuroForCausalLM
from vllm_lt.request import Stage


def model():
    torch.manual_seed(7)
    result = OuroForCausalLM(OuroConfig.tiny())
    with torch.no_grad():
        result.model.early_exit_gate.weight.zero_()
        result.model.early_exit_gate.bias.fill_(math.log(0.4 / 0.6))
        result.lm_head.weight.zero_()  # Actual greedy token is always zero.
    return result


def trace(tokens=(0, 7, 8, 9), depths=(4, 2, 4, 3)):
    return {"output_token_ids": list(tokens), "exit_depths": list(depths)}


@pytest.mark.parametrize("mode", ["refill", "no_refill"])
def test_replay_preserves_real_work_and_consumes_exact_trace(mode, monkeypatch):
    traces = {"a": trace(), "b": trace((3, 4, 5), (4, 3, 2))}
    engine = ReplayEngine(
        model(),
        replay=traces,
        cache_config=CacheConfig(64, 2),
        scheduler_config=SchedulerConfig(mode=mode, max_num_batched_tokens=3),
    )
    calls, probabilities, actual_tokens, finalizations = Counter(), [], [], []
    original_recurrent = engine.model.recurrent
    original_coda = engine.model.coda
    original_sample = engine.model_runner._sample
    original_exit = engine._should_exit
    original_finalize = engine.cache_manager.finalize_token

    def recurrent(*args, **kwargs):
        calls["gate_rows"] += args[0].shape[0]
        return original_recurrent(*args, **kwargs)

    def coda(hidden):
        calls["coda_rows"] += hidden.shape[0]
        return original_coda(hidden)

    def sample(logits, request):
        token = original_sample(logits, request)
        actual_tokens.append(token)
        return token

    def should_exit(request):
        probabilities.append((request.loops_done, request.remaining_probability))
        return original_exit(request)

    def finalize(request_id, position, depth):
        original_finalize(request_id, position, depth)
        # Independently verify every layer's final source KV reaches skipped depths.
        cache = engine.cache_manager
        for layer in range(cache.num_layers):
            source = cache.read(layer, request_id, depth, position + 1)
            for skipped in range(depth + 1, cache.max_loops):
                destination = cache.read(layer, request_id, skipped, position + 1)
                torch.testing.assert_close(source[0][-1], destination[0][-1])
                torch.testing.assert_close(source[1][-1], destination[1][-1])
        finalizations.append((request_id, position, depth))

    monkeypatch.setattr(engine.model, "recurrent", recurrent)
    monkeypatch.setattr(engine.model, "coda", coda)
    monkeypatch.setattr(engine.model_runner, "_sample", sample)
    monkeypatch.setattr(engine, "_should_exit", should_exit)
    monkeypatch.setattr(engine.cache_manager, "finalize_token", finalize)
    prompts = {"a": [1, 2, 3], "b": [4, 5]}
    for request_id, prompt in prompts.items():
        engine.add_request(
            request_id,
            prompt,
            SamplingParams(
                max_tokens=len(traces[request_id]["output_token_ids"]),
                ignore_eos=True,
            ),
        )
    outputs, steps = {}, 0
    while engine.has_unfinished_requests():
        steps += 1
        assert steps < 100
        for output in engine.step():
            if output.finished:
                outputs[output.request_id] = output
    for key, output in outputs.items():
        assert output.token_ids == traces[key]["output_token_ids"]
        assert output.exit_depths == traces[key]["exit_depths"]
    assert len(outputs) == 2
    assert calls["coda_rows"] == len(actual_tokens) == 7
    assert actual_tokens == [0] * 7  # Forced nonzero IDs did not bypass actual sampling.
    assert calls["gate_rows"] == 5 * 4 + sum((2, 4, 3, 3, 2))
    for completed, remaining in probabilities:
        assert remaining == pytest.approx(0.6**completed)
    # Each output after the first consumes the previous ID; the last output is never run.
    assert sorted(finalizations) == sorted(
        [
            ("a", 3, 1),
            ("a", 4, 3),
            ("a", 5, 2),
            ("b", 2, 2),
            ("b", 3, 1),
        ]
    )
    assert engine.cache_manager.num_used_blocks == 0


@pytest.mark.parametrize(
    "bad",
    [
        {},
        {"a": trace([], [])},
        {"a": trace([1], [2])},
        {"a": trace([1, 2], [4])},
        {"a": trace([64], [4])},
        {"a": trace([True], [4])},
        {"a": trace([1], [True])},
        {"a": trace([1, 2], [4, 1])},
        {"a": {**trace(), "unknown": 1}},
    ],
)
def test_invalid_trace_fails_before_cache_allocation(bad, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid replay reached base engine allocation")

    monkeypatch.setattr("vllm_lt.engine.llm_engine.LLMEngine.__init__", forbidden)
    with pytest.raises(ValueError):
        ReplayEngine(model(), replay=bad)


@pytest.mark.parametrize(
    "params",
    [
        SamplingParams(max_tokens=3, ignore_eos=True),
        SamplingParams(max_tokens=4),
        SamplingParams(max_tokens=4, ignore_eos=True, temperature=0.5),
        SamplingParams(max_tokens=4, ignore_eos=True, max_loops=3),
    ],
)
def test_request_trace_contract_is_checked_before_admission(params):
    engine = ReplayEngine(model(), replay={"a": trace()})
    with pytest.raises(ValueError):
        engine.add_request("a", [1], params)
    assert not engine.has_unfinished_requests()
    assert engine.cache_manager.num_used_blocks == 0


def test_abort_failure_and_invalid_depth_release_requests(monkeypatch):
    engine = ReplayEngine(model(), replay={"a": trace()})
    params = SamplingParams(max_tokens=4, ignore_eos=True)
    engine.add_request("a", [1], params)
    engine.step()
    assert engine.abort_request("a").finish_reason == "abort"
    assert engine.cache_manager.num_used_blocks == 0
    engine.add_request("a", [1], params)
    engine.step()  # Complete prefill, then corrupt the expected coda depth.
    engine.scheduler.requests["a"].loops_done = 3
    with pytest.raises(RuntimeError, match="different depth"):
        engine.step()
    assert not engine.has_unfinished_requests()
    assert engine.cache_manager.num_used_blocks == 0

    engine.add_request("a", [1], params)

    def fail(_):
        raise RuntimeError("injected runner failure")

    monkeypatch.setattr(engine.model_runner, "execute", fail)
    with pytest.raises(RuntimeError, match="injected"):
        engine.step()
    assert not engine.has_unfinished_requests()


def test_replay_copies_fixture_and_rejects_overrun():
    supplied = {"a": trace()}
    engine = ReplayEngine(model(), replay=supplied)
    supplied["a"]["output_token_ids"][0] = 60
    engine.add_request("a", [1], SamplingParams(max_tokens=4, ignore_eos=True))
    engine.step()
    assert engine.step()[0].token_ids == [0]
    request = engine.scheduler.requests["a"]
    request.stage = Stage.RECURRENT
    request.loops_done = 3  # The upcoming second output requests depth two.
    with pytest.raises(RuntimeError, match="exceeded"):
        engine._should_exit(request)
    engine.abort_request("a")
