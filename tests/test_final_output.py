"""Offline output suppression preserves the complete sampling trace."""

import pytest
import torch

from tests.helpers import tiny_ouro_config
from vllm_rlt import (
    LLM,
    CacheConfig,
    ExecutionConfig,
    ExitConfig,
    SamplingParams,
    SpeculativeConfig,
)
from vllm_rlt.engine.llm_engine import LLMEngine
from vllm_rlt.models import OuroForCausalLM
from vllm_rlt.request import RequestOutput


def make_engine(*, speculative=False, device="cpu", asynchronous=False):
    torch.manual_seed(123)
    config = tiny_ouro_config(head_dim=64) if device == "cuda" else tiny_ouro_config()
    model = OuroForCausalLM(config).to(device)
    return LLMEngine(
        model,
        cache_config=CacheConfig(256, 2),
        speculative_config=SpeculativeConfig(3) if speculative else None,
        execution_config=ExecutionConfig(async_scheduling=asynchronous),
        exit_config=ExitConfig("ouro_delayed") if asynchronous else None,
        attention_backend="triton" if device == "cuda" else "torch",
    )


def complete(engine, final_only):
    outputs = []
    for _ in range(1000):
        if not engine.has_unfinished_requests():
            return outputs
        outputs.extend(engine.step(final_only=final_only))
    pytest.fail("scheduler did not finish")


@pytest.mark.parametrize("speculative", [False, True])
@pytest.mark.parametrize("temperature", [0.0, 0.8])
@pytest.mark.parametrize("reason", ["length", "stop", "eos"])
def test_complete_trace_and_materialization_count(monkeypatch, speculative, temperature, reason):
    baseline = make_engine(speculative=speculative)
    candidate = make_engine(speculative=speculative)
    if reason != "length":
        with torch.no_grad():
            baseline.model.lm_head.weight.zero_()
            candidate.model.lm_head.weight.zero_()
        temperature = 0.0
    params = SamplingParams(
        max_tokens=9,
        temperature=temperature,
        seed=17,
        logprobs=0,
        logprobs_mode="processed",
        ignore_eos=reason != "eos",
        stop_token_ids=(0,) if reason == "stop" else (),
    )
    for engine in (baseline, candidate):
        for index, prompt in enumerate([[2, 3], [4, 5, 6]]):
            engine.add_request(str(index), prompt, params)
    expected = [o for o in complete(baseline, False) if o.finished]
    materialized = []
    original = RequestOutput.from_request

    def record(cls, request):
        materialized.append(request.request_id)
        return original(request)

    monkeypatch.setattr(RequestOutput, "from_request", classmethod(record))
    actual = complete(candidate, True)
    assert actual == expected
    assert len(materialized) == len(actual) == 2
    assert all(len(o.log_probs) == len(o.token_ids) == len(o.exit_depths) for o in actual)
    assert candidate.cache_manager.num_used_blocks == 0


def test_abort_exports_suppressed_tokens_and_scores():
    engine = make_engine()
    engine.add_request("a", [2, 3], SamplingParams(max_tokens=8, logprobs=0, ignore_eos=True))
    request = engine.scheduler.requests["a"]
    while len(request.generated_token_ids) < 3:
        assert engine.step(final_only=True) == []
    output = engine.abort_request("a")
    assert output.finished and output.finish_reason == "abort"
    assert output.token_ids == request.generated_token_ids
    assert len(output.log_probs) == len(output.exit_depths) == 3
    assert engine.cache_manager.num_used_blocks == 0


def test_default_streaming_and_mid_request_switch():
    engine = make_engine()
    engine.add_request("a", [2, 3], SamplingParams(max_tokens=4, logprobs=0, ignore_eos=True))
    request = engine.scheduler.requests["a"]
    while not request.generated_token_ids:
        assert engine.step(final_only=True) == []
    outputs = complete(engine, False)
    assert [len(o.token_ids) for o in outputs] == [2, 3, 4]
    assert all(len(o.log_probs) == len(o.token_ids) for o in outputs)


def test_offline_facade_materializes_once_per_request(monkeypatch):
    engine = make_engine()
    calls = []
    original = RequestOutput.from_request

    def record(cls, request):
        calls.append(request.request_id)
        return original(request)

    monkeypatch.setattr(RequestOutput, "from_request", classmethod(record))
    outputs = LLM(engine).generate(
        [[2, 3], [4, 5]], SamplingParams(max_tokens=5, logprobs=0, ignore_eos=True)
    )
    assert calls == [o.request_id for o in outputs]
    assert [len(o.log_probs) for o in outputs] == [5, 5]


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_async_final_output_matches_streaming(device):
    params = SamplingParams(max_tokens=7, logprobs=0, ignore_eos=True)
    expected = make_engine(device=device, asynchronous=True)
    actual = make_engine(device=device, asynchronous=True)
    for engine in (expected, actual):
        engine.add_request("a", [2, 3], params)
        engine.add_request("b", [4, 5, 6], params)
    assert complete(actual, True) == [o for o in complete(expected, False) if o.finished]
    assert actual.cache_manager.num_used_blocks == 0
