"""CPU tests for native generation stopping and cleanup."""

from types import SimpleNamespace

import pytest
import torch


class Engine:
    def __init__(self, failure=None):
        self.failure = failure
        self.active = False
        self.steps = 0
        self.aborts = 0

    def add_request(self, request_id, prompt_ids, params):
        assert request_id == "gsm8k"
        assert prompt_ids == [1, 2]
        assert params.min_loops == params.max_loops == 4
        assert params.max_tokens == 8
        self.active = True

    def has_unfinished_requests(self):
        return self.active

    def step(self):
        self.steps += 1
        if self.failure == "execution":
            raise RuntimeError("execution failed")
        return [
            SimpleNamespace(
                token_ids=[10, 11][: self.steps],
                exit_depths=[3 if self.failure == "depth" else 4] * self.steps,
                finished=False,
            )
        ]

    def abort_request(self, request_id):
        assert request_id == "gsm8k"
        self.aborts += 1
        self.active = False


@pytest.fixture
def generator(monkeypatch):
    pytest.importorskip("lm_eval")
    from benchmarks import gsm8k_backends as backends

    original_tensor = torch.tensor

    def cpu_tensor(*args, **kwargs):
        kwargs.pop("device", None)
        return original_tensor(*args, **kwargs)

    monkeypatch.setattr(backends.torch, "tensor", cpu_tensor)
    monkeypatch.setattr(
        backends,
        "stop_sequences_criteria",
        lambda tokenizer, stops, length, batch: (
            lambda ids, scores: original_tensor([ids.shape[1] >= length + 2])
        ),
    )
    generator = backends.Generator.__new__(backends.Generator)
    generator.backend = "native"
    generator.tokenizer = SimpleNamespace(eos_token_id=0, decode=lambda ids, **kw: "answer STOP")
    generator.llm = SimpleNamespace(engine=Engine())
    return generator


def test_stop_releases_request_without_an_extra_token(generator):
    result = generator.generate([1, 2], 8, ["STOP"])
    assert result["token_ids"] == [10, 11]
    assert result["text"] == "answer "
    assert result["finish_reason"] == "stop"
    assert generator.llm.engine.steps == 2
    assert generator.llm.engine.aborts == 1
    assert not generator.llm.engine.active


@pytest.mark.parametrize("failure", ["execution", "depth"])
def test_failure_releases_request(generator, failure):
    generator.llm.engine.failure = failure
    with pytest.raises(RuntimeError):
        generator.generate([1, 2], 8, ["STOP"])
    assert generator.llm.engine.aborts == 1
    assert not generator.llm.engine.active
