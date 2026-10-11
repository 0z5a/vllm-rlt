"""Contract tests for the extracted Sampler: greedy, top-k and top-p.

``Sampler`` only consumes a generator the caller resolved, so these tests create
one themselves (mirroring what ``RngRegistry.acquire`` does in production) and
keep ``RngRegistry``'s own creation and lifetime contracts separate. The tests pin
the current arithmetic; they are not a specification of what sampling should do in
the future.
"""

import pytest
import torch

import vllm_rlt.worker.sampler as sampler_module
from vllm_rlt.sampling_params import SamplingParams
from vllm_rlt.worker.sampler import RngRegistry, Sampler


def seeded_generator(params, device=torch.device("cpu")):
    """Caller-side generator creation, mirroring ``RngRegistry.acquire``."""
    return torch.Generator(device=device).manual_seed(params.seed)


def sample_many(sampler, logits, params, count=1):
    """Sample ``count`` tokens with a generator this test owns (greedy gets None)."""
    generator = None if params.temperature == 0 else seeded_generator(params)
    return [int(sampler.sample(logits, params, generator)) for _ in range(count)]


def test_greedy_returns_argmax_without_a_generator():
    logits = torch.tensor([0.1, 2.0, 1.0, -1.0])
    # top_k/top_p are deliberately set: the temperature == 0 short circuit runs
    # first, so greedy ignores them and needs no RNG.
    params = SamplingParams(temperature=0.0, top_k=1, top_p=0.5)
    token = Sampler().sample(logits, params, None)
    assert token.item() == int(logits.argmax())
    assert token.dim() == 0
    assert token.dtype == torch.long
    assert token.device == logits.device


def test_greedy_leaves_a_supplied_generator_untouched():
    logits = torch.tensor([0.1, 3.0, -1.0])
    generator = torch.Generator().manual_seed(9)
    before = generator.get_state().clone()
    token = Sampler().sample(logits, SamplingParams(temperature=0.0), generator)
    assert token.item() == 1
    assert torch.equal(generator.get_state(), before)


def test_non_greedy_requires_a_generator():
    # None would silently draw from the process-wide default generator, so the
    # contract is an explicit error rather than an implicit fallback.
    logits = torch.tensor([0.1, 2.0, 1.0])
    with pytest.raises(ValueError, match="requires a generator"):
        Sampler().sample(logits, SamplingParams(temperature=1.0), None)


def test_non_greedy_advances_the_supplied_generator():
    logits = torch.linspace(1.0, -1.0, 64)
    params = SamplingParams(temperature=0.7, top_k=8, seed=42)
    generator = seeded_generator(params)
    before = generator.get_state().clone()
    Sampler().sample(logits, params, generator)
    assert not torch.equal(generator.get_state(), before)


GREEDY_CASES = [
    torch.tensor([0.1, 2.0, 1.0, -1.0]),
    torch.tensor([-3.0, -3.5, -2.0, -9.0]),
    torch.tensor([5.0, 4.999, 4.998]),
]


@pytest.mark.parametrize("logits", GREEDY_CASES)
def test_top_k_one_is_equivalent_to_greedy(logits):
    sampler = Sampler()
    greedy = sample_many(sampler, logits, SamplingParams(temperature=0.0))[0]
    # top_k=1 makes the threshold the maximum, so masked_fill leaves exactly one
    # candidate and the result cannot depend on the RNG stream. Checking several
    # logits rows and seeds pins that mechanism, not one coincidental argmax.
    for seed in (0, 7, 12345):
        params = SamplingParams(temperature=1.0, top_k=1, seed=seed)
        assert sample_many(sampler, logits, params)[0] == greedy


def test_top_k_keeps_every_token_tied_at_the_threshold():
    sampler = Sampler()
    # top_k=2 selects a threshold of 3.0, which three tokens share. The mask is
    # ``logits < threshold``, so the candidate set is larger than k.
    logits = torch.tensor([3.0, 3.0, 3.0, 0.0])
    params = SamplingParams(temperature=1.0, top_k=2, seed=11)
    observed = set(sample_many(sampler, logits, params, count=200))
    assert observed <= {0, 1, 2}
    assert len(observed) > 1


def test_top_p_keeps_at_least_one_token():
    sampler = Sampler()
    # A top_p far below the head probability would empty the candidate set; the
    # mask shift keeps the single most likely token instead.
    logits = torch.linspace(1.0, 0.0, 100)
    params = SamplingParams(temperature=1.0, top_p=0.005, seed=3)
    assert set(sample_many(sampler, logits, params, count=20)) == {0}


def test_top_p_one_does_not_truncate_the_support():
    sampler = Sampler()
    logits = torch.tensor([1.0, 0.9, 0.8, 0.7])
    truncated = SamplingParams(temperature=1.0, top_p=0.5, seed=13)
    untruncated = SamplingParams(temperature=1.0, top_p=1.0, seed=13)
    assert set(sample_many(sampler, logits, truncated, count=200)) <= {0, 1}
    # Index 3 is the least likely token, so it is reachable only while the
    # support is intact. This asserts reachability over a fixed 200-draw stream
    # (p(miss) ~ 0.79**200), not the exact draw sequence: the exact sequence
    # would couple the test to torch's multinomial implementation.
    assert 3 in set(sample_many(sampler, logits, untruncated, count=200))


def sequence(sampler, logits, params, count=32):
    return sample_many(sampler, logits, params, count=count)


def test_same_seed_reproduces_the_same_sequence():
    sampler = Sampler()
    logits = torch.linspace(1.0, -1.0, 64)
    params = SamplingParams(temperature=0.7, top_k=8, top_p=0.9, seed=42)
    assert sequence(sampler, logits, params) == sequence(sampler, logits, params)


def test_different_seeds_produce_different_sequences():
    sampler = Sampler()
    logits = torch.linspace(1.0, -1.0, 64)
    base = dict(temperature=0.7, top_k=8, top_p=0.9)
    assert sequence(sampler, logits, SamplingParams(seed=1, **base)) != sequence(
        sampler, logits, SamplingParams(seed=2, **base)
    )


def test_generator_is_reused_and_advanced_across_calls():
    sampler = Sampler()
    logits = torch.linspace(1.0, -1.0, 64)
    params = SamplingParams(temperature=0.7, top_k=8, seed=42)
    generator = seeded_generator(params)
    state = generator.get_state().clone()
    sampler.sample(logits, params, generator)
    # Later calls must advance the same generator instead of reseeding it, which
    # is what makes a multi-token request reproducible.
    sampler.sample(logits, params, generator)
    assert not torch.equal(generator.get_state(), state)


def test_rng_registry_creates_lazily_and_reuses_the_slot():
    registry = RngRegistry()
    params = SamplingParams(temperature=1.0, seed=42)
    assert registry.peek("r") is None

    first = registry.acquire("r", params, torch.device("cpu"))
    assert registry.peek("r") is first
    # A later draw for the same request must advance the same generator.
    assert registry.acquire("r", params, torch.device("cpu")) is first


def test_rng_registry_creates_the_generator_on_the_requested_device(monkeypatch):
    # The registry is the single creation owner, so pin that it forwards the device
    # instead of falling back to the process default. The fake device never reaches
    # torch: the recorded constructor returns a real CPU generator for the caller.
    recorded = []
    real_generator = torch.Generator

    def recording_generator(device=None):
        recorded.append(device)
        return real_generator(device="cpu")

    monkeypatch.setattr(sampler_module.torch, "Generator", recording_generator)
    registry = RngRegistry()
    registry.acquire("r", SamplingParams(temperature=1.0, seed=1), torch.device("meta"))
    assert recorded == [torch.device("meta")]


def test_rng_registry_drop_removes_only_that_requests_slot():
    registry = RngRegistry()
    params = SamplingParams(temperature=1.0, seed=5)
    dropped = registry.acquire("dropped", params, torch.device("cpu"))
    kept = registry.acquire("kept", params, torch.device("cpu"))

    registry.drop("dropped")
    assert registry.peek("dropped") is None
    assert registry.peek("kept") is kept
    # Termination can run twice for the same request; the second drop is a no-op.
    registry.drop("dropped")
    assert registry.peek("kept") is kept
    assert dropped is not kept
