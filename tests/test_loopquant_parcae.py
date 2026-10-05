import pytest
import torch
from torch.nn import functional as F

from loopquant.adapters.parcae import ParcaeAdapter
from loopquant.parcae_quality import native_parcae_window_nll
from loopquant.quantizers import FP8FakeLinear, ScaleLayout
from loopquant.train import Q0Config, Q0Trainer, TrainingBatch
from tests.loopkv.test_parcae import make_cache, tiny_config
from tests.reference.parcae import dense_parcae_reference
from vllm_rlt.models.parcae import ParcaeForCausalLM


@pytest.mark.parametrize("length", [2, 5])
@pytest.mark.parametrize(
    "dtype",
    [
        torch.float32,
        pytest.param(
            torch.bfloat16,
            marks=pytest.mark.xfail(
                strict=True,
                reason="Retained BF16 manual-equation reduction discrepancy; author SDPA checked separately",
            ),
        ),
    ],
)
def test_parcae_adapter_matches_independent_equations(length, dtype):
    torch.manual_seed(41)
    model = ParcaeForCausalLM(tiny_config()).to(dtype).eval()
    adapter = ParcaeAdapter(model, attention_backend="eager")
    tokens = torch.tensor([[257, 511, 9, 768, 1023][:length]])
    initial = (
        torch.linspace(-0.1, 0.1, length * model.config.n_embd).reshape(1, length, -1).to(dtype)
    )
    states, logits, _ = dense_parcae_reference(model, tokens[0], initial[0])
    result = adapter(tokens, torch.ones_like(tokens, dtype=torch.bool), 3, initial)
    atol, rtol = (3e-5, 3e-5) if dtype == torch.float32 else (0.03, 0.02)
    for expected, actual in zip(states, result.states, strict=True):
        torch.testing.assert_close(actual[0], expected, atol=atol, rtol=rtol)
    torch.testing.assert_close(result.logits[0], logits, atol=atol, rtol=rtol)
    assert model.lm_head.weight is model.transformer.wte.weight
    assert model.freqs_cis.dtype == torch.float32


@pytest.mark.parametrize("seed", [17, 31])
def test_parcae_native_likelihood_matches_adapter_and_restores_rng(seed):
    torch.manual_seed(43)
    model = ParcaeForCausalLM(tiny_config()).eval()
    adapter = ParcaeAdapter(model)
    tokens = torch.tensor([257, 511, 9, 768, 1023])
    with torch.inference_mode(), torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        result = adapter(tokens[None], torch.ones(1, 5, dtype=torch.bool), 3)
        expected = F.cross_entropy(result.logits[0, :-1], tokens[1:], reduction="sum")
    cache = make_cache(model, "native")
    rng = torch.random.get_rng_state().clone()
    actual = native_parcae_window_nll(model, tokens, 3, cache, seed=seed)
    assert actual[0] == pytest.approx(float(expected), abs=3e-5, rel=3e-5)
    assert actual[1] == 4 and cache.num_used_blocks == 0
    assert torch.equal(rng, torch.random.get_rng_state())
    assert actual == native_parcae_window_nll(model, tokens, 3, cache, seed=seed)


def test_parcae_full_recurrence_gradient_and_matched_q0_initial_state():
    torch.manual_seed(41)
    model = ParcaeForCausalLM(tiny_config()).eval().requires_grad_(False)
    adapter = ParcaeAdapter(model)
    projection = FP8FakeLinear(
        model.transformer.core_block[0].attn.c_q, torch.tensor([0.01]), ScaleLayout(3)
    )
    adapter.attach("transformer.core_block.0.attn.c_q", projection)
    tokens = torch.tensor([[257, 511, 9, 768]])
    valid = torch.ones_like(tokens, dtype=torch.bool)
    result = adapter(tokens, valid, 3)
    for state in result.states:
        state.retain_grad()
    result.logits.square().mean().backward()
    assert all(state.grad is not None and state.grad.abs().sum() > 0 for state in result.states)
    assert projection.log_scale.grad is not None and projection.log_scale.grad.abs().sum() > 0
    trainer = Q0Trainer(
        adapter, Q0Config(loops=3, tokens_per_update=4, kl_weight=0.1, trajectory_weight=0.1)
    )
    seen = []

    def capture(module, args):
        seen.append(args[3].clone())

    adapter.register_forward_pre_hook(capture)
    trainer.teacher.register_forward_pre_hook(capture)
    row = trainer.step([TrainingBatch(tokens, valid)])
    assert row["gradient_norm"] > 0 and len(seen) == 2
    assert torch.equal(seen[0], seen[1])
