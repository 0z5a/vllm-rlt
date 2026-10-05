import torch

from loopquant.adapters.huginn import HuginnAdapter
from loopquant.quantizers import FP8FakeLinear, ScaleLayout
from loopquant.train import Q0Config, Q0Trainer, TrainingBatch
from tests.reference.huginn import dense_huginn_reference
from tests.test_huginn import tiny_huginn_config
from vllm_rlt.models.huginn import HuginnForCausalLM


def test_huginn_explicit_initial_state_and_full_recurrence_gradient():
    torch.manual_seed(17)
    model = HuginnForCausalLM(tiny_huginn_config(mean_recurrence=4)).requires_grad_(False).eval()
    adapter = HuginnAdapter(model)
    tokens = torch.tensor([[5, 7, 2, 11]])
    valid = torch.ones_like(tokens, dtype=torch.bool)
    initial = torch.linspace(-0.2, 0.2, tokens.numel() * model.config.n_embd).reshape(1, 4, -1)
    expected_states, expected_logits = dense_huginn_reference(model, tokens[0], initial[0])
    actual = adapter(tokens, valid, 4, initial)
    for expected, observed in zip(expected_states, actual.states, strict=True):
        torch.testing.assert_close(observed[0], expected, atol=3e-6, rtol=3e-5)
    torch.testing.assert_close(actual.logits[0], expected_logits, atol=3e-6, rtol=3e-5)
    name = "transformer.core_block.0.attn.Wqkv"
    quantized = FP8FakeLinear(
        model.transformer.core_block[0].attn.Wqkv, torch.tensor([0.01]), ScaleLayout(4)
    )
    adapter.attach(name, quantized)
    result = adapter(tokens, valid, 4, initial)
    for state in result.states:
        state.retain_grad()
    result.logits.square().mean().backward()
    assert quantized.log_scale.grad is not None and quantized.log_scale.grad.abs().sum() > 0
    assert all(state.grad is not None and state.grad.abs().sum() > 0 for state in result.states)
    assert model.lm_head.weight is model.transformer.wte.weight


def test_huginn_q0_teacher_uses_the_same_stochastic_initial_state():
    torch.manual_seed(17)
    model = HuginnForCausalLM(tiny_huginn_config(mean_recurrence=4)).requires_grad_(False).eval()
    adapter = HuginnAdapter(model)
    adapter.attach(
        "transformer.core_block.0.attn.Wqkv",
        FP8FakeLinear(
            model.transformer.core_block[0].attn.Wqkv, torch.tensor([0.01]), ScaleLayout(4)
        ),
    )
    trainer = Q0Trainer(
        adapter, Q0Config(tokens_per_update=4, kl_weight=0.1, trajectory_weight=0.1)
    )
    seen = []

    def before(module, args):
        seen.append(args[3].clone())

    adapter.register_forward_pre_hook(before)
    trainer.teacher.register_forward_pre_hook(before)
    batch = TrainingBatch(torch.tensor([[5, 7, 2, 11]]), torch.ones(1, 4, dtype=torch.bool))
    result = trainer.step([batch])
    assert result["gradient_norm"] > 0 and len(seen) == 2
    torch.testing.assert_close(seen[0], seen[1], atol=0, rtol=0)
