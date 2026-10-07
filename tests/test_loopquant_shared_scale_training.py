import pytest
import torch

from loopquant.adapters.ouro import OuroAdapter
from loopquant.quantizers import FP8FakeLinear, ScaleLayout
from tests.helpers import tiny_ouro_config
from vllm_rlt.models.ouro import OuroForCausalLM


def test_shared_producer_receives_sum_of_consumer_gradients():
    torch.manual_seed(23)
    model = OuroForCausalLM(tiny_ouro_config()).requires_grad_(False)
    shared, separate = OuroAdapter(model), OuroAdapter(model)
    names = tuple("model.layers.0.self_attn." + name for name in ("q_proj", "k_proj", "v_proj"))
    for adapter in (shared, separate):
        for name in names:
            adapter.attach(
                name, FP8FakeLinear(model.get_submodule(name), torch.tensor([0.01]), ScaleLayout(4))
            )
    shared.share_activation_scales([names])
    tokens = torch.tensor([[5, 7, 2, 11]])
    valid = torch.ones_like(tokens, dtype=torch.bool)
    outputs = [adapter(tokens, valid, 4).logits for adapter in (shared, separate)]
    torch.testing.assert_close(*outputs, atol=0, rtol=0)
    for logits in outputs:
        logits.square().mean().backward()
    parameter = shared.quantized[names[0].replace(".", "__")].log_scale
    gradients = [separate.quantized[name.replace(".", "__")].log_scale.grad for name in names]
    torch.testing.assert_close(parameter.grad, sum(gradients), atol=1e-7, rtol=1e-5)
    assert parameter.grad.abs().sum() > 0
    assert all(shared.quantized[name.replace(".", "__")].log_scale is parameter for name in names)
    assert len(list(shared.quantized.parameters())) == 1
    assert all(value.grad is None for value in model.parameters())


def test_shared_producer_rejects_different_calibrated_scales():
    model = OuroForCausalLM(tiny_ouro_config()).requires_grad_(False)
    adapter = OuroAdapter(model)
    names = ("model.layers.0.mlp.gate_proj", "model.layers.0.mlp.up_proj")
    for name, scale in zip(names, (0.01, 0.02), strict=True):
        adapter.attach(
            name, FP8FakeLinear(model.get_submodule(name), torch.tensor([scale]), ScaleLayout(4))
        )
    with pytest.raises(ValueError, match="matching initial scales"):
        adapter.share_activation_scales([names])
