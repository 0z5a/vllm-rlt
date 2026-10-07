import torch

from loopquant.qat import ouro_qat_payload, prepare_ouro_q0
from loopquant.quantizers import ScaleLayout
from loopquant.train import Q0Config, Q0Trainer, TrainingBatch
from tests.helpers import tiny_ouro_config
from vllm_rlt.models.ouro import OuroForCausalLM


def test_trained_shared_producers_keep_native_weight_bytes_and_layout(tmp_path):
    torch.manual_seed(17)
    model = OuroForCausalLM(tiny_ouro_config())
    scales = {
        name: torch.tensor([0.01, 0.02])
        for name, module in model.named_modules()
        if name.startswith("model.layers.") and isinstance(module, torch.nn.Linear)
    }
    student = prepare_ouro_q0(model, scales, ScaleLayout(4, (2,)))
    trainer = Q0Trainer(student, Q0Config(tokens_per_update=4))
    assert len(trainer.parameters) == 4 * model.config.num_hidden_layers
    before = {
        key: value.packed_weight.view(torch.uint8).clone()
        for key, value in student.quantized.items()
    }
    tokens = torch.tensor([[5, 7, 2, 11]])
    trainer.step([TrainingBatch(tokens, torch.ones_like(tokens, dtype=torch.bool))])
    trainer.save(tmp_path / "q0.pt")
    layout, trained_scales, packed = ouro_qat_payload(student)
    assert layout == ScaleLayout(4, (2,))
    for name, (weight, weight_scales) in packed.items():
        module = student.quantized[name.replace(".", "__")]
        assert weight.data_ptr() == module.packed_weight.data_ptr()
        assert weight.stride() == (1, weight.shape[0])
        assert torch.equal(weight.t().view(torch.uint8), before[name.replace(".", "__")])
        torch.testing.assert_close(
            weight_scales, module.weight_scale.expand(1, weight.shape[1]), atol=0, rtol=0
        )
        torch.testing.assert_close(trained_scales[name], module.log_scale.exp(), atol=0, rtol=0)
    for index in range(model.config.num_hidden_layers):
        prefix = f"model.layers.{index}.self_attn."
        assert torch.equal(trained_scales[prefix + "q_proj"], trained_scales[prefix + "v_proj"])
    resumed = prepare_ouro_q0(model, scales, layout)
    other = Q0Trainer(resumed, trainer.config)
    other.resume(tmp_path / "q0.pt")
    assert len(other.parameters) == len(trainer.parameters)
    assert all(parameter.grad is None for parameter in model.parameters())
