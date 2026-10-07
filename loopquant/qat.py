"""Bind Q0 scales to native producers and deploy their existing encoded weights."""

import torch
from torch import nn

from vllm_rlt.models.ouro import OuroForCausalLM

from .adapters.ouro import OuroAdapter
from .quantizers import FP8FakeLinear, ScaleLayout
from .serving import FP8Scratch, quantize_ouro_native


def prepare_ouro_q0(
    model: OuroForCausalLM, scales: dict[str, torch.Tensor], layout: ScaleLayout
) -> OuroAdapter:
    model.eval().requires_grad_(False)
    student = OuroAdapter(model)
    for name, module in model.named_modules():
        if name.startswith("model.layers.") and isinstance(module, nn.Linear):
            student.attach(name, FP8FakeLinear(module, scales[name], layout))
    groups = [
        tuple(f"model.layers.{index}.{suffix}" for suffix in members)
        for index in range(model.config.num_hidden_layers)
        for members in (
            ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"),
            ("mlp.gate_proj", "mlp.up_proj"),
        )
    ]
    student.share_activation_scales(groups)
    return student


def ouro_qat_payload(
    student: OuroAdapter,
) -> tuple[ScaleLayout, dict[str, torch.Tensor], dict[str, tuple[torch.Tensor, torch.Tensor]]]:
    """Preserve training weight storage; only expand the native scale-vector ABI."""
    names = {
        name
        for name, module in student.model.named_modules()
        if name.startswith("model.layers.") and isinstance(module, nn.Linear)
    }
    if not names or names != {key.replace("__", ".") for key in student.quantized}:
        raise ValueError("QAT deployment requires every physical core projection")
    layouts = {module.layout for module in student.quantized.values()}
    if len(layouts) != 1:
        raise ValueError("QAT deployment requires one registered loop layout")
    packed, scales = {}, {}
    for name in sorted(names):
        module = student.quantized[name.replace(".", "__")]
        weight = module.packed_weight.detach().t()
        packed[name] = (weight, module.weight_scale.expand(1, weight.shape[1]).contiguous())
        scales[name] = module.log_scale.detach().exp()
    return layouts.pop(), scales, packed


def deploy_ouro_q0(student: OuroAdapter, *, max_rows: int) -> FP8Scratch:
    """Consume the training model after saving the optimizer checkpoint."""
    layout, scales, packed = ouro_qat_payload(student)
    return quantize_ouro_native(
        student.model, scales, layout, "static", max_rows=max_rows, packed=packed
    )
