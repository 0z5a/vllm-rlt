"""Native FP8 execution, with explicit capability probes and no dtype fallback."""

from dataclasses import asdict, dataclass

import torch
from torch import nn

from .quantizers import fp8_encode


@dataclass(frozen=True)
class FP8Capability:
    device: str
    architecture: tuple[int, int]
    torch_version: str
    cuda_version: str
    kernel_api: str
    weight_format: str = "fp8_e4m3fn"
    activation_format: str = "fp8_e4m3fn"
    scale_granularity: str = "per_tensor"
    packed_weight_copies: int = 1
    fused_producer_quantization: bool = False
    mixed_loop_row_scales: bool = False
    cuda_graph: str = "unverified"
    native_probe: str = "pass"


class NativeFP8Linear(nn.Module):
    """One FP8 weight storage shared across calls; scalar activation dequant scale.

    This initial unfused backend is for encoding/ABI validation, not the strong
    deployment baseline. Per-row scales and fusion need separate GPU evidence.
    """

    def __init__(self, linear: nn.Linear, activation_scale: torch.Tensor) -> None:
        super().__init__()
        if linear.weight.device.type != "cuda" or linear.bias is not None:
            raise ValueError("native FP8 requires CUDA and bias-free weights")
        if linear.in_features % 16 or linear.out_features % 16:
            raise ValueError("native FP8 requires K and N divisible by 16")
        if activation_scale.numel() != 1 or not bool(
            torch.isfinite(activation_scale).all() & (activation_scale > 0).all()
        ):
            raise ValueError("expected a positive finite scalar activation scale")
        weight = linear.weight.detach().float()
        scale = weight.abs().amax().clamp_min(torch.finfo(torch.float32).tiny) / 448
        self.register_buffer("packed_weight", fp8_encode(weight, scale).t())
        self.register_buffer("weight_scale", scale.reshape(1))
        self.register_buffer("activation_scale", activation_scale.detach().float().reshape(1))

    @classmethod
    def from_packed(
        cls, packed: torch.Tensor, weight_scale: torch.Tensor, activation_scale: torch.Tensor
    ) -> "NativeFP8Linear":
        if packed.device.type != "cuda" or packed.dtype != torch.float8_e4m3fn:
            raise ValueError("native packed weights must be CUDA E4M3 tensors")
        if packed.ndim != 2 or packed.stride() != (1, packed.shape[0]):
            raise ValueError("packed GEMM weights must be column-major")
        if weight_scale.numel() != 1 or activation_scale.numel() != 1:
            raise ValueError("this backend supports scalar scales only")
        module = cls.__new__(cls)
        nn.Module.__init__(module)
        module.register_buffer("packed_weight", packed)
        module.register_buffer("weight_scale", weight_scale.float().reshape(1))
        module.register_buffer("activation_scale", activation_scale.float().reshape(1))
        return module

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 2 or values.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("native FP8 expects a packed FP16/BF16 matrix")
        rows = values.shape[0]
        values = nn.functional.pad(values, (0, 0, 0, (-rows) % 16))
        activation = fp8_encode(values, self.activation_scale)
        return torch._scaled_mm(
            activation,
            self.packed_weight,
            scale_a=self.activation_scale,
            scale_b=self.weight_scale,
            out_dtype=values.dtype,
            use_fast_accum=False,
        )[:rows]

    def reference(self, values: torch.Tensor) -> torch.Tensor:
        activation = fp8_encode(values, self.activation_scale).float() * self.activation_scale
        weight = self.packed_weight.float() * self.weight_scale
        return (activation @ weight).to(values.dtype)


def probe_fp8(device: torch.device) -> dict[str, str | bool | int | tuple[int, int]]:
    """Run a native operation before advertising support; failures propagate."""
    linear = nn.Linear(64, 32, bias=False, device=device, dtype=torch.bfloat16)
    values = torch.randn(17, 64, device=device, dtype=torch.bfloat16)
    native = NativeFP8Linear(linear, torch.tensor(0.01, device=device))
    actual = native(values)
    torch.testing.assert_close(actual, native.reference(values), atol=0.02, rtol=0.02)
    torch.cuda.synchronize(device)
    return asdict(
        FP8Capability(
            torch.cuda.get_device_name(device),
            torch.cuda.get_device_capability(device),
            torch.__version__,
            str(torch.version.cuda),
            "torch._scaled_mm",
        )
    )
