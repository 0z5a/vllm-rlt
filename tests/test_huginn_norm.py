"""Huginn inference normalization must retain the reference's rounding."""

import pytest
import torch

from vllm_rlt.layers import RMSNorm
from vllm_rlt.models.huginn import HuginnRMSNorm


@pytest.mark.gpu
@pytest.mark.parametrize("width", [32, 384, 5280])
@pytest.mark.parametrize("rows", [1, 4, 31])
@pytest.mark.parametrize("strided", [False, True])
@torch.inference_mode()
def test_epilogue_matches_reference_bits_and_replays(width, rows, strided):
    torch.manual_seed(42)
    reference = RMSNorm(width, eps=1e-6).to(device="cuda", dtype=torch.bfloat16)
    fused = HuginnRMSNorm(width, eps=1e-6).to(device="cuda", dtype=torch.bfloat16)
    reference.weight.copy_(torch.randn_like(reference.weight))
    fused.load_state_dict(reference.state_dict())
    storage = torch.randn(rows, width * (2 if strided else 1), device="cuda", dtype=torch.bfloat16)
    hidden = storage[:, :width]
    for scale in (0.0, 0.001, 1.0, 1000.0):
        hidden.copy_(torch.randn_like(hidden) * scale)
        assert torch.equal(fused(hidden), reference(hidden))

    # Warm on the capture stream and replay with changed input and weights.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            fused(hidden)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        actual = fused(hidden)
    for _ in range(3):
        hidden.copy_(torch.randn_like(hidden))
        reference.weight.copy_(torch.randn_like(reference.weight))
        fused.weight.copy_(reference.weight)
        graph.replay()
        assert torch.equal(actual, reference(hidden))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_cpu_training_keeps_reference_gradients_and_checkpoint(dtype):
    reference = RMSNorm(32).to(dtype)
    fused = HuginnRMSNorm(32).to(dtype)
    fused.load_state_dict(reference.state_dict(), strict=True)
    first = torch.randn(2, 3, 32, dtype=dtype, requires_grad=True)
    second = first.detach().clone().requires_grad_()
    expected, actual = reference(first), fused(second)
    assert torch.equal(actual, expected)
    expected.float().square().sum().backward()
    actual.float().square().sum().backward()
    assert torch.equal(first.grad, second.grad)
    assert torch.equal(reference.weight.grad, fused.weight.grad)
