"""GPU checks shared by pytest and the finite immutable-environment probe."""

import torch
from torch.nn.functional import ScalingType, SwizzleType, scaled_mm

from loopquant.fused_nvfp4 import NVFP4InputQuantizer, NVFP4Workspace
from loopquant.nvfp4 import quantize_nvfp4, swizzle_block_scales


def check_fused_nvfp4(rows: int, columns: int, outputs: int, norm: bool, fixed: bool) -> dict:
    torch.manual_seed(17)
    values = torch.randn(rows, columns, device="cuda", dtype=torch.bfloat16)
    values[0].zero_()
    values[-1, :2] = torch.tensor([16, -16], device="cuda", dtype=torch.bfloat16)
    gain = torch.randn(columns, device="cuda", dtype=torch.bfloat16) if norm else None
    global_scale = torch.tensor(0.01, device="cuda")
    local = (
        (2.0 ** ((torch.arange(columns // 16, device="cuda") % 8) - 1)).to(torch.float8_e4m3fn)
        if fixed
        else None
    )
    quantizer = NVFP4InputQuantizer(global_scale, columns, block_scales=local)
    workspace = NVFP4Workspace.allocate(rows, columns, torch.device("cuda"))
    pointers = workspace.packed.data_ptr(), workspace.swizzled_scales.data_ptr()

    def reference():
        x = values
        if gain is not None:
            inverse = torch.rsqrt(values.float().square().mean(-1, keepdim=True) + 1e-6)
            x = (values.float() * inverse).bfloat16() * gain
        return quantize_nvfp4(
            x,
            global_scale=global_scale,
            block_scales=local.expand(rows, -1) if local is not None else None,
        )

    def compare_bytes():
        expected = reference()
        assert torch.equal(workspace.packed, expected.packed.view(torch.uint8))
        assert torch.equal(
            workspace.swizzled_scales.view(torch.uint8),
            swizzle_block_scales(expected.block_scales).view(torch.uint8),
        )
        return expected

    quantizer(values, workspace, norm_weight=gain)
    activation = compare_bytes()
    weight = torch.randn(outputs, columns, device="cuda", dtype=torch.bfloat16) / columns**0.5
    packed_weight = quantize_nvfp4(weight)
    actual = scaled_mm(
        workspace.packed.view(torch.float4_e2m1fn_x2),
        packed_weight.packed.t(),
        scale_a=[workspace.swizzled_scales, quantizer.global_scale],
        scale_recipe_a=[ScalingType.BlockWise1x16, ScalingType.TensorWise],
        scale_b=[swizzle_block_scales(packed_weight.block_scales), packed_weight.global_scale],
        scale_recipe_b=[ScalingType.BlockWise1x16, ScalingType.TensorWise],
        swizzle_a=[SwizzleType.SWIZZLE_32_4_4, SwizzleType.NO_SWIZZLE],
        swizzle_b=[SwizzleType.SWIZZLE_32_4_4, SwizzleType.NO_SWIZZLE],
        output_dtype=torch.bfloat16,
    )
    expected = (activation.dequantize() @ packed_weight.dequantize().t()).bfloat16()
    torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.02)
    max_abs = float((actual.float() - expected.float()).abs().max())
    quantizer(values, workspace, norm_weight=gain)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        quantizer(values, workspace, norm_weight=gain)
    for factor in [0.75, 1.25, 0.0]:
        values.mul_(factor)
        graph.replay()
        compare_bytes()
        assert pointers == (workspace.packed.data_ptr(), workspace.swizzled_scales.data_ptr())
    return dict(
        rows=rows,
        columns=columns,
        outputs=outputs,
        norm=norm,
        fixed_blocks=fixed,
        packed_bytes_exact=True,
        swizzled_scales_exact=True,
        gemm_max_abs=max_abs,
        graph_replays=3,
        scratch_pointers_stable=True,
    )
