import torch

from loopquant.nvfp4 import pack_e2m1, quantize_nvfp4, swizzle_block_scales, unpack_e2m1


def test_e2m1_every_code_roundtrips_and_midpoints_round_to_even():
    codes = torch.arange(256, dtype=torch.uint8).reshape(1, -1)
    values = unpack_e2m1(codes.view(torch.float4_e2m1fn_x2))
    assert torch.equal(pack_e2m1(values).view(torch.uint8), codes)
    edges = torch.tensor([[0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 100.0]])
    expected = torch.tensor([[0, 1, 1, 2, 2, 4, 4, 6]])
    torch.testing.assert_close(unpack_e2m1(pack_e2m1(edges)), expected.float(), atol=0, rtol=0)
    torch.testing.assert_close(unpack_e2m1(pack_e2m1(-edges)), -expected.float(), atol=0, rtol=0)


def test_nvfp4_zero_tail_and_explicit_scale_reuse():
    values = torch.zeros(2, 19)
    values[1, :4] = torch.tensor([-8, -1, 0.5, 3])
    quantized = quantize_nvfp4(values)
    assert quantized.packed.shape == (2, 16)
    assert quantized.block_scales.shape == (2, 2)
    assert torch.count_nonzero(unpack_e2m1(quantized.packed)[:, 19:]) == 0
    actual = quantized.dequantize()
    assert torch.isfinite(actual).all() and actual.shape == values.shape
    assert torch.count_nonzero(actual[0]) == 0
    repeated = quantize_nvfp4(
        values, global_scale=quantized.global_scale, block_scales=quantized.block_scales
    )
    assert torch.equal(repeated.packed.view(torch.uint8), quantized.packed.view(torch.uint8))


def test_swizzle_maps_each_scale_to_the_documented_tile_coordinate():
    scales = torch.arange(129 * 7).reshape(129, 7).float()
    blocked = swizzle_block_scales(scales)
    assert blocked.numel() == 256 * 8
    for row in range(256):
        for column in range(8):
            tile = (row // 128) * 2 + column // 4
            index = tile * 512 + (row % 32) * 16 + ((row % 128) // 32) * 4 + column % 4
            assert blocked[index] == (scales[row, column] if row < 129 and column < 7 else 0)
