import pytest
import torch

from loopquant.int4_backend import tinygemm_inputs
from tests.loopquant_int4_checks import check_native_int4


def test_tinygemm_encoding_preserves_signed_codes_and_declares_bf16_scale_rounding():
    codes = torch.arange(-8, 8, dtype=torch.int8).repeat(2, 8)
    nibbles = codes.to(torch.uint8) & 15
    packed = nibbles[:, 0::2] | nibbles[:, 1::2] << 4
    scales = torch.tensor([[0.010123], [0.200987]])
    unsigned, parameters = tinygemm_inputs(packed, scales)
    decoded_codes = torch.stack((unsigned >> 4, unsigned & 15), -1).flatten(1).to(torch.int16) - 8
    assert torch.equal(decoded_codes, codes)
    assert parameters.shape == (1, 2, 2) and parameters.dtype == torch.bfloat16
    assert torch.equal(parameters[:, :, 0].t(), scales.bfloat16())
    assert parameters[:, :, 1].count_nonzero() == 0


@pytest.mark.gpu
@pytest.mark.parametrize("rows", [1, 17, 32, 64, 128])
@pytest.mark.parametrize("columns,outputs", [(128, 32), (2048, 2048), (2048, 5632), (5280, 32)])
def test_native_int4_matches_the_same_bf16_decoded_weights(rows, columns, outputs):
    check_native_int4(rows, columns, outputs)
