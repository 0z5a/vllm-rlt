"""Reusable native INT4 arithmetic qualification without test-runner dependencies."""

import torch

from loopquant.int4_backend import NativeInt4Linear
from loopquant.quantizers import int4_pack, int4_unpack


def check_native_int4(rows: int, columns: int, outputs: int) -> dict:
    torch.manual_seed(19)
    weight = torch.randn(outputs, columns, device="cuda") * 0.03
    packed, scales = int4_pack(weight)
    native = NativeInt4Linear(packed, scales, columns)
    values = torch.randn(rows, columns, device="cuda", dtype=torch.bfloat16)
    reference = int4_unpack(packed, scales.bfloat16().float(), columns).bfloat16()
    expected = torch.nn.functional.linear(values, reference)
    actual = native(values)
    torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.02)
    assert native.packed_weight.numel() * 4 == packed.numel()
    return dict(
        rows=rows,
        columns=columns,
        outputs=outputs,
        max_abs=float((actual.float() - expected.float()).abs().max()),
        packed_bytes=packed.numel(),
        padded_columns=native.padded_columns,
    )
