"""Packed ABI preservation through persistence and native engine reloading."""

import pytest
import torch

from loopquant.fused_fp8 import RowwiseFP8Linear
from tests.loopquant_serving_checks import check_fused_native_model


def test_packed_reload_preserves_encoded_storage_and_column_major_layout(tmp_path):
    weight = (
        torch.arange(32 * 64).reshape(32, 64).float().remainder(448).to(torch.float8_e4m3fn).t()
    )
    scales = torch.full((1, 32), 0.125)
    path = tmp_path / "packed.pt"
    torch.save(dict(weight=weight, scales=scales), path)
    tensors = torch.load(path, weights_only=True)
    loaded = RowwiseFP8Linear.from_packed(tensors["weight"], tensors["scales"])
    assert loaded.packed_weight.data_ptr() == tensors["weight"].data_ptr()
    assert loaded.weight_scales.data_ptr() == tensors["scales"].data_ptr()
    assert loaded.packed_weight.stride() == (1, 64)
    assert torch.equal(loaded.packed_weight.t().view(torch.uint8), weight.t().view(torch.uint8))
    assert not list(loaded.parameters())
    with pytest.raises(ValueError, match="column-major"):
        RowwiseFP8Linear.from_packed(weight.contiguous(), scales)


@pytest.mark.gpu
@pytest.mark.parametrize("policy", ["DYN", "SH1", "ST2", "LOOP"])
@pytest.mark.parametrize("graphs", [False, True])
@pytest.mark.parametrize("fuse_norm", [False, True])
def test_exported_native_engine_keeps_mixed_depth_tokens(policy, graphs, fuse_norm, tmp_path):
    check_fused_native_model(policy, graphs, fuse_norm, export_folder=tmp_path / "export")
