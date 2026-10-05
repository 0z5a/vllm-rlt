import pytest

from tests.loopquant_parcae_fp8_checks import check_parcae_fp8


@pytest.mark.gpu
@pytest.mark.parametrize("policy", ["DYN", "SH1", "ST2", "LOOP"])
@pytest.mark.parametrize("graphs", [False, True])
@pytest.mark.parametrize("fuse_norm", [False, True])
def test_parcae_fp8_native_reload_and_fixed_depth_engine(tmp_path, policy, graphs, fuse_norm):
    check_parcae_fp8(tmp_path / "export", policy, graphs, fuse_norm)
    (tmp_path / "export/tensors.pt").unlink()
