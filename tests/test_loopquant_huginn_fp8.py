import pytest

from tests.loopquant_huginn_fp8_checks import check_huginn_fp8


@pytest.mark.gpu
@pytest.mark.parametrize("policy", ["DYN", "SH1", "ST2", "LOOP"])
@pytest.mark.parametrize("graphs", [False, True])
@pytest.mark.parametrize("fuse_norm", [False, True])
def test_huginn_fp8_native_reload_and_mixed_depth_engine(tmp_path, policy, graphs, fuse_norm):
    check_huginn_fp8(tmp_path / "export", policy, graphs, fuse_norm)
    (tmp_path / "export/tensors.pt").unlink()
