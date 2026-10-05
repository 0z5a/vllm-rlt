import pytest

from tests.loopquant_loopformer_fp8_checks import check_loopformer_fp8


@pytest.mark.gpu
@pytest.mark.parametrize("policy", ["DYN", "SH1", "ST2", "LOOP"])
@pytest.mark.parametrize("graphs", [False, True])
@pytest.mark.parametrize("compact", [False, True])
def test_loopformer_fp8_native_reload_and_fixed_depth_engine(tmp_path, policy, graphs, compact):
    check_loopformer_fp8(tmp_path / "export", policy, graphs, compact)
    (tmp_path / "export/tensors.pt").unlink()
