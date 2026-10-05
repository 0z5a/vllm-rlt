import pytest


@pytest.mark.gpu
@pytest.mark.parametrize("rows", [1, 17, 128, 129])
@pytest.mark.parametrize("columns,outputs", [(64, 32), (2048, 2048), (2048, 5632), (5632, 2048)])
@pytest.mark.parametrize("norm", [False, True])
@pytest.mark.parametrize("fixed", [False, True])
def test_fused_nvfp4_bytes_native_gemm_and_graph_replay(rows, columns, outputs, norm, fixed):
    from tests.loopquant_nvfp4_checks import check_fused_nvfp4

    check_fused_nvfp4(rows, columns, outputs, norm, fixed)
