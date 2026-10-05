import pytest

from loopquant.serving import FP8Scratch
from tests.helpers import tiny_nanbeige_config
from tests.loopquant_serving_checks import check_fused_native_model
from vllm_rlt.models.nanbeige import NanbeigeForCausalLM


def test_scratch_includes_attention_output_width_distinct_from_hidden_and_mlp():
    model = NanbeigeForCausalLM(
        tiny_nanbeige_config(hidden_size=64, head_dim=32, intermediate_size=256)
    )
    scratch = FP8Scratch(model, 17)
    assert scratch.storage.keys() == {64, 128, 256}
    assert all(buffer.packed.shape == (32, width) for width, buffer in scratch.storage.items())


@pytest.mark.gpu
@pytest.mark.parametrize("policy", ["DYN", "SH1", "ST2", "LOOP"])
@pytest.mark.parametrize("graphs", [False, True])
@pytest.mark.parametrize("fuse_norm", [False, True])
@pytest.mark.parametrize("skip_loop_final_norm", [False, True])
def test_nanbeige_native_export_preserves_fixed_loop_readout(
    policy, graphs, fuse_norm, skip_loop_final_norm, tmp_path
):
    check_fused_native_model(
        policy,
        graphs,
        fuse_norm,
        export_folder=tmp_path / "export",
        family="nanbeige",
        skip_loop_final_norm=skip_loop_final_norm,
    )
