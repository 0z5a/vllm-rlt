"""Storage accounting and native CUDA serving qualification."""

import pytest

from loopquant.serving import FP8Scratch
from tests.helpers import tiny_ouro_config
from tests.loopquant_serving_checks import check_fused_native_model
from vllm_rlt.models.ouro import OuroForCausalLM


def test_scratch_buckets_share_storage_and_memory_grows_linearly():
    model = OuroForCausalLM(tiny_ouro_config(hidden_size=64, intermediate_size=128))
    scratch = FP8Scratch(model, 512)
    assert scratch.nbytes == 512 * (64 + 128 + 2 * 4)
    for columns in [64, 128]:
        addresses = {
            scratch.buffers[rows, columns].packed.data_ptr() for rows in range(16, 513, 16)
        }
        assert len(addresses) == 1


@pytest.mark.gpu
@pytest.mark.parametrize("policy", ["DYN", "SH1", "ST2"])
@pytest.mark.parametrize("graphs", [False, True])
@pytest.mark.parametrize("fuse_norm", [False, True])
def test_fused_native_matches_quantized_reference(policy, graphs, fuse_norm):
    check_fused_native_model(policy, graphs, fuse_norm)
