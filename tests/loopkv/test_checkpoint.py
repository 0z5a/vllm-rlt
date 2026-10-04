import pytest
import torch
from safetensors.torch import save_file

from experiments.loopkv.checkpoint import load_weights
from tests.helpers import tiny_ouro_config
from vllm_rlt.models import OuroForCausalLM


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_standalone_reader_matches_native_parameter_bits(tmp_path, dtype):
    torch.manual_seed(99)
    reference = OuroForCausalLM(tiny_ouro_config()).to(dtype=dtype)
    weights = {name: p.detach().contiguous() for name, p in reference.named_parameters()}
    save_file(weights, tmp_path / "model.safetensors")
    with torch.device("meta"):
        loaded = OuroForCausalLM(tiny_ouro_config())
    load_weights(loaded, tmp_path, torch.device("cpu"), dtype)
    for name, parameter in loaded.named_parameters():
        assert torch.equal(parameter, weights[name])
        assert parameter.data_ptr() != weights[name].data_ptr()
