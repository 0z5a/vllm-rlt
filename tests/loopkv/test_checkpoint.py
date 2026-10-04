import json

import pytest
import torch
from safetensors.torch import save_file

from experiments.loopkv.checkpoint import load_model, load_weights
from tests.helpers import tiny_nanbeige_config, tiny_ouro_config
from vllm_rlt.models import NanbeigeForCausalLM, OuroForCausalLM


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


@pytest.mark.parametrize("model_type", ["ouro", "nanbeige"])
@pytest.mark.parametrize("sharded", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_standalone_model_matches_native_loader(tmp_path, model_type, sharded, dtype):
    torch.manual_seed(23)
    if model_type == "ouro":
        reference = OuroForCausalLM(tiny_ouro_config()).to(dtype=dtype)
    else:
        reference = NanbeigeForCausalLM(tiny_nanbeige_config()).to(dtype=dtype)
    (tmp_path / "config.json").write_text(json.dumps(reference.config.to_dict()))
    weights = {name: p.detach().contiguous() for name, p in reference.named_parameters()}
    if sharded:
        names = list(weights)
        parts = [names[::2], names[1::2]]
        weight_map = {}
        for i, part in enumerate(parts):
            name = f"model-{i + 1:05d}-of-00002.safetensors"
            save_file({key: weights[key] for key in part}, tmp_path / name)
            weight_map.update({key: name for key in part})
        (tmp_path / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": weight_map})
        )
    else:
        save_file(weights, tmp_path / "model.safetensors")
    native = type(reference).from_pretrained(tmp_path, dtype=dtype)
    loaded = load_model(tmp_path, torch.device("cpu"), dtype)
    assert type(loaded) is type(native)
    for name, value in loaded.state_dict().items():
        assert torch.equal(value, native.state_dict()[name])
    assert torch.equal(loaded.model.rotary_emb.inv_freq, native.model.rotary_emb.inv_freq)
    assert not loaded.training
