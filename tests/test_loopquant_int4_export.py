import json

import pytest
import torch

from loopquant.int4_export import core_projections, export_int4_model
from loopquant.quantizers import int4_pack, int4_unpack
from tests.loopquant_int4_checks import int4_reference_model


@pytest.mark.parametrize("family", ["ouro", "nanbeige", "hrm_text", "loopformer"])
def test_portable_int4_artifact_preserves_protected_weights_and_core_codes(tmp_path, family):
    torch.manual_seed(43)
    model = int4_reference_model(family).bfloat16()
    folder = tmp_path / "rtn"
    export_int4_model(model, folder, model_revision="a" * 40)
    manifest = json.loads((folder / "manifest.json").read_text())
    tensors = torch.load(folder / "tensors.pt", weights_only=True)
    assert manifest["packed_weight_matrices"] == len(core_projections(model))
    expected_count = {"ouro": 14, "nanbeige": 14, "hrm_text": 16, "loopformer": 8}[family]
    assert len(tensors["codes"]) == expected_count
    if family == "loopformer":
        assert all("adaLN_modulation" not in name for name in tensors["codes"])
        assert "gpt.transformer.wte.weight" in tensors["protected"]
        assert "gpt.time_embedder.mlp.0.weight" in tensors["protected"]
    if family == "hrm_text":
        assert sum(name.startswith("model.L_module.") for name in tensors["codes"]) == 8
        assert "model.z_L_init" in tensors["protected"]
    assert manifest["group_scale_dtype"] == "bfloat16"
    assert manifest["native_quality"] == "not_run"
    for name, codes in tensors["codes"].items():
        layer = model.get_submodule(name)
        expected, scales = int4_pack(layer.weight)
        assert torch.equal(codes, expected)
        assert torch.equal(tensors["scales"][name], scales.bfloat16())
        decoded = int4_unpack(codes, scales.bfloat16().float(), layer.in_features)
        assert decoded.shape == layer.weight.shape and torch.isfinite(decoded).all()
    assert all(
        torch.equal(model.state_dict()[name], value) for name, value in tensors["protected"].items()
    )
    calibrated = {
        name: (value, tensors["scales"][name]) for name, value in tensors["codes"].items()
    }
    export_int4_model(
        model, tmp_path / "supplied", model_revision="a" * 40, method="GPTQ-all", packed=calibrated
    )
    supplied = torch.load(tmp_path / "supplied/tensors.pt", weights_only=True)
    assert all(
        torch.equal(supplied["codes"][name], codes) for name, codes in tensors["codes"].items()
    )


@pytest.mark.gpu
@pytest.mark.parametrize("family", ["ouro", "nanbeige", "hrm_text", "loopformer"])
@pytest.mark.parametrize("graphs", [False, True])
@pytest.mark.parametrize("compact", [False, True])
def test_int4_artifact_reloads_and_runs_mixed_depth_native_engine(
    tmp_path, family, graphs, compact
):
    from tests.loopquant_int4_checks import check_int4_native_model

    check_int4_native_model(family, graphs, tmp_path / "export", compact=compact)
