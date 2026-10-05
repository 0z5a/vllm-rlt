import json

import pytest
import torch

from loopquant.huginn_int4 import core_projections, export_huginn_int4
from loopquant.quantizers import int4_pack
from tests.loopquant_huginn_int4_checks import check_huginn_int4, tiny_int4_huginn


@pytest.mark.parametrize("method", ["RTN", "GPTQ-first", "GPTQ-all", "GPTQ-matched"])
def test_huginn_int4_portable_codes_and_protected_boundaries(tmp_path, method):
    torch.manual_seed(43)
    model = tiny_int4_huginn()
    projections = core_projections(model)
    packed = {name: int4_pack(layer.weight) for name, layer in projections.items()}
    export_huginn_int4(
        model,
        tmp_path / "export",
        model_revision="a" * 40,
        method=method,
        packed=None if method == "RTN" else packed,
    )
    manifest = json.loads((tmp_path / "export/manifest.json").read_text())
    payload = torch.load(tmp_path / "export/tensors.pt", weights_only=True)
    assert manifest["packed_weight_matrices"] == 8
    assert manifest["protected_aliases"] == {"lm_head.weight": "transformer.wte.weight"}
    assert "lm_head.weight" not in payload["protected"]
    assert "transformer.adapter.weight" in payload["protected"]
    assert payload["protected"]["freqs_cis"].dtype == torch.float32
    for name, (codes, scales) in packed.items():
        assert torch.equal(payload["codes"][name], codes)
        assert torch.equal(payload["scales"][name], scales.bfloat16())
    assert all(
        torch.equal(value, model.state_dict()[name]) for name, value in payload["protected"].items()
    )
    assert model.lm_head.weight is model.transformer.wte.weight
    (tmp_path / "export/tensors.pt").unlink()


def test_huginn_int4_rejects_rounded_rotary_before_creating_artifact(tmp_path):
    model = tiny_int4_huginn()
    model.freqs_cis = model.freqs_cis.bfloat16()
    with pytest.raises(ValueError, match="freqs_cis"):
        export_huginn_int4(model, tmp_path / "export", model_revision="a" * 40)
    assert not (tmp_path / "export").exists()


@pytest.mark.gpu
@pytest.mark.parametrize("graphs", [False, True])
def test_huginn_int4_native_roundtrip_seeded_quality_and_engine(tmp_path, graphs):
    check_huginn_int4(tmp_path / "export", graphs)
