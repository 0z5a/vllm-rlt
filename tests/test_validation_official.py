"""CPU compatibility checks; no checkpoint downloads or device discovery."""

import hashlib
import importlib.metadata
import importlib.util
import socket
from pathlib import Path

import pytest
import torch

from vllm_lt.models.config import OuroConfig
from vllm_lt.models.ouro import OuroForCausalLM
from vllm_lt.validation import official


def _compatible_environment():
    try:
        version = importlib.metadata.version("transformers")
    except importlib.metadata.PackageNotFoundError:
        return False
    return version == "4.55.0" and importlib.util.find_spec("kernels") is None


requires_official = pytest.mark.skipif(
    not _compatible_environment(), reason="requires prepared Transformers 4.55.0 reference env"
)


@pytest.fixture
def tiny_model():
    torch.manual_seed(7)
    return OuroForCausalLM(
        OuroConfig.tiny(hidden_size=16, intermediate_size=32, num_attention_heads=2)
    )


def test_vendored_source_identity():
    source = Path(official.__file__).with_name("reference_code")
    for name, expected in official.OFFICIAL_SOURCE_SHA256.items():
        assert hashlib.sha256((source / name).read_bytes()).hexdigest() == expected
    assert official.official_provenance()["revision"] == official.OFFICIAL_REVISION


def test_rejects_unreviewed_dependency_before_model_construction(monkeypatch):
    metadata = official.official_provenance()
    metadata["dependencies"]["transformers"] = "5.14.1"
    monkeypatch.setattr(official, "official_provenance", lambda: metadata)
    with pytest.raises(RuntimeError, match="requires transformers==4.55.0"):
        official.OfficialOuroReference({}, {})


def test_rejects_optional_kernel_integration_before_import(monkeypatch):
    metadata = official.official_provenance()
    metadata["dependencies"]["transformers"] = "4.55.0"
    metadata["optional_kernels_present"] = True
    monkeypatch.setattr(official, "official_provenance", lambda: metadata)
    with pytest.raises(RuntimeError, match="'kernels' to be absent"):
        official.OfficialOuroReference({}, {})


def test_source_tampering_rejected_before_import(tmp_path, monkeypatch):
    for name in official.OFFICIAL_SOURCE_SHA256:
        (tmp_path / name).write_text("raise AssertionError('must never execute')\n")
    monkeypatch.setattr(official, "_SOURCE_DIR", tmp_path)
    with pytest.raises(RuntimeError, match="source hash mismatch"):
        official.OfficialOuroReference({}, {})


@requires_official
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_storage_sharing_dtype_and_network_free_forward(tiny_model, monkeypatch, dtype):
    def forbidden(*args, **kwargs):
        raise AssertionError("network or native inference helper was called")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    for name in ("is_available", "device_count", "current_device", "_lazy_init"):
        monkeypatch.setattr(torch.cuda, name, forbidden)
    for name in ("prelude", "recurrent", "coda"):
        monkeypatch.setattr(OuroForCausalLM, name, forbidden)
    tiny_model.to(dtype=dtype)
    tiny_model.lm_head.weight.requires_grad_(True)
    weights = dict(tiny_model.named_parameters())
    before = {name: value.detach().clone() for name, value in weights.items()}
    flags = {name: value.requires_grad for name, value in weights.items()}
    reference = official.OfficialOuroReference(tiny_model.config.to_dict(), weights)
    assert all(
        p.data_ptr() == weights[name].data_ptr() for name, p in reference.model.named_parameters()
    )
    assert all(not p.requires_grad for p in reference.model.parameters())
    rotary = reference.model.model.rotary_emb
    expected = 1 / (tiny_model.config.rope_theta ** (torch.arange(0, 8, 2).float() / 8))
    assert rotary.inv_freq.dtype == torch.float32
    torch.testing.assert_close(rotary.inv_freq, expected, rtol=0, atol=0)
    assert rotary.original_inv_freq is rotary.inv_freq
    predictions = reference.predict([1, 2, 3, 4], [5, 6, 7, 8, 9, 10, 11, 12])
    assert predictions.shape == (9, tiny_model.config.vocab_size)
    assert predictions.dtype == dtype and not predictions.requires_grad
    assert torch.isfinite(predictions).all()
    assert predictions._base is None
    reference.close()
    for name, value in weights.items():
        assert torch.equal(value, before[name])
        assert value.requires_grad == flags[name]
    with pytest.raises(RuntimeError, match="closed"):
        reference.predict([1], [])


@requires_official
def test_rejects_replaced_rmsnorm_before_forward(tiny_model, monkeypatch):
    reference = official.OfficialOuroReference(tiny_model.config.to_dict(), tiny_model.state_dict())
    monkeypatch.setattr(reference.model.model.norm, "forward", lambda value: value)
    with pytest.raises(RuntimeError, match="RMSNorm forward has been replaced"):
        reference.predict([1], [])


@requires_official
def test_fourth_loop_selected_and_nine_positions_aligned(tiny_model):
    config = tiny_model.config.to_dict()
    config["early_exit_threshold"] = 0.0
    reference = official.OfficialOuroReference(config, tiny_model.state_dict())
    captured, layer_calls = [], []
    handle = reference.model.model.register_forward_hook(
        lambda module, args, output: captured.append(output)
    )
    layer_handle = reference.model.model.layers[0].register_forward_hook(
        lambda *args: layer_calls.append(1)
    )
    actual = reference.predict([1, 2, 3], [4, 5, 6, 7, 8, 9, 10, 11])
    handle.remove()
    layer_handle.remove()
    output, hidden, gates = captured[0]
    assert output.past_key_values is None
    assert len(hidden) == len(gates) == len(layer_calls) == 4
    with torch.inference_mode():
        expected = reference.model.lm_head(hidden[3])[0, 2:]
        first_loop = reference.model.lm_head(hidden[0])[0, 2:]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert not torch.allclose(actual, first_loop)


@requires_official
def test_full_sequence_predictions_match_causal_prefixes(tiny_model):
    reference = official.OfficialOuroReference(tiny_model.config.to_dict(), tiny_model.state_dict())
    prompt, continuation = [1, 2, 3], [4, 5, 6, 7, 8, 9, 10, 11]
    full = reference.predict(prompt, continuation)
    for index in range(9):
        prefix = reference.predict(prompt + continuation[:index], [])
        torch.testing.assert_close(full[index], prefix[0], atol=1e-6, rtol=1e-5)


@requires_official
@pytest.mark.parametrize("prompt,continuation", [([], []), ([True], []), ([64], []), ([1], [-1])])
def test_invalid_teacher_forced_inputs(tiny_model, prompt, continuation):
    reference = official.OfficialOuroReference(tiny_model.config.to_dict(), tiny_model.state_dict())
    with pytest.raises(ValueError):
        reference.predict(prompt, continuation)


@requires_official
def test_strict_parameter_mapping_and_four_loop_contract(tiny_model):
    config, weights = tiny_model.config.to_dict(), tiny_model.state_dict()
    config["total_ut_steps"] = 3
    with pytest.raises(ValueError, match="exactly four"):
        official.OfficialOuroReference(config, weights)
    config["total_ut_steps"] = 4
    weights.pop("lm_head.weight")
    with pytest.raises(RuntimeError, match="Missing key"):
        official.OfficialOuroReference(config, weights)
