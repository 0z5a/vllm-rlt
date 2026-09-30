# SPDX-License-Identifier: Apache-2.0
"""Unit tests for AutoModelForCausalLM loading flow, approval logic, and dispatch."""

import json

import pytest
import torch
from safetensors.torch import save_file

from tests.helpers import tiny_nanbeige_config, tiny_ouro_config
from vllm_rlt.models import (
    AutoModelForCausalLM,
    NanbeigeForCausalLM,
    OuroForCausalLM,
)


def test_local_dispatch_ouro(tmp_path):
    config = tiny_ouro_config()
    (tmp_path / "config.json").write_text(json.dumps(config.to_dict()))
    model = OuroForCausalLM(config)
    save_file(model.state_dict(), tmp_path / "model.safetensors")

    loaded = AutoModelForCausalLM.from_pretrained(tmp_path, dtype=torch.float32)
    assert isinstance(loaded, OuroForCausalLM)


def test_local_dispatch_nanbeige(tmp_path):
    config = tiny_nanbeige_config()
    (tmp_path / "config.json").write_text(json.dumps(config.to_dict()))
    model = NanbeigeForCausalLM(config)
    save_file(model.state_dict(), tmp_path / "model.safetensors")

    loaded = AutoModelForCausalLM.from_pretrained(tmp_path, dtype=torch.float32)
    assert isinstance(loaded, NanbeigeForCausalLM)


def test_unsupported_model_type_raises_directly(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "llama"}))

    with pytest.raises(ValueError, match="Unsupported model_type 'llama'"):
        AutoModelForCausalLM.from_pretrained(tmp_path)


def test_remote_model_requires_approval_by_default():
    with pytest.raises(RuntimeError, match="Download requires approval"):
        AutoModelForCausalLM.from_pretrained("remote-org/unapproved-model")


def test_remote_approved_download_selects_by_config(tmp_path, monkeypatch):
    config = tiny_nanbeige_config()
    (tmp_path / "config.json").write_text(json.dumps(config.to_dict()))
    model = NanbeigeForCausalLM(config)
    save_file(model.state_dict(), tmp_path / "model.safetensors")

    # Mock hf_hub_download returning config from tmp_path
    def mock_hf_download(repo_id, filename, revision=None):
        return str(tmp_path / filename)

    # Mock snapshot_download returning the full dir
    def mock_snapshot(repo_id, revision=None, allow_patterns=None):
        return str(tmp_path)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", mock_hf_download)
    monkeypatch.setattr("huggingface_hub.snapshot_download", mock_snapshot)

    loaded = AutoModelForCausalLM.from_pretrained(
        "custom-org/custom-model", allow_download=True, dtype=torch.float32
    )
    assert isinstance(loaded, NanbeigeForCausalLM)


def test_remote_download_surfaces_errors_directly(monkeypatch):
    def mock_failing_download(repo_id, filename, revision=None):
        raise ConnectionError("Simulated network failure")

    monkeypatch.setattr("huggingface_hub.hf_hub_download", mock_failing_download)

    with pytest.raises(ConnectionError, match="Simulated network failure"):
        AutoModelForCausalLM.from_pretrained("some/remote-model", allow_download=True)


def test_remote_download_with_recipe_alias(tmp_path, monkeypatch):
    config = tiny_nanbeige_config()
    (tmp_path / "config.json").write_text(json.dumps(config.to_dict()))
    model = NanbeigeForCausalLM(config)
    save_file(model.state_dict(), tmp_path / "model.safetensors")

    downloaded_repo = []

    def mock_hf_download(repo_id, filename, revision=None):
        downloaded_repo.append((repo_id, revision))
        return str(tmp_path / filename)

    def mock_snapshot(repo_id, revision=None, allow_patterns=None):
        return str(tmp_path)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", mock_hf_download)
    monkeypatch.setattr("huggingface_hub.snapshot_download", mock_snapshot)
    monkeypatch.setattr("vllm_rlt.models.resolve_local_model_path", lambda *a, **k: None)

    loaded = AutoModelForCausalLM.from_pretrained(
        "nanbeige", allow_download=True, dtype=torch.float32
    )
    assert isinstance(loaded, NanbeigeForCausalLM)
    assert downloaded_repo[0][0] == "Nanbeige/Nanbeige4.2-3B"


def test_artifacts_basename_does_not_override_repository(tmp_path, monkeypatch):
    from vllm_rlt.models import resolve_local_model_path

    dummy = tmp_path / "artifacts/models/same-name"
    dummy.mkdir(parents=True)
    (dummy / "config.json").write_text("{}")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("huggingface_hub.try_to_load_from_cache", lambda *a, **k: None)
    assert resolve_local_model_path("other-org/same-name") is None
    assert resolve_local_model_path(dummy) == dummy
