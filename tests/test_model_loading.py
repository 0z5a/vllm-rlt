"""Regressions for caller identity and download authorization after PR #63."""

import argparse
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from safetensors.torch import save_file

from tests.helpers import tiny_nanbeige_config
from vllm_rlt.models import AutoModelForCausalLM, NanbeigeForCausalLM, resolve_local_model_path


@pytest.fixture
def checkpoint(tmp_path):
    model = NanbeigeForCausalLM(tiny_nanbeige_config())
    (tmp_path / "config.json").write_text(json.dumps(model.config.to_dict()))
    save_file(model.state_dict(), tmp_path / "model.safetensors")
    return tmp_path


@pytest.mark.parametrize("revision", [None, "my-tag"])
@pytest.mark.parametrize("source", ["custom-org/model", "ouro", "nanbeige"])
def test_caller_revision_reaches_hub_unchanged(monkeypatch, checkpoint, revision, source):
    fetch = Mock(return_value=str(checkpoint / "config.json"))
    monkeypatch.setattr("huggingface_hub.try_to_load_from_cache", lambda *a, **k: None)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", fetch)
    load = Mock(return_value=object())
    monkeypatch.setattr(NanbeigeForCausalLM, "from_pretrained", load)
    AutoModelForCausalLM.from_pretrained(source, revision=revision, allow_download=True)
    assert fetch.call_args.args == (source, "config.json")
    assert fetch.call_args.kwargs["revision"] == revision
    assert load.call_args.kwargs["revision"] == revision


def test_revision_specific_cached_checkpoint_needs_no_network(monkeypatch, checkpoint):
    lookup = Mock(return_value=str(checkpoint / "config.json"))
    monkeypatch.setattr("huggingface_hub.try_to_load_from_cache", lookup)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", Mock(side_effect=AssertionError))
    monkeypatch.setattr("huggingface_hub.snapshot_download", Mock(side_effect=AssertionError))
    model = AutoModelForCausalLM.from_pretrained("my-org/model", revision="pinned")
    assert isinstance(model, NanbeigeForCausalLM)
    assert lookup.call_args.kwargs["revision"] == "pinned"
    assert lookup.call_args.args[0] == "my-org/model"


def test_partial_cache_does_not_authorize_download(monkeypatch, tmp_path):
    (tmp_path / "config.json").write_text('{"model_type": "nanbeige"}')
    monkeypatch.setattr(
        "huggingface_hub.try_to_load_from_cache", lambda *a, **k: str(tmp_path / "config.json")
    )
    assert resolve_local_model_path("my-org/model") is None
    with pytest.raises(RuntimeError, match="Download requires approval"):
        AutoModelForCausalLM.from_pretrained("my-org/model")


def test_text_generation_uses_caller_source_and_revision(monkeypatch, checkpoint):
    from vllm_rlt import LLM, SamplingParams

    monkeypatch.setattr("huggingface_hub.try_to_load_from_cache", lambda *a, **k: None)
    monkeypatch.setattr(
        "huggingface_hub.hf_hub_download", lambda *a, **k: str(checkpoint / "config.json")
    )
    monkeypatch.setattr("huggingface_hub.snapshot_download", lambda **k: str(checkpoint))
    tokenizer = SimpleNamespace(encode=lambda text: [2, 3], decode=lambda ids, **kw: "text")
    token_load = Mock(return_value=tokenizer)
    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", token_load)
    llm = LLM("custom-org/checkpoint", revision="custom", allow_download=True, dtype=torch.float32)
    try:
        outputs = llm.generate("hello", SamplingParams(max_tokens=1, ignore_eos=True))
        assert outputs[0].text == "text"
        assert token_load.call_args.args == ("custom-org/checkpoint",)
        assert token_load.call_args.kwargs["revision"] == "custom"
    finally:
        llm.close()


@pytest.mark.parametrize("answer", ["y", "yes", " YES ", "n"])
def test_interactive_approval(monkeypatch, answer):
    from vllm_rlt.entrypoints.model_loading import approve_download

    monkeypatch.setattr(
        "vllm_rlt.entrypoints.model_loading.resolve_local_model_path", lambda *a, **k: None
    )
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda prompt: answer)
    args = SimpleNamespace(model="org/model", revision=None, assume_yes=False)
    if answer == "n":
        with pytest.raises(SystemExit):
            approve_download(args, argparse.ArgumentParser())
    else:
        assert approve_download(args, argparse.ArgumentParser())


def test_noninteractive_requires_explicit_approval(monkeypatch):
    from vllm_rlt.entrypoints.model_loading import approve_download

    monkeypatch.setattr(
        "vllm_rlt.entrypoints.model_loading.resolve_local_model_path", lambda *a, **k: None
    )
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    args = SimpleNamespace(model="org/model", revision=None, assume_yes=False)
    with pytest.raises(SystemExit):
        approve_download(args, argparse.ArgumentParser())
    args.assume_yes = True
    assert approve_download(args, argparse.ArgumentParser())


def test_pd_rejects_unapproved_download_before_spawning(monkeypatch):
    from vllm_rlt.pd.engine import PDEngine

    monkeypatch.setattr("huggingface_hub.try_to_load_from_cache", lambda *a, **k: None)
    spawn = Mock(side_effect=AssertionError("must reject before spawning"))
    monkeypatch.setattr("vllm_rlt.pd.engine.mp.get_context", spawn)
    with pytest.raises(RuntimeError, match="Download requires approval"):
        PDEngine("my-org/model")
    spawn.assert_not_called()


def test_serving_custom_repository_dispatches_from_config(monkeypatch, checkpoint):
    from tokenizers.decoders import WordPiece

    from vllm_rlt.entrypoints.serve import load_engine

    monkeypatch.setattr("huggingface_hub.try_to_load_from_cache", lambda *a, **k: None)
    monkeypatch.setattr(
        "huggingface_hub.hf_hub_download", lambda *a, **k: str(checkpoint / "config.json")
    )
    monkeypatch.setattr("huggingface_hub.snapshot_download", lambda **k: str(checkpoint))
    tokenizer = SimpleNamespace(backend_tokenizer=SimpleNamespace(decoder=WordPiece()))
    token_load = Mock(return_value=tokenizer)
    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", token_load)
    args = SimpleNamespace(
        model="custom-org/custom-name",
        revision="tag",
        allow_download=True,
        tokenizer=None,
        tokenizer_revision=None,
        device="cpu",
        dtype="float32",
        num_blocks=16,
        block_size=2,
        max_num_seqs=1,
        max_num_batched_tokens=3,
        mode="refill",
        attention_backend="torch",
    )
    engine, actual = load_engine(args)
    try:
        assert isinstance(engine.model, NanbeigeForCausalLM)
        assert actual is tokenizer
        assert token_load.call_args.args == (args.model,)
        assert token_load.call_args.kwargs["revision"] == "tag"
    finally:
        engine.close()


@pytest.mark.parametrize("approved", [False, True])
def test_pd_worker_preserves_download_authorization(monkeypatch, approved):
    from vllm_rlt.pd.worker import PDWorker

    monkeypatch.setattr(torch.cuda, "set_device", lambda device: None)
    load = Mock(side_effect=RuntimeError("stop before GPU allocation"))
    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", load)
    with pytest.raises(RuntimeError, match="stop before GPU allocation"):
        PDWorker(
            "prefill",
            0,
            "worker",
            None,
            "org/model",
            {"seed": 0, "dtype": "bfloat16", "revision": "tag", "allow_download": approved},
            None,
        )
    assert load.call_args.kwargs["allow_download"] is approved


def test_resolved_hub_snapshot_freezes_model_and_tokenizer_revision(monkeypatch, tmp_path):
    from vllm_rlt.models import resolve_model_config

    config = tmp_path / "snapshots" / ("a" * 40) / "config.json"
    config.parent.mkdir(parents=True)
    config.write_text('{"model_type": "nanbeige"}')
    monkeypatch.setattr("huggingface_hub.try_to_load_from_cache", lambda *a, **k: None)
    fetch = Mock(return_value=str(config))
    monkeypatch.setattr("huggingface_hub.hf_hub_download", fetch)
    source, revision, _ = resolve_model_config("custom-org/checkpoint", allow_download=True)
    assert source == "custom-org/checkpoint"
    assert fetch.call_args.kwargs["revision"] is None
    assert revision == "a" * 40


def test_model_loaders_require_explicit_source():
    from vllm_rlt.models import OuroForCausalLM

    for loader in (AutoModelForCausalLM, OuroForCausalLM, NanbeigeForCausalLM):
        with pytest.raises(TypeError):
            loader.from_pretrained()


def test_cli_requires_source_for_real_model(monkeypatch):
    from vllm_rlt.entrypoints.cli import main

    monkeypatch.setattr("sys.argv", ["vllm-rlt", "--prompt", "hello"])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
