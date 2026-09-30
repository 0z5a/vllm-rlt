# SPDX-License-Identifier: Apache-2.0
"""Model exports and unified AutoModelForCausalLM factory."""

import json
from pathlib import Path
from typing import Any

import torch

from vllm_rlt.recipes import get_recipe

from .nanbeige import NanbeigeConfig, NanbeigeForCausalLM
from .ouro import OuroConfig, OuroForCausalLM

MODEL_MAPPING = {
    "ouro": OuroForCausalLM,
    "nanbeige": NanbeigeForCausalLM,
}


def resolve_model_source(path_or_repo: str | Path) -> str:
    """Expand explicit local paths or recipe aliases without changing revision."""
    folder = Path(path_or_repo).expanduser()
    if folder.is_dir():
        return str(folder.resolve())
    raw = str(path_or_repo)
    recipe = get_recipe(raw)
    # A full repository ID is already an explicit source.
    return recipe.model_id if "/" not in raw and recipe is not None else raw


def resolve_local_model_path(path_or_repo: str | Path, revision=None) -> Path | None:
    """Find an explicit directory or a complete, revision-specific HF snapshot."""
    from huggingface_hub import try_to_load_from_cache

    source = resolve_model_source(path_or_repo)
    folder = Path(source)
    if folder.is_dir():
        return folder
    cached = try_to_load_from_cache(source, "config.json", revision=revision)
    if not isinstance(cached, str):
        return None
    folder = Path(cached).parent
    index = folder / "model.safetensors.index.json"
    if index.is_file():
        names = set(json.loads(index.read_text())["weight_map"].values())
        if names and all((folder / name).is_file() for name in names):
            return folder
    elif (folder / "model.safetensors").is_file():
        return folder
    return None


def resolve_model_config(path_or_repo, *, revision=None, allow_download=False):
    """Resolve an approved source and inspect its config before loading weights."""
    source = resolve_model_source(path_or_repo)
    local = resolve_local_model_path(source, revision=revision)
    if local is not None:
        source = str(local)
        config_path = local / "config.json"
    else:
        if not allow_download:
            raise RuntimeError(
                f"Model '{path_or_repo}' is not available locally. Download requires approval: "
                "pass allow_download=True or confirm at the interactive entrypoint."
            )
        from huggingface_hub import hf_hub_download

        config_path = Path(hf_hub_download(source, "config.json", revision=revision))
    config = json.loads(config_path.read_text())
    model_type = config.get("model_type", "").lower()
    if model_type not in MODEL_MAPPING:
        raise ValueError(f"Unsupported model_type {model_type!r} for {path_or_repo}")
    # Pin an approved remote load to the snapshot whose configuration was read.
    if config_path.parent.parent.name == "snapshots":
        revision = config_path.parent.name
    return source, revision, config


class AutoModelForCausalLM:
    """Factory for loading recurrent causal language models by configuration inspection."""

    @classmethod
    def _dispatch_from_config(
        cls,
        config_data: dict[str, Any],
        path_or_repo: str | Path,
        *,
        revision: str | None = None,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.bfloat16,
    ):
        model_type = config_data.get("model_type", "").lower()
        model_cls = MODEL_MAPPING.get(model_type)
        if model_cls is None:
            supported = sorted(MODEL_MAPPING.keys())
            raise ValueError(
                f"Unsupported model_type {model_type!r} for {path_or_repo}. "
                f"Supported types: {supported}"
            )
        return model_cls.from_pretrained(
            path_or_repo, revision=revision, device=device, dtype=dtype
        )

    @classmethod
    def from_pretrained(
        cls,
        path_or_repo: str | Path,
        *,
        revision: str | None = None,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.bfloat16,
        allow_download: bool = False,
    ):
        source, resolved_revision, config = resolve_model_config(
            path_or_repo, revision=revision, allow_download=allow_download
        )
        return cls._dispatch_from_config(
            config, source, revision=resolved_revision, device=device, dtype=dtype
        )


__all__ = [
    "OuroConfig",
    "OuroForCausalLM",
    "NanbeigeConfig",
    "NanbeigeForCausalLM",
    "AutoModelForCausalLM",
    "resolve_local_model_path",
    "resolve_model_source",
    "resolve_model_config",
]
