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


def resolve_local_model_path(path_or_repo: str | Path) -> Path | None:
    """Find a locally available checkpoint directory containing config.json."""
    folder = Path(path_or_repo).expanduser()
    if folder.is_dir() and (folder / "config.json").is_file():
        return folder

    raw = str(path_or_repo)
    candidates = [
        Path("artifacts/models") / raw,
        Path("artifacts/models") / raw.split("/")[-1],
    ]
    recipe = get_recipe(raw)
    if recipe is not None:
        candidates.append(Path("artifacts/models") / recipe.model_id.split("/")[-1])
        candidates.append(Path("artifacts/models") / recipe.model_id)

    for cand in candidates:
        if cand.is_dir() and (cand / "config.json").is_file():
            return cand
    return None


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
        local_folder = resolve_local_model_path(path_or_repo)

        # 1. Load locally when available (never access network if config.json exists locally)
        if local_folder is not None:
            config_data = json.loads((local_folder / "config.json").read_text())
            return cls._dispatch_from_config(
                config_data, local_folder, revision=revision, device=device, dtype=dtype
            )

        # 2. Check recipe aliases if applicable
        recipe = get_recipe(str(path_or_repo))
        repo_id = recipe.model_id if recipe is not None else str(path_or_repo)
        if revision is None and recipe is not None:
            revision = recipe.revision

        # 3. Remote model requires explicit download approval
        if not allow_download:
            raise RuntimeError(
                f"Model '{path_or_repo}' is not available locally. Download requires approval: "
                f"pass allow_download=True or confirm at the interactive entrypoint."
            )

        # 4. Download config.json from Hub; surface errors directly without guessing or falling back
        from huggingface_hub import hf_hub_download

        config_path = hf_hub_download(
            repo_id=repo_id,
            filename="config.json",
            revision=revision,
        )
        config_data = json.loads(Path(config_path).read_text())

        # 5. Select implementation strictly based on config.json
        return cls._dispatch_from_config(
            config_data, repo_id, revision=revision, device=device, dtype=dtype
        )


__all__ = [
    "OuroConfig",
    "OuroForCausalLM",
    "NanbeigeConfig",
    "NanbeigeForCausalLM",
    "AutoModelForCausalLM",
    "resolve_local_model_path",
]
