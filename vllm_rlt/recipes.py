# SPDX-License-Identifier: Apache-2.0
"""Model recipes providing canonical checkpoint IDs, pinned revisions, and metadata."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelRecipe:
    """Canonical metadata and pinned provenance for supported model checkpoints."""

    model_id: str
    revision: str | None = None
    served_model_name: str | None = None
    description: str = ""


# Ouro
OURO_MODEL_ID = "ByteDance/Ouro-1.4B"
OURO_REVISION = "574fa66cb8bf5abdc979642d01cf2b79b16bfab1"
ouro_1_4b = ModelRecipe(
    model_id=OURO_MODEL_ID,
    revision=OURO_REVISION,
    served_model_name="ouro",
    description="Official ByteDance/Ouro-1.4B recurrent causal language model.",
)
OURO = ouro_1_4b

# Nanbeige
NANBEIGE_MODEL_ID = "Nanbeige/Nanbeige4.2-3B"
NANBEIGE_REVISION = None
nanbeige4_2_3b = ModelRecipe(
    model_id=NANBEIGE_MODEL_ID,
    revision=NANBEIGE_REVISION,
    served_model_name="nanbeige",
    description="Official Nanbeige/Nanbeige4.2-3B recurrent causal language model.",
)
NANBEIGE = nanbeige4_2_3b

# Registry & Lookup
RECIPES: dict[str, ModelRecipe] = {
    "ouro": ouro_1_4b,
    "ouro-1.4b": ouro_1_4b,
    "bytedance/ouro-1.4b": ouro_1_4b,
    "nanbeige": nanbeige4_2_3b,
    "nanbeige-3b": nanbeige4_2_3b,
    "nanbeige4.2-3b": nanbeige4_2_3b,
    "nanbeige/nanbeige4.2-3b": nanbeige4_2_3b,
}


def get_recipe(name: str) -> ModelRecipe | None:
    """Lookup a known model recipe by repository name or alias."""
    return RECIPES.get(name.lower())


def list_recipes() -> list[str]:
    """Return canonical names of supported recipes."""
    return ["ouro", "nanbeige"]


__all__ = [
    "ModelRecipe",
    "OURO",
    "OURO_MODEL_ID",
    "OURO_REVISION",
    "ouro_1_4b",
    "NANBEIGE",
    "NANBEIGE_MODEL_ID",
    "NANBEIGE_REVISION",
    "nanbeige4_2_3b",
    "RECIPES",
    "get_recipe",
    "list_recipes",
]
