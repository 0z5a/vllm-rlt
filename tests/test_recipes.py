# SPDX-License-Identifier: Apache-2.0
"""Unit tests for model recipes and clean caller-specified revision handling."""

import pytest

import vllm_rlt.models as models
import vllm_rlt.models.nanbeige as nanbeige_module
import vllm_rlt.models.ouro as ouro_module
from vllm_rlt.models import NanbeigeForCausalLM, OuroForCausalLM
from vllm_rlt.recipes import (
    NANBEIGE,
    NANBEIGE_MODEL_ID,
    NANBEIGE_REVISION,
    OURO,
    OURO_MODEL_ID,
    OURO_REVISION,
    ModelRecipe,
    get_recipe,
    list_recipes,
    nanbeige4_2_3b,
    ouro_1_4b,
)


def test_recipe_attributes_and_lookup():
    assert isinstance(ouro_1_4b, ModelRecipe)
    assert isinstance(nanbeige4_2_3b, ModelRecipe)
    assert ouro_1_4b is OURO
    assert nanbeige4_2_3b is NANBEIGE

    assert OURO_MODEL_ID == "ByteDance/Ouro-1.4B"
    assert OURO_REVISION == "574fa66cb8bf5abdc979642d01cf2b79b16bfab1"
    assert ouro_1_4b.model_id == OURO_MODEL_ID
    assert ouro_1_4b.revision == OURO_REVISION
    assert ouro_1_4b.served_model_name == "ouro"

    assert NANBEIGE_MODEL_ID == "Nanbeige/Nanbeige4.2-3B"
    assert NANBEIGE_REVISION is None
    assert nanbeige4_2_3b.model_id == NANBEIGE_MODEL_ID
    assert nanbeige4_2_3b.revision is NANBEIGE_REVISION
    assert nanbeige4_2_3b.served_model_name == "nanbeige"

    assert get_recipe("ouro") is ouro_1_4b
    assert get_recipe("ByteDance/Ouro-1.4B") is ouro_1_4b
    assert get_recipe("nanbeige") is nanbeige4_2_3b
    assert get_recipe("Nanbeige/Nanbeige4.2-3B") is nanbeige4_2_3b
    assert get_recipe("unknown_model") is None
    assert list_recipes() == ["ouro", "nanbeige"]


def test_model_modules_do_not_contain_checkpoint_constants():
    assert not hasattr(ouro_module, "OURO_MODEL_ID")
    assert not hasattr(ouro_module, "OURO_REVISION")
    assert not hasattr(nanbeige_module, "NANBEIGE_MODEL_ID")
    assert not hasattr(nanbeige_module, "NANBEIGE_REVISION")
    assert "OURO_MODEL_ID" not in models.__all__
    assert "OURO_REVISION" not in models.__all__
    assert "NANBEIGE_MODEL_ID" not in models.__all__


def test_from_pretrained_requires_path_or_repo():
    with pytest.raises(TypeError):
        OuroForCausalLM.from_pretrained()  # pyright: ignore[reportCallIssue]

    with pytest.raises(TypeError):
        NanbeigeForCausalLM.from_pretrained()  # pyright: ignore[reportCallIssue]
