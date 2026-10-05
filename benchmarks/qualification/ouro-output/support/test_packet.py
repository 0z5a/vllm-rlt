"""Qualify the copied input reader and finite scheduler harness on the frozen runtime."""

import json

import pytest
import torch
from native_checkpoint import load_ouro
from qualify import compare_outputs, make_engine, serve
from safetensors.torch import save_file

from vllm_rlt.models.ouro import OuroConfig, OuroForCausalLM


def tiny_config(kv_heads: int = 2) -> OuroConfig:
    return OuroConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=kv_heads,
        head_dim=8,
        max_position_embeddings=64,
    )


@pytest.mark.parametrize("source_dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("target_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("kv_heads", [1, 2])
def test_private_input_matches_public_loader(tmp_path, source_dtype, target_dtype, kv_heads):
    torch.manual_seed(81)
    original = OuroForCausalLM(tiny_config(kv_heads)).to(source_dtype).eval()
    (tmp_path / "config.json").write_text(json.dumps(original.config.to_dict()))
    save_file(original.state_dict(), tmp_path / "model.safetensors")
    public = OuroForCausalLM.from_pretrained(tmp_path, device="cpu", dtype=target_dtype)
    private = load_ouro(tmp_path, dtype=target_dtype)
    assert public.state_dict().keys() == private.state_dict().keys()
    for name, value in public.state_dict().items():
        torch.testing.assert_close(value, private.state_dict()[name], atol=0, rtol=0)
    assert not any(p.is_meta or p.requires_grad for p in private.parameters())
    outputs = []
    for model in (public, private):
        engine = make_engine(model, 1, "sync", "torch")
        outputs.append(serve(engine, 1, True)["outputs"])
        engine.close()
    assert outputs[0] == outputs[1]


@pytest.mark.parametrize("mode", ["sync", "async-single", "async-multi"])
@pytest.mark.parametrize("concurrency", [1, 4, 16])
def test_finite_output_harness_and_request_reuse(mode, concurrency):
    torch.set_num_threads(1)
    torch.manual_seed(81)
    model = OuroForCausalLM(tiny_config()).eval()
    engine = make_engine(model, concurrency, mode, "torch")
    cumulative = serve(engine, concurrency, False)
    terminal = serve(engine, concurrency, True)
    assert compare_outputs(cumulative["outputs"], terminal["outputs"])["scores_exact"]
    assert cumulative["peak_live_requests"] == terminal["peak_live_requests"] == concurrency
    engine.close()
