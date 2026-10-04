import pytest
import torch

from loopquant.adapters.nanbeige import NanbeigeAdapter
from loopquant.quantizers import FP8FakeLinear, ScaleLayout
from loopquant.train import Q0Config, Q0Trainer, TrainingBatch
from tests.helpers import tiny_nanbeige_config
from vllm_rlt.core.kv_cache_manager import KVCacheManager
from vllm_rlt.models.nanbeige import NanbeigeForCausalLM


@pytest.mark.parametrize("attention_backend", ["eager", "sdpa"])
def test_nanbeige_matches_native_fixed_loops_and_masks(attention_backend):
    torch.manual_seed(29)
    config = tiny_nanbeige_config()
    model = NanbeigeForCausalLM(config).requires_grad_(False).eval()
    adapter = NanbeigeAdapter(model, attention_backend=attention_backend)
    adapter.collect = True
    tokens = torch.tensor([[5, 7, 2, 11], [3, 9, 0, 0]])
    valid = torch.tensor([[True, True, True, True], [True, True, False, False]])
    result = adapter(tokens, valid, 2)
    cache = KVCacheManager(
        config.num_hidden_layers,
        config.num_key_value_heads,
        config.head_dim,
        num_blocks=16,
        block_size=2,
        max_loops=2,
    )
    for index, length in enumerate([4, 2]):
        request = str(index)
        assert cache.allocate(request, length)
        hidden = model.prelude(tokens[index, :length])
        for loop in range(2):
            hidden, _ = model.recurrent(
                hidden, [request] * length, [loop] * length, list(range(length)), cache
            )
            torch.testing.assert_close(
                hidden, result.states[loop][index, :length], atol=2e-6, rtol=2e-5
            )
        torch.testing.assert_close(
            model.coda(hidden), result.logits[index, :length], atol=2e-6, rtol=2e-5
        )
        cache.free(request)
    assert all(stats.rows == 6 for stats in adapter.statistics.values())


def test_nanbeige_scale_gradient_and_matching_teacher():
    torch.manual_seed(23)
    model = NanbeigeForCausalLM(tiny_nanbeige_config()).requires_grad_(False).eval()
    student = NanbeigeAdapter(model, attention_backend="sdpa")
    name = "model.layers.0.self_attn.q_proj"
    quantized = FP8FakeLinear(
        model.model.layers[0].self_attn.q_proj, torch.tensor([0.002]), ScaleLayout(2)
    )
    student.attach(name, quantized)
    trainer = Q0Trainer(student, Q0Config(loops=2, tokens_per_update=4, kl_weight=0.1))
    assert isinstance(trainer.teacher, NanbeigeAdapter)
    assert trainer.teacher.attention_backend == "sdpa"
    batch = TrainingBatch(torch.tensor([[5, 7, 2, 11]]), torch.ones(1, 4, dtype=torch.bool))
    pointer = quantized.packed_weight.data_ptr()
    for _ in range(2):
        receipt = trainer.step([batch])
        assert receipt["gradient_norm"] > 0
        assert quantized.packed_weight.data_ptr() == pointer
        assert sum(receipt["scale_update_norms"].values()) > 0
