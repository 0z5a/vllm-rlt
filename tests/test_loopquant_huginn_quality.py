import pytest
import torch
from torch.nn import functional as F

from loopquant.adapters.huginn import HuginnAdapter
from loopquant.huginn_quality import native_huginn_window_nll
from tests.test_huginn import make_cache, tiny_huginn_config
from vllm_rlt.models.huginn import HuginnForCausalLM


@pytest.mark.parametrize("loops", [1, 3])
@pytest.mark.parametrize("seed", [17, 31])
@pytest.mark.parametrize("length", [2, 7])
def test_native_huginn_nll_matches_dense_and_reuses_empty_boundary_cache(loops, seed, length):
    torch.manual_seed(43)
    model = HuginnForCausalLM(tiny_huginn_config()).eval()
    adapter = HuginnAdapter(model)
    tokens = torch.arange(4, 4 + length)
    with torch.inference_mode(), torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        expected = adapter(tokens[None], torch.ones(1, length, dtype=torch.bool), loops)
        expected_nll = F.cross_entropy(expected.logits[0, :-1], tokens[1:], reduction="sum")
    cache = make_cache(model)
    rng = torch.random.get_rng_state().clone()
    first = native_huginn_window_nll(model, tokens, loops, cache, seed=seed)
    assert cache.num_used_blocks == 0 and torch.equal(rng, torch.random.get_rng_state())
    second = native_huginn_window_nll(model, tokens, loops, cache, seed=seed)
    assert first == second and first[1] == length - 1 and cache.num_used_blocks == 0
    assert first[0] == pytest.approx(float(expected_nll), abs=3e-6, rel=3e-5)


def test_native_huginn_rejects_a_cache_with_no_boundary_layer_contract():
    from vllm_rlt.core.kv_cache_manager import KVCacheManager

    model = HuginnForCausalLM(tiny_huginn_config()).eval()
    cache = KVCacheManager(4, 4, 8, 128, 2, 3, device="cpu", dtype=torch.float32)
    with pytest.raises(ValueError, match="boundary and recurrent"):
        native_huginn_window_nll(model, torch.tensor([4, 7]), 3, cache, seed=17)
    assert cache.num_used_blocks == 0
