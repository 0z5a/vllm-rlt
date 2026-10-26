"""Loop-order SHARED prefill: packable prompt loops over one retained KV plane."""

import pytest
import torch

from tests.helpers import tiny_ouro_config
from vllm_rlt import CacheConfig, ExecutionConfig, SamplingParams, SchedulerConfig
from vllm_rlt.core.kv_cache_manager import KVCacheManager
from vllm_rlt.engine.llm_engine import LLMEngine
from vllm_rlt.models import OuroForCausalLM

PROMPT = [3, 7, 1, 9, 4, 2]
PARAMS = SamplingParams(max_tokens=3, exit_threshold=1.0, ignore_eos=True)


def model(device="cpu", dtype=torch.float32):
    torch.manual_seed(31)
    return OuroForCausalLM(tiny_ouro_config()).to(device=device, dtype=dtype)


def engine(layout, chunk, *, order="position", prefix=False, device="cpu", backend="torch", uva=False):
    dtype = torch.float32 if device == "cpu" else torch.bfloat16
    return LLMEngine(
        model(device, dtype),
        cache_config=CacheConfig(
            128, 2, layout, shared_prefill=order, enable_prefix_caching=prefix
        ),
        scheduler_config=SchedulerConfig(
            max_num_seqs=2, max_num_batched_tokens=8, prefill_chunk_size=chunk
        ),
        execution_config=ExecutionConfig(prefill_uva=uva),
        attention_backend=backend,
    )


def run(e, prompts):
    """Drain the engine; keep each request's prompt KV per plane as it is freed."""
    cache, kept, done = e.cache_manager, {}, {}
    free = cache.free

    def keep(request_id):
        if request_id in cache._allocations:
            kept[request_id] = [
                [cache.read(layer, request_id, plane, len(PROMPT)) for layer in range(2)]
                for plane in range(cache.storage_depths)
            ]
        free(request_id)

    cache.free = keep
    for request_id, prompt in prompts.items():
        e.add_request(request_id, prompt, PARAMS)
    for _ in range(1000):
        if not e.has_unfinished_requests():
            return done, kept
        for out in e.step():
            if out.finished:
                done[out.request_id] = out
    pytest.fail("engine did not drain")


def assert_kv_equal(left, right, exact=True):
    for (k1, v1), (k2, v2) in zip(left, right):
        if exact:
            assert torch.equal(k1, k2) and torch.equal(v1, v2)
        else:
            torch.testing.assert_close(k1, k2, rtol=1e-5, atol=1e-6)
            torch.testing.assert_close(v1, v2, rtol=1e-5, atol=1e-6)


def test_loop_order_requires_shared_kv():
    with pytest.raises(ValueError, match="requires shared KV"):
        CacheConfig(64, 2, shared_prefill="loop")
    with pytest.raises(ValueError, match="requires shared KV"):
        KVCacheManager(1, 1, 4, 8, 2, 2, shared_prefill="loop")
    with pytest.raises(ValueError, match="'position' or 'loop'"):
        CacheConfig(64, 2, "shared", shared_prefill="chunk")
    loop = KVCacheManager(1, 1, 4, 8, 2, 2, layout="shared", shared_prefill="loop")
    assert loop.packs_prompt_loops
    assert not KVCacheManager(1, 1, 4, 8, 2, 2, layout="shared").packs_prompt_loops


def test_single_chunk_matches_last_exited_prompt_and_keeps_its_deepest_plane():
    # Inside one chunk, loop d reads the chunk's own loop-d KV exactly as
    # LAST_EXITED does; the shared plane ends with LAST_EXITED's final plane.
    base, base_kv = run(engine("last_exited", 8), {"r": PROMPT})
    loop, loop_kv = run(engine("shared", 8, order="loop"), {"r": PROMPT})
    assert loop["r"].token_ids[0] == base["r"].token_ids[0]
    assert len(loop_kv["r"]) == 1 and len(base_kv["r"]) == 4
    assert_kv_equal(loop_kv["r"][0], base_kv["r"][3])


def test_one_token_chunks_match_position_order():
    # A one-token chunk runs every loop before the next token: that is the
    # position order, where history is always read at its final KV.
    position, position_kv = run(engine("shared", 4), {"r": PROMPT})
    loop, loop_kv = run(engine("shared", 1, order="loop"), {"r": PROMPT})
    assert loop["r"].token_ids == position["r"].token_ids
    assert_kv_equal(loop_kv["r"][0], position_kv["r"][0], exact=False)


def test_loop_order_depends_on_chunk_boundaries():
    _, one = run(engine("shared", 8, order="loop"), {"r": PROMPT})
    _, two = run(engine("shared", 2, order="loop"), {"r": PROMPT})
    # The first chunk is the same computation up to batch-size rounding; later
    # positions read that chunk at its final KV instead of loop by loop.
    head = max(
        (a[:2] - b[:2]).abs().max().item()
        for pair in zip(one["r"][0], two["r"][0])
        for a, b in zip(*pair)
    )
    tail = max((k1[2:] - k2[2:]).abs().max().item() for (k1, _), (k2, _) in zip(one["r"][0], two["r"][0]))
    assert head < 1e-5 and tail > 1e3 * max(head, 1e-7)


@pytest.mark.parametrize("order", ["position", "loop"])
def test_shared_prefix_reuse_matches_cold_generation(order):
    e = engine("shared", 2, order=order, prefix=True)
    cold, cold_kv = run(e, {"first": PROMPT})
    assert e.cache_manager.lookup_prefix(PROMPT)
    warm, warm_kv = run(e, {"second": PROMPT})
    assert e.cache_manager.prefix_hits >= 4
    assert warm["second"].token_ids == cold["first"].token_ids
    assert_kv_equal(warm_kv["second"][0], cold_kv["first"][0])
    assert e.cache_manager.num_used_blocks == 0


@pytest.mark.gpu
@pytest.mark.parametrize("uva", [False, True])
def test_cuda_loop_order_single_chunk_matches_last_exited(uva):
    base, base_kv = run(
        engine("last_exited", 8, device="cuda", backend="triton", uva=uva), {"r": PROMPT}
    )
    loop, loop_kv = run(
        engine("shared", 8, order="loop", device="cuda", backend="triton", uva=uva), {"r": PROMPT}
    )
    assert loop["r"].token_ids[0] == base["r"].token_ids[0]
    assert_kv_equal(loop_kv["r"][0], base_kv["r"][3])
