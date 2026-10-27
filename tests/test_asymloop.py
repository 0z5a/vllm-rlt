"""AsymLoop on Ouro weights: auxiliary KV planes, two-source attention, engine parity."""

import math

import pytest
import torch

from tests.helpers import tiny_ouro_config
from vllm_rlt import CacheConfig, ExecutionConfig, SamplingParams, SchedulerConfig
from vllm_rlt.core.kv_cache_manager import KVCacheManager
from vllm_rlt.engine.llm_engine import LLMEngine
from vllm_rlt.kernels.paged_attention import torch_paged_attention
from vllm_rlt.models import OuroForCausalLM
from vllm_rlt.models.asymloop import AsymLoopOuroForCausalLM

PROMPTS = {"a": [3, 7, 1, 9, 4, 2], "b": [5, 8, 2, 6, 1, 3, 9, 7, 4]}
PARAMS = SamplingParams(max_tokens=4, exit_threshold=1.0, ignore_eos=True)


def asymloop(device="cpu", dtype=torch.float32, *, loops=3):
    torch.manual_seed(7)
    base = OuroForCausalLM(tiny_ouro_config(num_hidden_layers=4)).to(device=device, dtype=dtype)
    return AsymLoopOuroForCausalLM(base, encoder_layers=2, loops=loops)


def dense_logits(model, tokens):
    """Unpaged AsymLoop: encoder once, fixed global KV, decoder loops, one softmax."""
    base, config = model.base, model.config
    layers = base.model.layers
    count, dim = len(tokens), config.head_dim
    hidden = base.model.embed_tokens(torch.tensor(tokens, device=base.lm_head.weight.device))
    positions = torch.arange(count, device=hidden.device)
    angles = positions.float()[:, None] * base.model.rotary_emb.inv_freq.float()
    angles = torch.cat((angles, angles), dim=-1)
    cos, sin = angles.cos().to(hidden.dtype)[:, None], angles.sin().to(hidden.dtype)[:, None]

    def rope(x):
        half = x.shape[-1] // 2
        return x * cos + torch.cat((-x[..., half:], x[..., :half]), dim=-1) * sin

    def attend(q, keys, values):
        k = torch.cat(keys).repeat_interleave(q.shape[1] // keys[0].shape[1], dim=1)
        v = torch.cat(values).repeat_interleave(q.shape[1] // values[0].shape[1], dim=1)
        scores = torch.einsum("qhd,khd->hqk", q, k) / math.sqrt(dim)
        visible = torch.ones(count, count, dtype=torch.bool, device=q.device).tril()
        scores = scores.masked_fill(~visible.repeat(1, len(keys)), -math.inf)
        return torch.einsum("hqk,khd->qhd", scores.softmax(-1), v).reshape(count, -1)

    def layer_step(layer, x, extra_keys=(), extra_values=()):
        attention = layer.self_attn
        normed = layer.input_layernorm(x)
        q = rope(attention.q_proj(normed).view(count, -1, dim))
        k = rope(attention.k_proj(normed).view(count, -1, dim))
        v = attention.v_proj(normed).view(count, -1, dim)
        out = attention.o_proj(attend(q, [*extra_keys, k], [*extra_values, v]))
        x = x + layer.input_layernorm_2(out)
        return x + layer.post_attention_layernorm_2(layer.mlp(layer.post_attention_layernorm(x)))

    for layer in layers[: model.encoder_layers]:
        hidden = layer_step(layer, hidden)
    fixed = []
    for layer, projection in zip(layers[model.encoder_layers :], model.global_kv):
        k, v = projection(layer.input_layernorm(hidden)).view(count, 2, -1, dim).unbind(1)
        fixed.append((rope(k), v))
    for _ in range(config.total_ut_steps):
        for layer, (k, v) in zip(layers[model.encoder_layers :], fixed):
            hidden = layer_step(layer, hidden, [k], [v])
        hidden = base.model.norm(hidden)
    return base.lm_head(hidden)


def dense_greedy(model, prompt, steps):
    tokens = list(prompt)
    for _ in range(steps):
        tokens.append(int(dense_logits(model, tokens)[-1].argmax()))
    return tokens[len(prompt) :]


def engine(model, chunk, *, backend="torch", uva=False):
    return LLMEngine(
        model,
        cache_config=CacheConfig(256, 2, "last_exited"),
        scheduler_config=SchedulerConfig(
            max_num_seqs=2, max_num_batched_tokens=8, prefill_chunk_size=chunk
        ),
        execution_config=ExecutionConfig(prefill_uva=uva),
        attention_backend=backend,
    )


def drain(e, prompts):
    done = {}
    for request_id, prompt in prompts.items():
        e.add_request(request_id, prompt, PARAMS)
    for _ in range(1000):
        if not e.has_unfinished_requests():
            return done
        for out in e.step():
            if out.finished:
                done[out.request_id] = out.token_ids
    pytest.fail("engine did not drain")


def test_two_source_reference_is_one_softmax():
    torch.manual_seed(0)
    keys, values = torch.randn(10, 4, 2, 8), torch.randn(10, 4, 2, 8)
    q = torch.randn(3, 4, 8)
    tables = torch.tensor([[0, 1, 5, 6], [2, 3, 7, 8], [4, -1, 9, -1]], dtype=torch.int32)
    lengths = torch.tensor([7, 8, 3], dtype=torch.int32)
    out = torch_paged_attention(q, keys, values, tables, lengths, sources=2)
    for row, length in enumerate(lengths.tolist()):
        index = torch.arange(length)
        blocks = torch.cat([tables[row, s * 2 + index // 4] for s in range(2)])
        k = keys[blocks, index.repeat(2) % 4].repeat_interleave(2, dim=1)
        v = values[blocks, index.repeat(2) % 4].repeat_interleave(2, dim=1)
        weights = (torch.einsum("hd,thd->ht", q[row], k) / math.sqrt(8)).softmax(-1)
        torch.testing.assert_close(out[row], torch.einsum("ht,thd->hd", weights, v))


def test_auxiliary_planes_allocate_whole_page_groups():
    cache = KVCacheManager(1, 1, 4, 20, 2, 3, aux_planes=2)
    # 20 blocks, 5 planes per page: 4 groups; plane p of group g is p * 4 + g.
    assert (cache.num_blocks, cache.block_groups, cache.storage_depths) == (20, 4, 5)
    assert cache.required_blocks(3) == 10 and cache.num_free_blocks == 20
    assert cache.allocate("a", 3, initial_tokens=1)
    assert cache.allocate("b", 4)
    assert cache.ensure_capacity("a", 3)
    for request_id in ("a", "b"):
        tables = cache.plane_block_tables(request_id)
        for page in zip(*tables):
            assert [b // 4 for b in page] == list(range(5))
            assert len({b % 4 for b in page}) == 1
    assert cache.num_free_blocks == 0 and not cache.allocate("c", 1)
    cache.free("b")
    assert cache.num_free_blocks == 10 and cache.allocate("c", 2)
    blocks = torch.tensor(cache.plane_block_tables("c")[1])
    assert cache._aux_blocks(blocks, 1).tolist() == list(cache.plane_block_tables("c")[4])
    with pytest.raises(ValueError, match="last_exited"):
        KVCacheManager(1, 1, 4, 20, 2, 3, layout="shared", aux_planes=2)


def test_engine_kv_holds_encoder_global_and_local_slots():
    model = asymloop()
    e = engine(model, 4)
    cache = e.cache_manager
    # Decoder-width slots, three loop planes plus encoder and global planes.
    assert (cache.num_layers, cache.storage_depths, cache.aux_planes) == (2, 5, 2)
    assert cache.num_blocks % 5 == 0


@pytest.mark.parametrize("chunk", [2, 4, 8])
def test_engine_matches_dense_reference(chunk):
    model = asymloop()
    done = drain(engine(model, chunk), PROMPTS)
    for request_id, prompt in PROMPTS.items():
        assert done[request_id] == dense_greedy(model, prompt, PARAMS.max_tokens)


def test_prompt_logits_match_dense_reference():
    model = asymloop(loops=2)
    cache = KVCacheManager(2, 2, 8, 50, 2, 2, aux_planes=2)
    tokens = PROMPTS["b"]
    assert cache.allocate("r", len(tokens))
    ids, positions = ["r"] * len(tokens), list(range(len(tokens)))
    boundary = cache._prepare_batch(ids, [0] * len(tokens), positions)
    hidden = model.prelude_prepared(torch.tensor(tokens), boundary, cache)
    for depth in range(2):
        hidden, _ = model.recurrent(hidden, ids, [depth] * len(tokens), positions, cache)
    torch.testing.assert_close(model.coda(hidden), dense_logits(model, tokens))


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_triton_two_source_kernels_match_reference(dtype):
    from vllm_rlt.kernels.triton_attention import paged_attention, paged_prefill_attention

    torch.manual_seed(1)
    device = "cuda"
    keys = torch.randn(64, 16, 2, 64, device=device, dtype=dtype)
    values = torch.randn_like(keys)
    lengths = torch.tensor([1, 37, 200, 513], device=device, dtype=torch.int32)
    width = 33
    tables = torch.randint(0, 64, (4, 2 * width), device=device, dtype=torch.int32)
    q = torch.randn(4, 8, 64, device=device, dtype=dtype)
    expected = torch_paged_attention(q, keys, values, tables, lengths, sources=2)
    tolerance = dict(rtol=1e-4, atol=1e-4) if dtype == torch.float32 else dict(rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(
        paged_attention(q, keys, values, tables, lengths, sources=2), expected, **tolerance
    )
    # Packed prefill: the last query rows of each sequence see the same keys as decode rows.
    query_lengths = [1, 5, 64, 70]
    cu = torch.tensor([0, *torch.tensor(query_lengths).cumsum(0).tolist()], device=device)
    packed = torch.randn(sum(query_lengths), 8, 64, device=device, dtype=dtype)
    out = paged_prefill_attention(
        packed, keys, values, tables, lengths, cu.to(torch.int32), max(query_lengths), sources=2
    )
    for seq, count in enumerate(query_lengths):
        rows = packed[cu[seq] : cu[seq] + count]
        prefix = int(lengths[seq]) - count
        reference = torch_paged_attention(
            rows,
            keys,
            values,
            tables[seq : seq + 1].expand(count, -1).contiguous(),
            torch.arange(prefix + 1, prefix + count + 1, device=device, dtype=torch.int32),
            sources=2,
        )
        torch.testing.assert_close(out[cu[seq] : cu[seq] + count], reference, **tolerance)


@pytest.mark.gpu
@pytest.mark.parametrize("uva", [False, True])
def test_cuda_engine_matches_dense_reference(uva):
    model = asymloop("cuda", torch.float32)
    done = drain(engine(model, 4, backend="triton", uva=uva), PROMPTS)
    for request_id, prompt in PROMPTS.items():
        assert done[request_id] == dense_greedy(model, prompt, PARAMS.max_tokens)
