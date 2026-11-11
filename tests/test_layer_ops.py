# SPDX-License-Identifier: Apache-2.0
"""Single-launch decoder-layer operators against the composite operators."""

import pytest
import torch
import torch.nn.functional as F

from tests.helpers import make_speculative_engine, tiny_ouro_config
from vllm_rlt import (
    CacheConfig,
    ExecutionConfig,
    SamplingParams,
    SchedulerConfig,
    SparseDecodeAttentionConfig,
    SpeculativeConfig,
)
from vllm_rlt.core.kv_cache_manager import KVCacheManager
from vllm_rlt.engine.stage_engine import StageEngine
from vllm_rlt.layers import RMSNorm, apply_rotary_pos_emb, rotate_half
from vllm_rlt.models import OuroForCausalLM

pytestmark = pytest.mark.gpu
ROUNDED = [torch.bfloat16, torch.float16]


@pytest.fixture(autouse=True)
def _check_fast_launches(monkeypatch):
    """Every reused launch must be the kernel Triton's own dispatch would pick."""
    try:
        from vllm_rlt.kernels import sparse_decode
    except ImportError:  # CPU-only installs have no Triton
        return
    monkeypatch.setattr(sparse_decode, "_LAUNCH_CHECK", True)


def _angles(rows, dim, dtype):
    angles = torch.rand(rows, dim // 2, device="cuda") * 6.0
    angles = torch.cat((angles, angles), dim=-1)
    return angles.cos().to(dtype).unsqueeze(1), angles.sin().to(dtype).unsqueeze(1)


@pytest.mark.parametrize("dtype", ROUNDED)
@pytest.mark.parametrize(
    ("q_heads", "k_heads", "dim"), [(4, 2, 8), (16, 16, 128), (3, 1, 6), (8, 8, 64)]
)
def test_rope_is_the_composite_result_bit_for_bit(dtype, q_heads, k_heads, dim):
    from vllm_rlt.kernels import layer_ops

    torch.manual_seed(3)
    rows = 5
    q = (torch.randn(rows, q_heads, dim, device="cuda") * 2).to(dtype)
    k = (torch.randn(rows, k_heads, dim, device="cuda") * 2).to(dtype)
    cos, sin = _angles(rows, dim, dtype)
    expected = (q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin)
    with torch.no_grad():
        direct = layer_ops.rope_qk(q, k, cos, sin, q_heads=q_heads, k_heads=k_heads)
        # Both operands inside one projection output, the key heads at an offset.
        packed = torch.cat((q.flatten(1), k.flatten(1), torch.ones_like(q.flatten(1))), dim=1)
        stacked = layer_ops.rope_qk(
            packed, packed, cos, sin, q_heads=q_heads, k_heads=k_heads, k_offset=q_heads * dim
        )
        dispatched = apply_rotary_pos_emb(q, k, cos, sin)
    for actual in (direct, stacked, dispatched):
        assert torch.equal(actual[0], expected[0])
        assert torch.equal(actual[1], expected[1])


def test_rope_dispatch_takes_one_launch_only_where_it_is_exact(monkeypatch):
    from vllm_rlt.kernels import layer_ops

    calls = []
    original = layer_ops.rope_qk
    monkeypatch.setattr(
        layer_ops, "rope_qk", lambda *args, **kwargs: calls.append(1) or original(*args, **kwargs)
    )
    q = torch.randn(3, 4, 8, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(3, 2, 8, device="cuda", dtype=torch.bfloat16)
    cos, sin = _angles(3, 8, torch.bfloat16)
    with torch.no_grad():
        apply_rotary_pos_emb(q, k, cos, sin)
        assert len(calls) == 1
        # FP32, non-contiguous heads and broadcast angles keep the composite operators.
        apply_rotary_pos_emb(q.float(), k.float(), cos.float(), sin.float())
        apply_rotary_pos_emb(q[:, :, ::2], k[:, :, ::2], cos[..., ::2], sin[..., ::2])
        apply_rotary_pos_emb(q, k, cos[:1], sin[:1])
    apply_rotary_pos_emb(q, k, cos, sin)
    assert len(calls) == 1


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_kv_store_is_the_indexed_assignment(dtype):
    from vllm_rlt.kernels import layer_ops

    torch.manual_seed(5)
    blocks, layers, tokens, heads, dim = 7, 3, 4, 2, 24
    keys = torch.zeros(blocks, layers, tokens, heads, dim, device="cuda", dtype=dtype)
    values = torch.zeros_like(keys)
    expected_keys, expected_values = keys.clone(), values.clone()
    rows = 6
    wide = torch.randn(rows + 2, 3 * heads * dim, device="cuda").to(dtype)
    k = torch.randn(rows + 2, heads, dim, device="cuda").to(dtype)
    # The values as a slice of a wider projection output: a strided row.
    v = wide.view(rows + 2, 3 * heads, dim)[:, 2 * heads :]
    flat = torch.randperm(blocks * tokens, device="cuda")[:rows]
    block, offset = flat // tokens, flat % tokens
    assert layer_ops.kv_store_supported(keys, values, k, v)
    layer_ops.kv_store(keys, values, 1, k, v, block, offset, rows)
    expected_keys[block, 1, offset] = k[:rows]
    expected_values[block, 1, offset] = v[:rows]
    assert torch.equal(keys, expected_keys)
    assert torch.equal(values, expected_values)
    assert not layer_ops.kv_store_supported(keys, values, k[:, :, ::2], v)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("width", [32, 1000, 2048])
def test_norm_add_is_the_norm_then_the_dtype_addition(dtype, width):
    from vllm_rlt.kernels import layer_ops

    torch.manual_seed(11)
    rows = 6
    x = (torch.randn(rows, width, device="cuda") * 3).to(dtype)
    residual = torch.randn(rows, width, device="cuda").to(dtype)
    weight = torch.randn(width, device="cuda").to(dtype)
    following = torch.randn(width, device="cuda").to(dtype)
    with torch.no_grad():
        expected = residual + layer_ops.rms_norm(x, weight, 1e-6)
        expected_next = layer_ops.rms_norm(expected, following, 1e-5)
        total, normed = layer_ops.rms_norm_add(x, weight, 1e-6, residual, following, 1e-5)
        alone, nothing = layer_ops.rms_norm_add(x, weight, 1e-6, residual)
    assert nothing is None
    if dtype == torch.float32:
        # FP32 has no rounding between the product and the sum to pin the result.
        torch.testing.assert_close(total, expected, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(alone, expected, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(normed, expected_next, rtol=1e-5, atol=1e-6)
    else:
        assert torch.equal(total, expected)
        assert torch.equal(alone, expected)
        assert torch.equal(normed, expected_next)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("width", [64, 1500, 5632])
def test_silu_mul_matches_the_composite_gating(dtype, width):
    from vllm_rlt.kernels import layer_ops

    torch.manual_seed(13)
    gate_up = (torch.randn(5, 2 * width, device="cuda") * 4).to(dtype)
    gate_up[0, :8] = torch.tensor(
        [0.0, -0.0, 30.0, -30.0, 100.0, -100.0, 1e-3, -1e-3], device="cuda", dtype=dtype
    )
    # A strided row: the gate and up halves inside a wider tensor.
    wide = torch.cat((gate_up, torch.ones_like(gate_up)), dim=1)[:, : 2 * width]
    expected = F.silu(gate_up[:, :width]) * gate_up[:, width:]
    actual = layer_ops.silu_mul(wide)
    assert actual.shape == expected.shape and actual.dtype == dtype
    ulp = {torch.float32: 2.0**-21, torch.float16: 2.0**-9, torch.bfloat16: 2.0**-6}[dtype]
    difference = (actual.float() - expected.float()).abs()
    assert torch.all(difference <= ulp * expected.float().abs().clamp_min(1e-30))


def _cache(config, dtype):
    return KVCacheManager(
        num_layers=config.num_hidden_layers,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        num_blocks=64,
        block_size=2,
        max_loops=config.total_ut_steps,
        device="cuda",
        dtype=dtype,
    )


def _traverse(model, tokens, cache, name):
    assert cache.allocate(name, len(tokens))
    hidden = model.prelude(tokens)
    states = []
    with torch.no_grad():
        for depth in range(model.config.total_ut_steps):
            hidden, gate = model.recurrent(
                hidden, [name] * len(tokens), [depth] * len(tokens), list(range(len(tokens))), cache
            )
            states.append((hidden, gate))
    return states


@pytest.mark.parametrize("dtype", ROUNDED)
@pytest.mark.parametrize("stacked", [True, False])
@pytest.mark.parametrize("kv_heads", [2, 4])
def test_single_launch_traversal_matches_the_module_traversal(
    monkeypatch, dtype, stacked, kv_heads
):
    from vllm_rlt.kernels import layer_ops

    torch.manual_seed(17)
    config = tiny_ouro_config(num_key_value_heads=kv_heads)
    model = OuroForCausalLM(config).to("cuda", dtype)
    before = {name: value.clone() for name, value in model.state_dict().items()}
    if stacked:
        model.stack_projections()
    after = model.state_dict()
    assert before.keys() == after.keys()
    assert all(torch.equal(before[name], after[name]) for name in before)
    attention = model.model.layers[0].self_attn
    assert (attention._qkv.weight is not None) == stacked
    assert (model.model.layers[0].mlp._gate_up.weight is not None) == stacked

    tokens = torch.tensor([5, 7, 2, 11, 3, 9], device="cuda")
    calls = []
    original = layer_ops.rms_norm_add
    monkeypatch.setattr(layer_ops, "rms_norm_add", lambda *args: calls.append(1) or original(*args))
    fused = _traverse(model, tokens, _cache(config, dtype), "fused")
    assert len(calls) == 2 * config.num_hidden_layers * config.total_ut_steps
    monkeypatch.setattr(model, "_single_launch_ops", lambda hidden: None)
    reference = _traverse(model, tokens, _cache(config, dtype), "reference")
    assert len(calls) == 2 * config.num_hidden_layers * config.total_ut_steps
    for (hidden, gate), (expected_hidden, expected_gate) in zip(fused, reference, strict=True):
        # The norm reduces in another order and the stacked product may too:
        # a few units in the last place of the 16-bit dtype per operator.
        torch.testing.assert_close(hidden.float(), expected_hidden.float(), rtol=0.05, atol=0.05)
        torch.testing.assert_close(gate.float(), expected_gate.float(), rtol=0.05, atol=0.05)


def test_stacked_projections_fall_back_after_the_weights_move():
    torch.manual_seed(19)
    config = tiny_ouro_config()
    model = OuroForCausalLM(config).to("cuda", torch.bfloat16)
    model.stack_projections()
    attention, mlp = model.model.layers[0].self_attn, model.model.layers[0].mlp
    hidden = torch.randn(3, config.hidden_size, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        stacked = F.linear(hidden, attention._qkv.weight)
        separate = torch.cat(
            [layer(hidden) for layer in (attention.q_proj, attention.k_proj, attention.v_proj)], 1
        )
    torch.testing.assert_close(stacked.float(), separate.float(), rtol=0.02, atol=0.02)
    # A replaced parameter and a dtype conversion both leave the stack.
    attention.k_proj.weight = torch.nn.Parameter(attention.k_proj.weight.clone(), False)
    assert attention._qkv.weight is None and mlp._gate_up.weight is not None
    model.half()
    assert mlp._gate_up.weight is None
    tokens = torch.tensor([1, 2, 3], device="cuda")
    states = _traverse(model, tokens, _cache(config, torch.float16), "moved")
    assert all(torch.isfinite(hidden.float()).all() for hidden, _ in states)


def test_rmsnorm_module_is_unchanged_by_the_chained_path():
    norm = RMSNorm(32).to("cuda", torch.bfloat16)
    x = torch.randn(4, 32, device="cuda", dtype=torch.bfloat16)
    from vllm_rlt.kernels import layer_ops

    with torch.no_grad():
        assert torch.equal(norm(x), layer_ops.rms_norm(x, norm.weight, norm.variance_epsilon))


def _stacked_bf16_model():
    torch.manual_seed(123)
    model = OuroForCausalLM(tiny_ouro_config(head_dim=64)).to("cuda", torch.bfloat16)
    model.stack_projections()
    return model


def _drain(engine, requests):
    for i, (prompt, tokens) in enumerate(requests):
        engine.add_request(str(i), prompt, SamplingParams(max_tokens=tokens, ignore_eos=True))
    outputs = {}
    for _ in range(2000):
        if not engine.has_unfinished_requests():
            break
        for output in engine.step():
            outputs[output.request_id] = output
    return {rid: output.token_ids for rid, output in outputs.items()}


def test_stacked_single_launch_graphs_replay_the_eager_tokens():
    requests = [([2, 3, 4] * 5, 23), ([3, 3, 4] * 5 + [5, 5], 9), ([4, 3, 4] * 10 + [5], 1)]
    results = []
    for graphs in (False, True):
        engine = StageEngine(
            _stacked_bf16_model(),
            attention_backend="triton",
            cache_config=CacheConfig(128, 16, incremental_allocation=True),
            scheduler_config=SchedulerConfig(
                max_num_seqs=3, max_num_batched_tokens=16, prefill_chunk_size=8
            ),
            execution_config=ExecutionConfig(
                async_scheduling=True,
                multi_stream=True,
                static_buffers=True,
                pad_to_power_of_two=True,
                cuda_graphs=graphs,
                cuda_graph_max_batch_size=32,
                cuda_graph_max_graphs=16,
            ),
            speculative_config=SpeculativeConfig(3),
        )
        try:
            results.append(_drain(engine, requests))
            if graphs:
                assert engine.speculative_runner.graphs.replays > 0
        finally:
            engine.close()
    assert len(results[0]) == len(requests)
    assert results[0] == results[1]


@pytest.mark.parametrize("implementation", ["fused", "batched"])
def test_stacked_single_launch_sparse_engine_keeps_every_key_like_dense(implementation):
    requests = [([2, 3, 4], 12), ([5, 6], 8)]
    results = []
    for sparse in (
        None,
        SparseDecodeAttentionConfig(enabled=True, key_retention=1.0, implementation=implementation),
    ):
        engine = make_speculative_engine(_stacked_bf16_model(), sparse=sparse)
        try:
            results.append(_drain(engine, requests))
            if sparse is not None:
                assert engine.reference_store.outstanding == 0
        finally:
            engine.close()
    assert len(results[0]) == len(requests)
    assert results[0] == results[1]
