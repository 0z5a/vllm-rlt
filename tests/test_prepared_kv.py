"""Prepared metadata must preserve cache ownership, causal history and Ouro work."""

import math
import weakref
from types import SimpleNamespace

import pytest
import torch

from tests.helpers import tiny_ouro_config
from vllm_rlt.attention import BackendCapabilities
from vllm_rlt.core.kv_cache_manager import KVCacheManager, KVSpan
from vllm_rlt.models import OuroForCausalLM


@pytest.fixture(autouse=True)
def no_cuda(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("prepared KV CPU tests must not query or initialize CUDA")

    for name in ("is_available", "device_count", "init", "_lazy_init"):
        monkeypatch.setattr(torch.cuda, name, forbidden)


def make_cache(**overrides):
    values = dict(
        num_layers=3, num_kv_heads=1, head_dim=4, num_blocks=36, block_size=2, max_loops=4
    )
    values.update(overrides)
    return KVCacheManager(**values)


def kv_span(request_id, depth, start, count):
    return KVSpan(request_id=request_id, depth=depth, start=start, count=count)


def rows_from_spans(spans):
    ids, depths, positions = [], [], []
    for span in spans:
        ids.extend([span.request_id] * span.count)
        depths.extend([span.depth] * span.count)
        positions.extend(range(span.start, span.start + span.count))
    return ids, depths, positions


def assert_same_batch(row_batch, span_batch):
    assert [name for name, _ in row_batch.allocations] == [
        name for name, _ in span_batch.allocations
    ]
    assert [id(a) for _, a in row_batch.allocations] == [id(a) for _, a in span_batch.allocations]
    assert [id(a) for a, _, _ in row_batch.rows] == [id(a) for a, _, _ in span_batch.rows]
    assert [depth for _, depth, _ in row_batch.rows] == [depth for _, depth, _ in span_batch.rows]
    assert [position for _, _, position in row_batch.rows] == [
        position for _, _, position in span_batch.rows
    ]
    for name in (
        "position_ids",
        "write_blocks",
        "write_offsets",
        "block_tables",
        "context_lengths",
    ):
        a, b = getattr(row_batch, name), getattr(span_batch, name)
        assert a.dtype == b.dtype and a.device == b.device and a.shape == b.shape, name
        assert torch.equal(a.cpu(), b.cpu()), name
    assert row_batch.writable == span_batch.writable
    assert row_batch.max_seqlen_q == span_batch.max_seqlen_q
    assert (row_batch.cu_seqlens_q is None) == (span_batch.cu_seqlens_q is None)
    if row_batch.cu_seqlens_q is not None:
        assert torch.equal(row_batch.cu_seqlens_q.cpu(), span_batch.cu_seqlens_q.cpu())


@pytest.mark.parametrize("block_size", [1, 2, 16])
@pytest.mark.parametrize("layout", ["last_exited", "shared"])
@pytest.mark.parametrize("for_write", [True, False])
@pytest.mark.parametrize("shape", ["single", "aligned", "cross_page", "misaligned", "multi_page"])
def test_span_frontend_matches_row_frontend(block_size, layout, for_write, shape):
    cache = make_cache(num_blocks=512, block_size=block_size, layout=layout)
    assert cache.allocate("a", 64)
    start, count = {
        "single": (0, 1),
        "aligned": (0, block_size),
        "cross_page": (max(0, block_size - 1), 2),
        "misaligned": (5, 3),
        "multi_page": (0, 3 * block_size + 1),
    }[shape]
    spans = [kv_span("a", 2, start, count)]
    ids, depths, positions = rows_from_spans(spans)
    row_batch = cache._prepare_batch(ids, depths, positions, for_write=for_write)
    span_batch = cache._prepare_batch_spans(spans, for_write=for_write)
    assert_same_batch(row_batch, span_batch)


def test_span_frontend_preserves_mixed_depth_and_packed_sequence_order():
    cache = make_cache(num_blocks=128, block_size=4)
    assert cache.allocate("a", 20) and cache.allocate("b", 20)
    spans = [
        kv_span("a", 0, 0, 2),
        kv_span("a", 0, 2, 2),
        kv_span("b", 3, 0, 3),
        kv_span("a", 0, 8, 1),
    ]
    ids, depths, positions = rows_from_spans(spans)
    assert_same_batch(
        cache._prepare_batch(ids, depths, positions), cache._prepare_batch_spans(spans)
    )

    cache.attention = SimpleNamespace(capabilities=BackendCapabilities(packed_prefill=True))
    packed_spans = [kv_span("a", 0, 5, 3), kv_span("b", 1, 2, 2)]
    ids, depths, positions = rows_from_spans(packed_spans)
    assert_same_batch(
        cache._prepare_batch(ids, depths, positions, packed_prefill=True),
        cache._prepare_batch_spans(packed_spans, packed_prefill=True),
    )
    nonadjacent_spans = [
        kv_span("a", 0, 0, 2),
        kv_span("b", 0, 0, 2),
        kv_span("a", 0, 4, 2),
    ]
    ids, depths, positions = rows_from_spans(nonadjacent_spans)
    messages = []
    for prepare in (
        lambda: cache._prepare_batch(ids, depths, positions, packed_prefill=True),
        lambda: cache._prepare_batch_spans(nonadjacent_spans, packed_prefill=True),
    ):
        with pytest.raises(ValueError) as error:
            prepare()
        messages.append(str(error.value))
    assert messages == [
        "packed prefill request/depth must form one sequence",
        "packed prefill request/depth must form one sequence",
    ]


@pytest.mark.parametrize("layout", ["last_exited", "shared"])
def test_span_frontend_preserves_packed_capability_error(layout):
    cache = make_cache(layout=layout)
    assert cache.allocate("a", 2)
    spans = [kv_span("a", 0, 0, 1)]
    messages = []
    for prepare in (
        lambda: cache._prepare_batch(["a"], [0], [0], packed_prefill=True),
        lambda: cache._prepare_batch_spans(spans, packed_prefill=True),
    ):
        with pytest.raises(ValueError) as error:
            prepare()
        messages.append(str(error.value))
    assert messages == [
        "packed prefill requires LAST_EXITED and FlashAttention-4",
        "packed prefill requires LAST_EXITED and FlashAttention-4",
    ]


@pytest.mark.parametrize(
    "spans, expected",
    [
        ([kv_span("a", 4, 0, 1)], "depth must be in [0, 4)"),
        ([kv_span("a", 0, 64, 1)], "position must be in [0, 64)"),
        ([kv_span("a", 0, 63, 2)], "position must be in [0, 64)"),
        (
            [kv_span("a", 0, 0, 1), kv_span("a", 0, 0, 1)],
            "a write batch cannot contain duplicate request/depth/position addresses",
        ),
    ],
)
def test_span_frontend_preserves_row_validation_errors(spans, expected):
    cache = make_cache(num_blocks=128)
    assert cache.allocate("a", 64)
    ids, depths, positions = rows_from_spans(spans)
    messages = []
    for prepare in (
        lambda: cache._prepare_batch(ids, depths, positions),
        lambda: cache._prepare_batch_spans(spans),
    ):
        with pytest.raises(ValueError) as error:
            prepare()
        messages.append(str(error.value))
    assert messages == [expected, expected]

    with pytest.raises(ValueError, match="span token counts must be positive integers"):
        cache._prepare_batch_spans([kv_span("a", 0, 0, 0)])


def test_span_frontend_matches_prefix_cache_and_incremental_growth():
    cache = make_cache(
        num_blocks=128,
        block_size=4,
        enable_prefix_caching=True,
        incremental_allocation=True,
    )
    assert cache.allocate("source", 16, initial_tokens=8)
    cache.mark_imported_prefix("source", 8)
    tokens = list(range(20))
    cache.publish_prefix("source", tokens, 8)
    prefix = cache.lookup_prefix(tokens)
    assert prefix
    assert cache.allocate("a", 20, initial_tokens=8, prefix=prefix)
    assert cache.ensure_capacity("a", 16)

    spans = [kv_span("a", 0, 6, 2), kv_span("a", 1, 12, 4)]
    ids, depths, positions = rows_from_spans(spans)
    assert_same_batch(
        cache._prepare_batch(ids, depths, positions), cache._prepare_batch_spans(spans)
    )


def test_span_and_row_batches_keep_prefix_hole_validation_identical():
    cache = make_cache()
    assert cache.allocate("a", 4)
    values = torch.ones(1, 1, 4)
    for position in (3, 0, 1):
        cache.write(0, ["a"], [0], [position], values, values)

    short_spans = [kv_span("a", 0, 0, 2)]
    long_spans = [kv_span("a", 0, 0, 4)]
    short_ids, short_depths, short_positions = rows_from_spans(short_spans)
    long_ids, long_depths, long_positions = rows_from_spans(long_spans)
    short_batches = (
        cache._prepare_batch(short_ids, short_depths, short_positions, for_write=False),
        cache._prepare_batch_spans(short_spans, for_write=False),
    )
    long_batches = (
        cache._prepare_batch(long_ids, long_depths, long_positions, for_write=False),
        cache._prepare_batch_spans(long_spans, for_write=False),
    )
    query = torch.zeros(2, 1, 4)
    for batch in short_batches:
        assert cache._attend_prepared(0, batch, query).shape == query.shape
    errors = []
    for batch in long_batches:
        with pytest.raises(RuntimeError) as error:
            cache._attend_prepared(0, batch, torch.zeros(4, 1, 4))
        errors.append(str(error.value))
    assert errors[0] == errors[1]

    cache.write(0, ["a"], [0], [2], values, values)
    for batch in long_batches:
        assert cache._attend_prepared(0, batch, torch.zeros(4, 1, 4)).shape == (4, 1, 4)


def test_fragmented_mixed_depth_rows_match_dense_attention_without_layer_metadata(monkeypatch):
    torch.manual_seed(71)
    cache = make_cache()
    for name, size in (("left", 4), ("protected", 2), ("right", 4)):
        assert cache.allocate(name, size)
    cache.key_cache.fill_(-99)
    cache.value_cache.fill_(-98)
    cache.free("left")
    cache.free("right")
    assert cache.allocate("a", 5) and cache.allocate("b", 3)
    assert any(
        table != tuple(range(table[0], table[0] + len(table)))
        for table in cache._allocations["a"].block_tables
    )
    ids = ["a"] * 10 + ["b"] * 3
    depths = [0] * 5 + [3] * 5 + [1] * 3
    positions = list(range(5)) * 2 + list(range(3))
    batch = cache._prepare_batch(ids, depths, positions)
    assert len(batch.allocations) == 2
    assert batch.block_tables.shape == (13, 3)
    assert batch.block_tables.dtype == batch.context_lengths.dtype == torch.int32
    assert batch.position_ids.dtype == batch.write_blocks.dtype == torch.int64
    assert batch.write_offsets.dtype == torch.int64
    assert batch.block_tables[-1, -1].item() == -1
    keys, values, query = torch.randn(13, 1, 4), torch.randn(13, 1, 4), torch.randn(13, 2, 4)
    protected_pages = [
        page for table in cache._allocations["protected"].block_tables for page in table
    ]
    protected = cache.key_cache[protected_pages].clone(), cache.value_cache[protected_pages].clone()

    def no_tensor_construction(*args, **kwargs):
        pytest.fail("a prepared layer rebuilt metadata tensors")

    with monkeypatch.context() as local:
        local.setattr(torch, "tensor", no_tensor_construction)
        for layer in range(cache.num_layers):
            layer_values = values + layer
            cache._write_prepared(layer, batch, keys, layer_values)
            actual = cache._attend_prepared(layer, batch, query)
            for row, (name, depth, position) in enumerate(zip(ids, depths, positions)):
                selected = [
                    i
                    for i, address in enumerate(zip(ids, depths, positions))
                    if address[:2] == (name, depth) and address[2] <= position
                ]
                dense_k = keys[selected].repeat_interleave(2, dim=1)
                dense_v = layer_values[selected].repeat_interleave(2, dim=1)
                scores = torch.einsum("hd,thd->ht", query[row], dense_k) / math.sqrt(4)
                expected = torch.einsum("ht,thd->hd", scores.softmax(-1), dense_v)
                torch.testing.assert_close(actual[row], expected, atol=0, rtol=0)
    torch.testing.assert_close(cache.key_cache[protected_pages], protected[0], atol=0, rtol=0)
    torch.testing.assert_close(cache.value_cache[protected_pages], protected[1], atol=0, rtol=0)


@pytest.mark.parametrize("tensor_positions", [False, True])
def test_preparation_owns_normalized_metadata_without_aliasing_caller_inputs(tensor_positions):
    cache = make_cache()
    cache.allocate("a", 2)
    ids, depths = ["a"], [0]
    positions = torch.tensor([0], dtype=torch.int32) if tensor_positions else [0]
    batch = cache._prepare_batch(ids, depths, positions)
    ids[0], depths[0], positions[0] = "missing", 3, 1
    assert batch.position_ids.tolist() == [0]
    assert batch.rows[0][1:] == (0, 0)
    k = torch.ones(1, 1, 4)
    cache._write_prepared(0, batch, k, k + 1)
    torch.testing.assert_close(cache.read(0, "a", 0, 1)[0], k, atol=0, rtol=0)
    assert cache.read(0, "a", 3)[0].shape[0] == 0


@pytest.mark.parametrize("operation", ["write", "attend"])
@pytest.mark.parametrize("stale_kind", ["free", "same_id_reallocation", "wrong_manager"])
def test_stale_batches_cannot_access_recycled_pages(operation, stale_kind):
    cache = make_cache(num_blocks=8)
    cache.allocate("a", 4)
    batch = cache._prepare_batch(["a"], [0], [0])
    if stale_kind == "wrong_manager":
        target = make_cache(num_blocks=8)
        target.allocate("a", 4)
    else:
        target = cache
        cache.free("a")
        if stale_kind == "same_id_reallocation":
            cache.allocate("a", 4)
    target.key_cache.fill_(17)
    target.value_cache.fill_(19)
    key_before, value_before = target.key_cache.clone(), target.value_cache.clone()
    tensor = torch.ones(1, 1, 4)
    with pytest.raises((ValueError, RuntimeError), match="different cache|stale"):
        if operation == "write":
            target._write_prepared(0, batch, tensor, tensor)
        else:
            target._attend_prepared(0, batch, tensor)
    torch.testing.assert_close(target.key_cache, key_before, atol=0, rtol=0)
    torch.testing.assert_close(target.value_cache, value_before, atol=0, rtol=0)
    for allocation in target._allocations.values():
        assert all(
            written.prefix == 0 and not written.pending
            for layers in allocation.written
            for written in layers
        )


def test_other_request_release_does_not_invalidate_a_live_descriptor():
    cache = make_cache()
    cache.allocate("a", 2)
    cache.allocate("b", 2)
    batch = cache._prepare_batch(["a"], [0], [0])
    cache.free("b")
    cache.allocate("replacement", 2)
    values = torch.ones(1, 1, 4)
    cache._write_prepared(0, batch, values, values)
    torch.testing.assert_close(cache._attend_prepared(0, batch, values), values)


def test_preparation_and_other_layers_do_not_authorize_uninitialized_prefixes():
    cache = make_cache()
    cache.allocate("a", 4)
    batch = cache._prepare_batch(["a", "a"], [0, 0], [2, 0])
    values = torch.ones(2, 1, 4)
    for layer in range(cache.num_layers):
        with pytest.raises(RuntimeError, match="uninitialized"):
            cache._attend_prepared(layer, batch, values)
    cache._write_prepared(0, batch, values, values)
    with pytest.raises(RuntimeError, match="uninitialized"):
        cache._attend_prepared(0, batch, values)
    cache.write(0, ["a"], [0], [1], values[:1], values[:1])
    torch.testing.assert_close(cache._attend_prepared(0, batch, values), values)
    with pytest.raises(RuntimeError, match="layer 1"):
        cache._attend_prepared(1, batch, values)
    with pytest.raises(RuntimeError, match="every layer"):
        cache.finalize_token("a", 2, 0)
    assert cache._allocations["a"].written[0][1].prefix == 0


def test_duplicate_query_rows_remain_valid_but_never_authorize_writes():
    cache = make_cache()
    cache.allocate("a", 2)
    k, v = torch.ones(1, 1, 4), torch.full((1, 1, 4), 3.0)
    cache.write(0, ["a"], [0], [0], k, v)
    q = torch.zeros(2, 2, 4)
    batch = cache._prepare_batch(["a", "a"], [0, 0], [0, 0], for_write=False)
    torch.testing.assert_close(cache._attend_prepared(0, batch, q), torch.full_like(q, 3))
    torch.testing.assert_close(
        cache.attend(0, ["a", "a"], [0, 0], [0, 0], q), torch.full_like(q, 3)
    )
    with pytest.raises(ValueError, match="read-only"):
        cache._write_prepared(0, batch, k.expand(2, -1, -1), v.expand(2, -1, -1))
    with pytest.raises(ValueError, match="duplicate"):
        cache.write(0, ["a", "a"], [0, 0], [0, 0], k.expand(2, -1, -1), v.expand(2, -1, -1))


@pytest.mark.parametrize("method", ["write", "attend"])
@pytest.mark.parametrize(
    "ids,depths,positions,exception",
    [
        (["a"], [0], [True], ValueError),
        (["a"], [0], [0.5], ValueError),
        (["a"], [0], [2], ValueError),
        (["a"], [4], [0], ValueError),
        (["a"], [], [0], ValueError),
        (["missing"], [0], [0], KeyError),
        (["a"], [0], torch.tensor([[0]]), ValueError),
        (["a"], [0], torch.tensor([0.0]), ValueError),
    ],
)
def test_invalid_public_rows_fail_before_kv_mutation(method, ids, depths, positions, exception):
    cache = make_cache()
    cache.allocate("a", 2)
    cache.key_cache.zero_()
    cache.value_cache.zero_()
    tensor = torch.ones(1, 1, 4)
    with pytest.raises(exception):
        args = [0, ids, depths, positions, tensor]
        getattr(cache, method)(*args, *([tensor] if method == "write" else []))
    assert not cache.key_cache.count_nonzero() and not cache.value_cache.count_nonzero()
    assert cache.read(0, "a", 0)[0].shape[0] == 0


def test_empty_adapters_preserve_layer_and_tensor_validation():
    cache = make_cache()
    k = torch.empty(0, 1, 4)
    q = torch.empty(0, 2, 4)
    assert cache.write(0, [], [], [], k, k) is None
    assert cache.attend(0, [], [], torch.empty(0, dtype=torch.int32), q).shape == q.shape
    with pytest.raises(ValueError, match="layer"):
        cache.write(3, [], [], [], k, k)
    with pytest.raises(ValueError, match="dtype"):
        cache.write(0, [], [], [], k.half(), k.half())
    with pytest.raises(ValueError, match="positive multiple"):
        cache.attend(0, [], [], [], torch.empty(0, 0, 4))
    with pytest.raises(ValueError, match="equal lengths"):
        cache.attend(0, [], [0], [], q)


def test_one_preparation_and_position_tensor_per_core_across_24_layers(monkeypatch):
    torch.manual_seed(43)
    config = tiny_ouro_config(num_hidden_layers=24)
    model = OuroForCausalLM(config)
    cache = make_cache(
        num_layers=24, num_kv_heads=config.num_key_value_heads, head_dim=config.head_dim
    )
    cache.allocate("a", 3)
    original_prepare = cache._prepare_batch
    tensor_constructor = torch.tensor
    preparations, metadata_counts, rotary_ids, layer_ids = [], [], [], []

    def prepare(*args, **kwargs):
        count = 0

        def tensor(*args, **kwargs):
            nonlocal count
            count += 1
            return tensor_constructor(*args, **kwargs)

        with monkeypatch.context() as local:
            local.setattr(torch, "tensor", tensor)
            batch = original_prepare(*args, **kwargs)
        preparations.append(weakref.ref(batch))
        metadata_counts.append(count)
        return batch

    def forbidden(*args, **kwargs):
        pytest.fail("Ouro layers must consume prepared metadata, not public adapters")

    monkeypatch.setattr(cache, "_prepare_batch", prepare)
    monkeypatch.setattr(cache, "write", forbidden)
    monkeypatch.setattr(cache, "attend", forbidden)
    handles = [
        model.model.rotary_emb.register_forward_pre_hook(
            lambda module, args: rotary_ids.append(args[1].data_ptr())
        )
    ]
    handles += [
        layer.register_forward_pre_hook(
            lambda module, args: layer_ids.append((id(args[2]), args[2].position_ids.data_ptr()))
        )
        for layer in model.model.layers
    ]
    hidden = model.prelude(torch.tensor([1, 2, 3]))
    try:
        for depth in range(4):
            hidden, _ = model.recurrent(hidden, ["a"] * 3, [depth] * 3, [0, 1, 2], cache)
    finally:
        for handle in handles:
            handle.remove()
    # Two staged host tensors (int64 addresses, int32 tables/lengths) per core traversal.
    assert metadata_counts == [2] * 4
    assert len(layer_ids) == 96
    for traversal in range(4):
        group = layer_ids[traversal * 24 : (traversal + 1) * 24]
        assert len(set(group)) == 1
        assert group[0][1] == rotary_ids[traversal]
    assert all(reference() is None for reference in preparations)


@pytest.mark.parametrize("packed", [False, True])
def test_depth_batches_share_depth_independent_metadata(packed):
    cache = make_cache(num_blocks=64)
    if packed:
        cache.attention = SimpleNamespace(capabilities=BackendCapabilities(packed_prefill=True))
    assert cache.allocate("a", 7) and cache.allocate("b", 5)
    ids = ["a"] * 3 + ["b"] * 2
    positions = [4, 5, 6, 2, 3]
    depth_sets = [[depth] * len(ids) for depth in range(4)]
    batches = cache._prepare_batches(ids, depth_sets, positions, packed_prefill=packed)
    for depths, batch in zip(depth_sets, batches):
        single = cache._prepare_batch(ids, depths, positions, packed_prefill=packed)
        assert batch.rows == single.rows and batch.allocations == single.allocations
        assert batch.max_seqlen_q == single.max_seqlen_q
        for name in (
            "position_ids",
            "write_blocks",
            "write_offsets",
            "block_tables",
            "context_lengths",
            "cu_seqlens_q",
        ):
            ours, reference = getattr(batch, name), getattr(single, name)
            assert (ours is None) == (reference is None)
            if ours is not None:
                assert torch.equal(ours, reference), name
    # Each depth addresses its own plane; depth-independent metadata is one copy.
    assert len({tuple(batch.write_blocks.tolist()) for batch in batches}) == 4
    for name in ("position_ids", "write_offsets", "context_lengths"):
        assert len({getattr(batch, name).data_ptr() for batch in batches}) == 1
    with pytest.raises(ValueError, match="depth"):
        cache._prepare_batches(ids, [[0] * 5, [4] * 5], positions)
    with pytest.raises(ValueError, match="equal lengths"):
        cache._prepare_batches(ids, [[0] * 5, [1] * 4], positions)
