"""CPU checks for the capture boundary; kernel substitutes do not qualify replay."""

import sys
from dataclasses import fields, replace
from types import SimpleNamespace

import pytest
import torch

import vllm_lt.core.kv_cache_manager as kv_module
from vllm_lt.core.kv_cache_manager import KVCacheManager, _TensorKVView
from vllm_lt.kernels.paged_attention import torch_paged_attention
from vllm_lt.models import OuroConfig, OuroForCausalLM


@pytest.fixture(autouse=True)
def no_cuda(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("graph-boundary CPU tests must not discover or initialize CUDA")

    for name in ("is_available", "device_count", "current_device", "init", "_lazy_init"):
        monkeypatch.setattr(torch.cuda, name, forbidden)


def make_cache(config=None):
    config = config or OuroConfig.tiny()
    cache = KVCacheManager(
        config.num_hidden_layers,
        config.num_key_value_heads,
        config.head_dim,
        160,
        16,
        config.total_ut_steps,
    )
    cache.key_cache.fill_(71)
    cache.value_cache.fill_(-83)
    return cache


def prepare(cache, ids=("a",), depths=(0,), positions=(0,), *, row_count=4):
    storage = cache._allocate_metadata_storage(row_count)
    host = cache._prepare_host_batch(ids, depths, positions)
    ticket = cache._begin_decode_traversal(host)
    batch = cache._prepare_into(storage, host)
    cache._bind_decode_traversal(ticket, batch)
    return storage, batch, ticket


def prefixes(cache):
    return {
        name: tuple(tuple((p.prefix, frozenset(p.pending)) for p in depth) for depth in a.written)
        for name, a in cache._allocations.items()
    }


@pytest.mark.parametrize("row_count", [4, 8])
def test_fixed_capacity_is_derived_from_storage_and_preserves_odd_slots(row_count):
    cache = make_cache()
    ids = [f"r{i}" for i in range(row_count // 2)]
    for name in ids:
        cache.allocate(name, 2)
    storage = cache._allocate_metadata_storage(row_count)
    original_ptrs = {name: tensor.data_ptr() for name, tensor in storage.tensors.items()}
    assert sum(t.numel() * t.element_size() for t in storage.staging.values()) == 157 * row_count
    host = cache._prepare_host_batch(ids, [0] * len(ids), [0] * len(ids))
    batch = cache._prepare_into(storage, host)
    assert batch.row_count == row_count and batch.live_rows == tuple(range(1, row_count, 2))
    assert batch.active.tolist() == [False, True] * (row_count // 2)
    assert batch.block_tables.shape == (row_count, 32)
    assert batch.write_blocks[::2].tolist() == [-1] * (row_count // 2)
    assert batch.context_lengths.tolist() == [0, 1] * (row_count // 2)
    cache._release_prepared(batch)
    shorter = cache._prepare_into(storage, cache._prepare_host_batch(ids[:1], [0], [0]))
    assert shorter.live_rows == (1,) and shorter.generation == 2
    assert shorter.active.tolist() == [False, True] + [False] * (row_count - 2)
    assert {name: t.data_ptr() for name, t in storage.tensors.items()} == original_ptrs
    cache._release_prepared(shorter)
    extra = ids + ["extra"]
    cache.allocate("extra", 2)
    before = {name: value.clone() for name, value in storage.tensors.items()}
    with pytest.raises(ValueError, match="capacity"):
        cache._prepare_into(
            storage, cache._prepare_host_batch(extra, [0] * len(extra), [0] * len(extra))
        )
    assert not storage.in_use and storage.generation == 2
    assert all(torch.equal(value, storage.tensors[name]) for name, value in before.items())


@pytest.mark.parametrize("row_count", [0, 2, 16, True, 4.0])
def test_only_declared_storage_capacities_are_allocated(row_count):
    cache = make_cache()
    with pytest.raises(ValueError, match="4 or 8"):
        cache._allocate_metadata_storage(row_count)
    assert cache.num_free_blocks == cache.num_blocks


def test_four_row_history_limit_is_512_without_changing_default_eight_rows():
    cache = make_cache()
    cache.allocate("long", 513)
    storage = cache._allocate_metadata_storage(4)
    batch = cache._prepare_into(storage, cache._prepare_host_batch(["long"], [3], [511]))
    assert batch.context_lengths.tolist() == [0, 512, 0, 0]
    assert batch.write_offsets.tolist() == [-1, 15, -1, -1]
    cache._release_prepared(batch)
    with pytest.raises(ValueError, match="capacity"):
        cache._prepare_into(storage, cache._prepare_host_batch(["long"], [3], [512]))
    assert cache._allocate_metadata_storage().tensors["active"].shape == (8,)


def test_begin_checks_last_row_and_last_layer_before_any_copy_or_prefix_change(monkeypatch):
    cache = make_cache()
    for name in ("a", "b"):
        cache.allocate(name, 4)
        for tracker in cache._allocations[name].written[0]:
            tracker.add(0)
    cache._allocations["b"].written[0][-1].prefix = 0
    before = prefixes(cache)
    host = cache._prepare_host_batch(["a", "b"], [0, 0], [1, 1])

    def forbidden(*args, **kwargs):
        raise AssertionError("begin must not allocate or copy device metadata")

    monkeypatch.setattr(torch, "empty", forbidden)
    monkeypatch.setattr(torch.Tensor, "copy_", forbidden)
    with pytest.raises(RuntimeError, match="uninitialized KV history"):
        cache._begin_decode_traversal(host)
    assert prefixes(cache) == before


def test_begin_rejects_readonly_empty_duplicate_and_foreign_host():
    cache = make_cache()
    cache.allocate("a", 2)
    host = cache._prepare_host_batch(["a"], [0], [0])
    for invalid in (
        replace(host, writable=False),
        replace(host, rows=()),
        replace(host, owner=make_cache()),
        replace(host, rows=host.rows * 2, addresses=host.addresses * 2),
    ):
        with pytest.raises(ValueError):
            cache._begin_decode_traversal(invalid)


def test_commit_waits_for_completion_and_preserves_sparse_and_already_written_positions():
    cache = make_cache()
    cache.allocate("a", 4)
    for tracker in cache._allocations["a"].written[0]:
        tracker.add(0)
        tracker.add(2)
    before = prefixes(cache)
    storage, batch, ticket = prepare(cache, positions=(1,))
    assert prefixes(cache) == before and ticket.state == "bound"
    with pytest.raises(RuntimeError, match="finish before releasing"):
        cache._release_prepared(batch)
    with pytest.raises(RuntimeError, match="confirmed"):
        cache._commit_decode_traversal(ticket, completion_confirmed=False)
    assert prefixes(cache) == before
    cache._commit_decode_traversal(ticket, completion_confirmed=True)
    assert all(p.prefix == 3 and not p.pending for p in cache._allocations["a"].written[0])
    with pytest.raises(RuntimeError, match="already finished"):
        cache._commit_decode_traversal(ticket, completion_confirmed=True)
    cache._release_prepared(batch)
    assert not storage.in_use
    _, rewrite, again = prepare(cache, positions=(1,))
    cache._commit_decode_traversal(again, completion_confirmed=True)
    assert all(p.prefix == 3 for p in cache._allocations["a"].written[0])
    cache._release_prepared(rewrite)


def test_new_generation_binding_rejects_other_bucket_and_wrong_host():
    cache = make_cache()
    cache.allocate("a", 2)
    host = cache._prepare_host_batch(["a"], [0], [0])
    storage = cache._allocate_metadata_storage(4)
    old = cache._prepare_into(storage, host)
    cache._release_prepared(old)
    ticket = cache._begin_decode_traversal(host)
    batch = cache._prepare_into(storage, host)
    with pytest.raises(RuntimeError, match="generation"):
        cache._bind_decode_traversal(ticket, old)
    assert ticket.state == "begun"
    other = cache._prepare_into(cache._allocate_metadata_storage(8), host)
    cache._bind_decode_traversal(ticket, batch)
    assert ticket.batch.generation == 2
    with pytest.raises(RuntimeError, match="already bound"):
        cache._bind_decode_traversal(ticket, other)
    other_ticket = cache._begin_decode_traversal(cache._prepare_host_batch(["a"], [0], [0]))
    with pytest.raises(ValueError, match="own newly prepared"):
        cache._bind_decode_traversal(other_ticket, other)
    cache._cancel_decode_traversal(ticket)
    cache._release_prepared(batch)
    cache._release_prepared(other)
    assert all(p.prefix == 0 for p in cache._allocations["a"].written[0])


def test_commit_revalidates_all_owners_before_first_prefix_mutation():
    cache = make_cache()
    for name in ("a", "b"):
        cache.allocate(name, 2)
    storage, _, ticket = prepare(cache, ids=("a", "b"), depths=(0, 0), positions=(0, 0))
    old_b = cache._allocations["b"]
    cache.free("b")
    cache.allocate("b", 2)
    with pytest.raises(RuntimeError, match="stale"):
        cache._commit_decode_traversal(ticket, completion_confirmed=True)
    assert all(p.prefix == 0 for p in cache._allocations["a"].written[0])
    assert all(p.prefix == 0 for p in old_b.written[0])
    assert storage.failed and ticket.state == "failed"
    with pytest.raises(RuntimeError, match="quarantined"):
        cache.free("a")


def test_stale_generation_commit_cannot_publish_or_release_another_lease():
    cache = make_cache()
    cache.allocate("a", 2)
    storage, _, ticket = prepare(cache)
    # Simulate a broken caller that replaced the lease before completion.
    storage.generation += 1
    with pytest.raises(RuntimeError, match="generation"):
        cache._commit_decode_traversal(ticket, completion_confirmed=True)
    assert storage.in_use and storage.failed
    with pytest.raises(RuntimeError, match="another lease"):
        cache._abort_decode_traversal(ticket, completion_confirmed=True)
    assert storage.in_use
    assert all(p.prefix == 0 for p in cache._allocations["a"].written[0])


@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt])
def test_partial_host_commit_failure_quarantines_without_a_rollback_claim(monkeypatch, error_type):
    cache = make_cache()
    for name in ("a", "b"):
        cache.allocate(name, 2)
    storage, batch, ticket = prepare(cache, ids=("a", "b"), depths=(0, 0), positions=(0, 0))

    def fail(position):
        raise error_type("injected later tracker failure")

    monkeypatch.setattr(cache._allocations["b"].written[0][0], "add", fail)
    with pytest.raises(error_type, match="later tracker"):
        cache._commit_decode_traversal(ticket, completion_confirmed=True)
    assert all(p.prefix == 1 for p in cache._allocations["a"].written[0])
    assert cache._allocations["b"].written[0][0].prefix == 0
    assert ticket.state == "failed" and storage.failed
    cache._abort_decode_traversal(ticket, completion_confirmed=True)
    assert not storage.in_use and storage.transaction is None
    for operation in (
        lambda: cache.allocate("new", 1),
        lambda: cache.free("a"),
        lambda: cache._release_prepared(batch),
    ):
        with pytest.raises(RuntimeError, match="quarantined"):
            operation()


@pytest.mark.parametrize("confirmed", [False, True])
def test_abort_requires_completion_before_releasing_failed_storage(confirmed):
    cache = make_cache()
    cache.allocate("a", 2)
    storage, _, ticket = prepare(cache)
    before = prefixes(cache)
    cache._abort_decode_traversal(ticket, completion_confirmed=confirmed)
    assert prefixes(cache) == before
    assert storage.failed and ticket.state == "failed"
    assert storage.in_use is not confirmed
    if confirmed:
        cache.free("a")
        assert cache.num_used_blocks == 0
    else:
        with pytest.raises(RuntimeError, match="quarantined"):
            cache.free("a")


def test_failure_after_commit_preserves_committed_history():
    cache = make_cache()
    cache.allocate("a", 2)
    storage, _, ticket = prepare(cache)
    cache._commit_decode_traversal(ticket, completion_confirmed=True)
    before = prefixes(cache)
    # The caller's later routing/update failure must not rewind completed work.
    cache._abort_decode_traversal(ticket, completion_confirmed=True)
    assert prefixes(cache) == before
    assert all(p.prefix == 1 for p in cache._allocations["a"].written[0])
    assert storage.failed and not storage.in_use


def _cpu_scatter(keys, values, blocks, offsets, k, v, active):
    live = active.nonzero().flatten()
    keys[blocks[live], offsets[live]] = k[live]
    values[blocks[live], offsets[live]] = v[live]


def cpu_tensor_view(cache, storage, monkeypatch):
    # Substitute only raw kernels; the production view stays Triton-only.
    monkeypatch.setitem(
        sys.modules,
        "vllm_lt.kernels.triton_kv_write",
        SimpleNamespace(masked_kv_write=_cpu_scatter),
    )
    monkeypatch.setattr(kv_module, "triton_paged_attention", torch_paged_attention)
    monkeypatch.setattr(cache, "backend", "triton")
    return cache._make_tensor_decode_view(storage)


@pytest.mark.parametrize("row_count", [4, 8])
def test_shared_tensor_arithmetic_has_no_host_lookup_and_tracks_dynamic_metadata(
    monkeypatch, row_count
):
    torch.manual_seed(71)
    model = OuroForCausalLM(OuroConfig.tiny())
    checked, tensor_cache = make_cache(model.config), make_cache(model.config)
    for cache in (checked, tensor_cache):
        for name in ("a", "b"):
            cache.allocate(name, 3)
    storage = tensor_cache._allocate_metadata_storage(row_count)
    view = cpu_tensor_view(tensor_cache, storage, monkeypatch)
    assert isinstance(view, _TensorKVView)
    assert not any(
        field.name in {"owner", "rows", "allocations", "storage", "generation"}
        for field in fields(view)
    )
    ptrs = {name: getattr(view, name).data_ptr() for name in storage.tensors}
    for ids, depths, positions in [(["a", "b"], [0, 1], [0, 0]), (["b"], [1], [1])]:
        host = tensor_cache._prepare_host_batch(ids, depths, positions)
        ticket = tensor_cache._begin_decode_traversal(host)
        batch = tensor_cache._prepare_into(storage, host)
        tensor_cache._bind_decode_traversal(ticket, batch)
        plain = checked._pad_prepared(
            checked._prepare_batch(ids, depths, positions),
            row_indices=batch.live_rows,
            row_count=row_count,
            table_width=32,
        )
        hidden = torch.full((row_count, model.config.hidden_size), torch.nan)
        hidden[list(batch.live_rows)] = torch.randn(len(ids), model.config.hidden_size)
        expected = model._recurrent_prepared(hidden, plain, checked)
        before = prefixes(tensor_cache)

        def forbidden(*args, **kwargs):
            raise AssertionError("tensor body read host cache state")

        with monkeypatch.context() as patch:
            for name in (
                "_require_live_batch",
                "_require_prefix",
                "_require_usable",
                "_get_allocation",
                "_validate_tensor",
                "_write_prepared",
                "_attend_prepared",
            ):
                patch.setattr(tensor_cache, name, forbidden)
            actual = model._recurrent_tensor(hidden, view)
        assert prefixes(tensor_cache) == before
        assert all(torch.equal(a, b) for a, b in zip(actual, expected))
        assert all(torch.isfinite(value).all() for value in actual)
        assert all(torch.count_nonzero(value[~view.active]) == 0 for value in actual)
        assert torch.equal(checked.key_cache, tensor_cache.key_cache)
        assert torch.equal(checked.value_cache, tensor_cache.value_cache)
        tensor_cache._commit_decode_traversal(ticket, completion_confirmed=True)
        assert prefixes(checked) == prefixes(tensor_cache)
        tensor_cache._release_prepared(batch)
    assert {name: getattr(view, name).data_ptr() for name in storage.tensors} == ptrs


def test_tensor_view_requires_explicit_triton_and_legacy_empty_does_not_enter_body(monkeypatch):
    model = OuroForCausalLM(OuroConfig.tiny())
    cache = make_cache(model.config)
    with pytest.raises(ValueError, match="Triton"):
        cache._make_tensor_decode_view(cache._allocate_metadata_storage())
    batch = cache._pad_prepared(
        cache._prepare_batch([], [], []), row_indices=[], row_count=4, table_width=32
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("legacy empty traversal must retain its early return")

    monkeypatch.setattr(model, "_recurrent_body", forbidden)
    hidden, gates = model._recurrent_prepared(
        torch.full((4, model.config.hidden_size), torch.nan), batch, cache
    )
    assert torch.equal(hidden, torch.zeros_like(hidden)) and torch.equal(
        gates, torch.zeros_like(gates)
    )
    with pytest.raises(ValueError, match="shape"):
        model._recurrent_prepared(torch.empty(3, model.config.hidden_size), batch, cache)


@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt])
def test_partial_tensor_failure_does_not_publish_prefixes(monkeypatch, error_type):
    model = OuroForCausalLM(OuroConfig.tiny())
    cache = make_cache(model.config)
    cache.allocate("a", 2)
    storage, batch, ticket = prepare(cache)
    view = cpu_tensor_view(cache, storage, monkeypatch)
    calls = 0

    def fail_second_attention(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise error_type("injected partial tensor body failure")
        return torch_paged_attention(*args, **kwargs)

    view = replace(view, attention_kernel=fail_second_attention)
    hidden = torch.full((4, model.config.hidden_size), torch.nan)
    hidden[1] = model.prelude(torch.tensor([3]))[0]
    before = prefixes(cache)
    initial_keys = cache.key_cache.clone()
    with pytest.raises(error_type, match="partial tensor body"):
        model._recurrent_tensor(hidden, view)
    assert calls == 2 and not torch.equal(cache.key_cache, initial_keys)
    assert prefixes(cache) == before and ticket.state == "bound"
    cache._abort_decode_traversal(ticket, completion_confirmed=True)
    assert prefixes(cache) == before and storage.failed
    with pytest.raises(RuntimeError, match="generation"):
        cache._release_prepared(batch)
    cache.free("a")
    assert cache.num_used_blocks == 0


def test_tensor_mask_all_inactive_runs_fixed_body_without_touching_cache(monkeypatch):
    model = OuroForCausalLM(OuroConfig.tiny())
    cache = make_cache(model.config)
    storage = cache._allocate_metadata_storage(4)
    batch = cache._prepare_into(storage, cache._prepare_host_batch([], [], []))
    view = cpu_tensor_view(cache, storage, monkeypatch)
    original = cache.key_cache.clone(), cache.value_cache.clone()
    calls = []

    def record_scatter(*args):
        calls.append("write")
        return _cpu_scatter(*args)

    view = replace(view, write_kernel=record_scatter)
    hidden, gates = model._recurrent_tensor(
        torch.full((4, model.config.hidden_size), torch.nan), view
    )
    assert len(calls) == model.config.num_hidden_layers
    assert torch.equal(hidden, torch.zeros_like(hidden))
    assert torch.equal(gates, torch.zeros_like(gates))
    assert not torch.signbit(hidden).any() and not torch.signbit(gates).any()
    assert torch.equal(cache.key_cache, original[0])
    assert torch.equal(cache.value_cache, original[1])
    cache._release_prepared(batch)
