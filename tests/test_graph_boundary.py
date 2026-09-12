"""CPU checks for the capture boundary; kernel substitutes do not qualify replay."""

import pytest
import test_recurrent_graph as fixtures
import torch
from test_recurrent_graph import fake_runtime, make_cache, prefixes

model = fixtures.model
pytestmark = pytest.mark.usefixtures("forbid_cuda")


def prepare(cache, ids=("a",), depths=(0,), positions=(0,), *, row_count=4):
    storage = cache._allocate_metadata_storage(row_count)
    host = cache._prepare_host_batch(ids, depths, positions)
    ticket = cache._begin_decode_traversal(host)
    batch = cache._prepare_into(storage, host)
    cache._bind_decode_traversal(ticket, batch)
    return storage, batch, ticket


def test_commit_waits_for_completion_and_preserves_sparse_and_already_written_positions(model):
    cache = make_cache(model)
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


def test_commit_revalidates_all_owners_before_first_prefix_mutation(model):
    cache = make_cache(model)
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


@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt])
def test_partial_host_commit_failure_quarantines_without_a_rollback_claim(
    model, monkeypatch, error_type
):
    cache = make_cache(model)
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


def test_tensor_body_uses_only_device_metadata_and_zeroes_inactive_rows(model, monkeypatch):
    cache = make_cache(model)
    fake_runtime(monkeypatch, cache)
    storage = cache._allocate_metadata_storage(4)
    batch = cache._prepare_into(storage, cache._prepare_host_batch([], [], []))
    view = cache._make_tensor_decode_view(storage)
    original = cache.key_cache.clone(), cache.value_cache.clone()

    def forbidden(*args, **kwargs):
        pytest.fail("tensor body read host ownership or prefix state")

    for name in (
        "_require_live_batch",
        "_require_prefix",
        "_require_usable",
        "_get_allocation",
        "_write_prepared",
        "_attend_prepared",
    ):
        monkeypatch.setattr(cache, name, forbidden)
    hidden, gates = model._recurrent_tensor(
        torch.full((4, model.config.hidden_size), torch.nan), view
    )
    for value in (hidden, gates):
        assert torch.equal(value, torch.zeros_like(value)) and not torch.signbit(value).any()
    assert torch.equal(cache.key_cache, original[0]) and torch.equal(cache.value_cache, original[1])
    assert batch.live_rows == ()
