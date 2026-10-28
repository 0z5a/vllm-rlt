"""Sparse decode attention: reference lifecycle and plan contracts (RFC #156).

Covers the asynchronous self-speculation cell of the sparse decode attention
column: one reference slot per live token, plan modes for the draft depths,
GPU acceptance driving retirement of rejected candidates and their dependent
descendants, same-position replacement through incarnations, delayed
completion and reference-slot reuse.
"""

import math

import pytest
import torch

from tests.helpers import bootstrap_speculation, make_speculative_engine, tiny_ouro_model
from vllm_rlt import SamplingParams, SparseDecodeAttentionConfig
from vllm_rlt.attention.sparse import (
    CAPTURE,
    CONSUME,
    DENSE,
    REFERENCE_DEPTH,
    SparseDecodePlan,
    reference_mode,
    reference_statistics,
    retained_count,
    sparse_correction,
)
from vllm_rlt.worker.attention_reference import AttentionReferenceStore
from vllm_rlt.worker.metadata import MetadataBuilder

tiny = tiny_ouro_model
bootstrap = bootstrap_speculation


def drain(engine):
    outputs = {}
    for _ in range(1000):
        if not engine.has_unfinished_requests():
            return outputs
        for output in engine.step():
            outputs[output.request_id] = output
    pytest.fail("engine failed to drain")


def make_store(**overrides):
    values = dict(slots=6, layers=2, heads=2, head_dim=4, key_bucket=8, device="cpu")
    values.update(overrides)
    return AttentionReferenceStore(**values)


def fill_payload(store, slots, value=1.0):
    for slot in slots:
        store.mass[slot] = value
        store.output[slot] = value
        store.contribution[slot] = value
        store.selected[slot] = 1
        store.counts[slot] = 1


# ---------------------------------------------------------------------------
# Plan and arithmetic contracts (RFC #156 sections 3.2 and 6.3)
# ---------------------------------------------------------------------------


def test_reference_mode_follows_reference_depth():
    # Loop 2 (depth index 1) captures; later loops consume; loop 1 stays dense.
    assert reference_mode(0) == DENSE
    assert reference_mode(REFERENCE_DEPTH) == CAPTURE
    assert reference_mode(2) == CONSUME
    assert reference_mode(3) == CONSUME


def test_plan_groups_rows_by_mode():
    plan = SparseDecodePlan(
        modes=(DENSE, CAPTURE, CAPTURE, CONSUME),
        slots=(-1, 3, 4, 5),
        incarnations=(0, 1, 1, 1),
        key_retention=0.5,
        key_bucket=None,
    )
    assert plan.groups() == {DENSE: [0], CAPTURE: [1, 2], CONSUME: [3]}
    assert not plan.dense()
    assert plan.reference_depth == REFERENCE_DEPTH


def test_retained_count_follows_flashloop_ceil_rule():
    # FlashLoop aeaadee select_decode_keys: k = min(N, max(1, ceil(N * r))).
    assert retained_count(13, 0.1) == 2
    assert retained_count(10, 0.1) == 1
    assert retained_count(10, 0.5) == 5
    assert retained_count(9, 0.5) == 5
    assert retained_count(1, 0.1) == 1
    assert retained_count(4, 1.0) == 4
    assert retained_count(0, 0.5) == 0
    with pytest.raises(ValueError, match="key_retention"):
        retained_count(4, 1.5)


def test_reference_statistics_fail_when_capacity_cannot_hold_retained_keys():
    # Capacity never truncates the algorithm: a count above it fails loudly.
    q = torch.ones(1, 1)
    keys = torch.log(torch.tensor([0.2, 0.3, 0.5])).reshape(3, 1, 1)
    values = torch.ones(3, 1, 1)
    with pytest.raises(ValueError, match="cannot hold"):
        reference_statistics(q, keys, values, key_retention=0.5, key_bucket=1)
    # The same query passes with sufficient capacity.
    output, _, _, positions = reference_statistics(q, keys, values, key_retention=0.5, key_bucket=2)
    assert positions.shape[1] == 2
    assert output.shape == (1, 1)


def test_reference_statistics_match_rfc_worked_example():
    # RFC #156 section 3.2: probs [0.1, 0.3, 0.6], values [10, 20, 30];
    # full output 25; retaining the last two keys gives m=0.9, C_ref=24.
    # A later loop with local softmax [0.75, 0.25] over values [20, 40]
    # corrects to 25 - 24 + 0.9 * (0.75*20 + 0.25*40) = 23.5.
    q = torch.ones(1, 1)
    keys = torch.log(torch.tensor([0.1, 0.3, 0.6])).reshape(3, 1, 1)
    values = torch.tensor([10.0, 20.0, 30.0]).reshape(3, 1, 1)
    output, contribution, mass, positions = reference_statistics(q, keys, values, key_retention=0.5)
    assert output.item() == pytest.approx(25.0)
    assert mass.item() == pytest.approx(0.9)
    assert contribution.item() == pytest.approx(24.0)
    assert positions.tolist() == [[1, 2]]

    later_keys = torch.log(torch.tensor([3.0, 1.0])).reshape(1, 2, 1)
    later_values = torch.tensor([20.0, 40.0]).reshape(1, 2, 1)
    corrected = sparse_correction(q, later_keys, later_values, mass, output, contribution)
    assert corrected.item() == pytest.approx(23.5)


def test_full_retention_matches_dense_attention():
    torch.manual_seed(7)
    length, kv_heads, heads, dim = 9, 2, 4, 8
    q = torch.randn(heads, dim)
    keys = torch.randn(length, kv_heads, dim)
    values = torch.randn(length, kv_heads, dim)
    output, contribution, mass, positions = reference_statistics(q, keys, values, key_retention=1.0)
    # Full retention keeps every key, so the correction reproduces the dense
    # reference output exactly.
    dense = reference_statistics(q, keys, values, key_retention=1.0)[0]
    groups = heads // kv_heads
    corrected = sparse_correction(
        q,
        keys.permute(1, 0, 2).repeat_interleave(groups, dim=0),
        values.permute(1, 0, 2).repeat_interleave(groups, dim=0),
        mass,
        output,
        contribution,
    )
    assert positions.shape[1] == length
    assert torch.allclose(corrected, dense, atol=1e-5)
    assert torch.allclose(output, dense, atol=1e-5)


# ---------------------------------------------------------------------------
# Store lifecycle (RFC #156 section 5)
# ---------------------------------------------------------------------------


def test_reserve_uses_one_slot_per_live_token():
    store = make_store()
    lease = store.reserve("r", 0, 3, generation=1)
    assert len(lease) == 3
    assert len(set(lease.slots)) == 3
    assert lease.incarnations == (1, 1, 1)
    assert store.outstanding == 3
    assert len(store.free) == 3


def test_release_requires_owner_and_clears_payload():
    store = make_store()
    lease = store.reserve("r", 0, 2, generation=1)
    fill_payload(store, lease.slots)
    store.release(lease)
    for slot in lease.slots:
        assert store.mass[slot].sum() == 0
        assert store.output[slot].sum() == 0
        assert store.selected[slot].sum() == 0
    assert store.outstanding == 0
    # A retired lease cannot release its slots a second time.
    with pytest.raises(RuntimeError, match="different owner"):
        store.release(lease)


def test_accept_invalidates_rejected_candidates_and_descendants():
    store = make_store()
    lease = store.reserve("r", 0, 3, generation=1)
    fill_payload(store, lease.slots)
    store.accept([lease], torch.tensor([1]))
    accepted, middle, rejected = lease.slots
    # Rows 0..1 survive the accepted prefix; the rejected candidate and every
    # descendant lose the reference payload.
    assert store.mass[accepted].sum() > 0 and store.mass[middle].sum() > 0
    assert store.mass[rejected].sum() == 0
    assert store.output[rejected].sum() == 0
    assert store.selected[rejected].sum() == 0
    assert int(store.accepted[accepted]) == 1


def test_accept_keeps_root_when_every_candidate_is_rejected():
    store = make_store()
    lease = store.reserve("r", 0, 3, generation=1)
    fill_payload(store, lease.slots)
    store.accept([lease], torch.tensor([0]))
    assert store.mass[lease.slots[0]].sum() > 0
    for slot in lease.slots[1:]:
        assert store.mass[slot].sum() == 0


def test_capture_and_consume_mark_layer_state():
    store = make_store()
    lease = store.reserve("r", 0, 2, generation=1)
    plan = store.build_plan([(lease, 0), (lease, 1)], [CAPTURE, CONSUME], key_retention=0.5)
    assert not bool(store.valid[lease.slots[0], 0])
    store.capture(plan, layer=0)
    store.capture(plan, layer=1)
    assert bool(store.valid[lease.slots[0], 0]) and bool(store.valid[lease.slots[0], 1])
    assert not bool(store.valid[lease.slots[1], 0])
    store.consume(plan, layer=0)
    assert int(store.uses[lease.slots[1], 0]) == 1
    assert int(store.uses[lease.slots[0], 0]) == 0


def test_stale_plan_after_reincarnation_is_rejected():
    store = make_store()
    lease = store.reserve("r", 0, 2, generation=1)
    stale = store.build_plan([(lease, 0), (lease, 1)], [CAPTURE, CONSUME], key_retention=0.5)
    store.release(lease)
    # Same request and rows, a fresh token incarnation.
    renewed = store.reserve("r", 0, 2, generation=1)
    assert all(new > old for new, old in zip(renewed.incarnations, lease.incarnations))
    with pytest.raises(RuntimeError, match="stale"):
        store.capture(stale, layer=0)
    current = store.build_plan([(renewed, 0), (renewed, 1)], [CAPTURE, CONSUME], key_retention=0.5)
    store.capture(current, layer=0)


def test_outstanding_rounds_hold_disjoint_slots():
    store = make_store(slots=4)
    first = store.reserve("a", 0, 2, generation=1)
    second = store.reserve("b", 1, 2, generation=1)
    assert not set(first.slots) & set(second.slots)
    with pytest.raises(RuntimeError, match="no retired attention reference slots"):
        store.reserve("c", 2, 1, generation=1)
    store.release(first)
    third = store.reserve("c", 2, 1, generation=1)
    assert set(third.slots) <= set(first.slots)


def test_reincarnation_bumps_on_slot_reuse():
    store = make_store()
    lease = store.reserve("r", 0, 1, generation=1)
    slot = lease.slots[0]
    store.release(lease)
    again = store.reserve("r", 0, 1, generation=1)
    assert again.slots[0] == slot
    assert again.incarnations[0] == lease.incarnations[0] + 1
    assert int(store.incarnation[slot]) == again.incarnations[0]


def test_bytes_matches_layout():
    store = make_store(slots=3, layers=2, heads=2, head_dim=4, key_bucket=8)
    expected = 3 * 2 * 2 * (8 + 1 + 1 + 4 + 4) * 4 + 3 * 2 * (1 + 4) + 3 * (8 + 8 + 4)
    assert store.bytes == expected


def test_device_metadata_carries_reference_plan():
    class StubCache:
        device = torch.device("cpu")

        def _require_live_batch(self, batch):
            return None

    builder = MetadataBuilder(StubCache())
    plan = SparseDecodePlan(
        modes=(CAPTURE, CONSUME),
        slots=(0, 1),
        incarnations=(1, 1),
        key_retention=0.5,
        key_bucket=4,
    )
    common = dict(
        allocations=(("r", object()),),
        position_ids=torch.zeros(2, dtype=torch.int64),
        write_blocks=torch.zeros(2, dtype=torch.int64),
        write_offsets=torch.zeros(2, dtype=torch.int64),
        block_tables=torch.zeros((2, 1), dtype=torch.int32),
        context_lengths=torch.zeros(2, dtype=torch.int32),
    )
    batch = builder.prepare_device(**common, reference_plan=plan)
    assert batch.reference_plan is plan
    padded = {**common, "position_ids": torch.zeros(3, dtype=torch.int64)}
    with pytest.raises(ValueError, match="one row mode per query row"):
        builder.prepare_device(**padded, reference_plan=plan)


# ---------------------------------------------------------------------------
# Async speculative integration (GPU)
# ---------------------------------------------------------------------------


@pytest.mark.gpu
def test_async_sparse_reference_plans_cover_capture_and_consume():
    model = tiny("cuda")
    e = make_speculative_engine(model, sparse=SparseDecodeAttentionConfig(enabled=True))
    store = e.reference_store
    assert store is not None
    observed = []
    original = store.build_plan

    def recording(owners, modes, **kwargs):
        plan = original(owners, modes, **kwargs)
        observed.append(plan.modes)
        return plan

    store.build_plan = recording
    bootstrap(e)
    drain(e)
    assert any(CAPTURE in modes for modes in observed)
    assert any(CONSUME in modes for modes in observed)
    # Dense steps never publish a plan.
    assert all(DENSE not in modes for modes in observed)
    assert store.outstanding == 0
    assert len(store.free) == store.slots


@pytest.mark.gpu
@pytest.mark.parametrize("accepted", [0, 1, 2])
def test_async_sparse_reference_rejection_invalidates_rejected_rows(accepted, monkeypatch):
    model = tiny("cuda")
    e = make_speculative_engine(model, sparse=SparseDecodeAttentionConfig(enabled=True))
    store = e.reference_store
    bootstrap(e)
    calls = 0

    def controlled(hidden):
        nonlocal calls
        phase = calls % 4
        calls += 1
        logits = torch.full((len(hidden), model.config.vocab_size), -100.0, device="cuda")
        if phase < 3:
            logits[:, 10 + phase] = 100
        else:
            for row in range(4):
                logits[row, 10 + row if row < accepted else 20 + row] = 100
        return logits

    monkeypatch.setattr(e.speculative_runner, "_coda", controlled)

    observed = []
    original = store.accept

    def recording(leases, accepted_counts):
        # Pretend the reference loop published state, then let the device-side
        # retirement decide which rows keep it.
        for lease in leases:
            fill_payload(store, lease.slots)
        original(leases, accepted_counts)
        torch.cuda.synchronize()
        counts = [int(a) for a in accepted_counts.tolist()]
        for lease, count in zip(leases, counts):
            observed.append(
                (
                    count,
                    [float(store.mass[slot, 0, 0]) for slot in lease.slots],
                    [int(store.accepted[slot]) for slot in lease.slots],
                )
            )

    store.accept = recording
    e.step()
    e.step()
    e._collect_speculative()
    assert observed, "acceptance-driven retirement never ran"
    for count, masses, limits in observed:
        assert count == accepted
        assert all(limit == accepted for limit in limits)
        for row, mass in enumerate(masses):
            assert mass == (1.0 if row <= accepted else 0.0)
    e.abort_request("r")
    drain(e)
    assert store.outstanding == 0
    assert len(store.free) == store.slots


@pytest.mark.gpu
def test_async_sparse_reference_incarnations_advance_across_rounds():
    model = tiny("cuda")
    e = make_speculative_engine(model, sparse=SparseDecodeAttentionConfig(enabled=True))
    store = e.reference_store
    bootstrap(e)
    incarnations = []
    original = store.build_plan

    def recording(owners, modes, **kwargs):
        plan = original(owners, modes, **kwargs)
        incarnations.append(plan.incarnations)
        return plan

    store.build_plan = recording
    e.step()
    e.step()
    e._collect_speculative()
    assert len(incarnations) >= 2
    # Every new round is a new token incarnation, even at the same position.
    assert incarnations[-1][0] > incarnations[0][0]
    e.abort_request("r")
    drain(e)
    assert store.outstanding == 0


@pytest.mark.gpu
def test_sparse_engine_sizes_key_buffer_from_retention():
    model = tiny("cuda")
    e = make_speculative_engine(
        model, sparse=SparseDecodeAttentionConfig(enabled=True, key_retention=0.25)
    )
    # None key_bucket derives the capacity from context x retention, not the
    # full model context (FlashLoop ceil rule at the worst-case key count).
    assert e.reference_store.key_bucket == math.ceil(model.config.max_position_embeddings * 0.25)


@pytest.mark.gpu
def test_sparse_engine_rejects_insufficient_key_bucket():
    model = tiny("cuda")
    with pytest.raises(ValueError, match="key_bucket"):
        make_speculative_engine(
            model,
            sparse=SparseDecodeAttentionConfig(enabled=True, key_retention=1.0, key_bucket=4),
        )


@pytest.mark.gpu
def test_sparse_engine_e2e_multiple_requests_abort_and_drain():
    # Engine-level lifecycle e2e: concurrent rounds across requests, an abort
    # in flight, and full slot reclamation without stale references.
    model = tiny("cuda")
    e = make_speculative_engine(
        model, sparse=SparseDecodeAttentionConfig(enabled=True, key_retention=0.5)
    )
    store = e.reference_store
    max_outstanding = 0
    original_reserve = store.reserve

    def counting_reserve(*args, **kwargs):
        nonlocal max_outstanding
        lease = original_reserve(*args, **kwargs)
        max_outstanding = max(max_outstanding, store.outstanding)
        return lease

    store.reserve = counting_reserve
    params = [SamplingParams(max_tokens=n, ignore_eos=True) for n in (5, 9, 50)]
    for i in range(3):
        e.add_request(str(i), [2, 3, 4], params[i])
    outputs = {}
    for _ in range(1000):
        if outputs.get("0") is not None and outputs["0"].finished:
            break
        for output in e.step():
            outputs[output.request_id] = output
    # Abort one request mid-flight; the rest must finish cleanly.
    if "2" not in outputs or not outputs["2"].finished:
        e.abort_request("2")
    outputs.update(drain(e))
    assert outputs["0"].finished and len(outputs["0"].token_ids) == 5
    assert outputs["1"].finished and len(outputs["1"].token_ids) == 9
    assert store.outstanding == 0
    assert len(store.free) == store.slots
    assert max_outstanding > 0
