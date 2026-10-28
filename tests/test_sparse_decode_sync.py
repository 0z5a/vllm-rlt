"""Ordinary and synchronous block decode through the shared sparse contract."""

import pytest
import torch

from tests.helpers import tiny_ouro_model
from vllm_rlt import (
    CacheConfig,
    ExecutionConfig,
    SamplingParams,
    SchedulerConfig,
    SparseDecodeAttentionConfig,
    SpeculativeConfig,
)
from vllm_rlt.attention import BackendCapabilities, create_backend
from vllm_rlt.attention.sparse import CAPTURE, CONSUME, DENSE
from vllm_rlt.engine.stage_engine import StageEngine


def engine(*, draft=None, retention=1.0, device="cpu", dtype=torch.float32, **options):
    values = dict(
        attention_backend="triton" if device == "cuda" else "torch",
        cache_config=CacheConfig(512, 2, incremental_allocation=True),
        scheduler_config=SchedulerConfig(
            max_num_seqs=4, max_num_batched_tokens=16, prefill_chunk_size=3
        ),
        execution_config=ExecutionConfig(static_buffers=True, pad_to_power_of_two=True),
        speculative_config=None if draft is None else SpeculativeConfig(2, draft_loops=draft),
        sparse_decode_attention_config=(
            None
            if retention is None
            else SparseDecodeAttentionConfig(enabled=True, key_retention=retention)
        ),
    )
    values.update(options)
    return StageEngine(tiny_ouro_model(device, dtype), **values)


def generate(e, count=3):
    for i in range(count):
        e.add_request(
            str(i), [2, 3, 4, 5][: 2 + i % 3], SamplingParams(max_tokens=8, ignore_eos=True)
        )
    outputs = {}
    for _ in range(1000):
        if not e.has_unfinished_requests():
            break
        for output in e.step():
            if output.finished:
                outputs[output.request_id] = list(output.token_ids)
    assert len(outputs) == count
    assert all(len(tokens) == 8 for tokens in outputs.values())
    assert e.cache_manager.num_used_blocks == 0
    if e.reference_store is not None:
        assert e.reference_store.outstanding == 0
    return outputs


@pytest.mark.parametrize("draft", [None, 1, 2])
def test_full_retention_matches_dense_and_reclaims_slots(draft):
    dense = engine(draft=draft, retention=None)
    sparse = engine(draft=draft)
    try:
        assert generate(sparse) == generate(dense)
    finally:
        sparse.close()
        dense.close()


@pytest.mark.parametrize("draft", [None, 1, 2])
def test_plans_follow_tokens_and_keep_prompt_dense(draft, monkeypatch):
    e = engine(draft=draft, retention=0.5)
    seen = []
    original = e.cache_manager._attend_prepared

    def attend(layer, kv, q):
        plan = getattr(kv, "reference_plan", None)
        if plan is not None:
            assert len(plan.modes) == len(q)
            for row, mode in enumerate(plan.modes):
                if mode == DENSE:
                    continue
                allocation, depth, position = kv.rows[row]
                request_id = next(rid for rid, owner in kv.allocations if owner is allocation)
                assert position >= len(e.scheduler.requests[request_id].prompt_token_ids)
                assert mode == (CAPTURE if depth == 1 else CONSUME)
                assert int(e.reference_store.position[plan.slots[row]]) == position
                seen.append((mode, position, plan.slots[row]))
        return original(layer, kv, q)

    monkeypatch.setattr(e.cache_manager, "_attend_prepared", attend)
    try:
        generate(e)
        assert {mode for mode, _, _ in seen} == {CAPTURE, CONSUME}
    finally:
        e.close()


@pytest.mark.parametrize("draft", [None, 1, 2])
def test_cancel_and_request_id_reuse_cannot_reuse_old_reference(draft):
    e = engine(draft=draft, retention=0.5)
    try:
        for _ in range(2):
            e.add_request("r", [2, 3, 4], SamplingParams(max_tokens=32, ignore_eos=True))
            for _ in range(100):
                e.step()
                if e.reference_store.outstanding or (
                    draft is not None and e.speculative_runner.stats.rounds
                ):
                    break
            e.abort_request("r")
            assert e.reference_store.outstanding == 0
            assert e.cache_manager.num_used_blocks == 0
    finally:
        e.close()


def test_context_bound_is_enforced_before_request_admission():
    e = engine(
        sparse_decode_attention_config=SparseDecodeAttentionConfig(
            enabled=True, max_context_tokens=8
        )
    )
    try:
        with pytest.raises(ValueError, match="sparse.*context"):
            e.add_request("too-long", [2, 3, 4], SamplingParams(max_tokens=8))
        assert "too-long" not in e.scheduler.requests
    finally:
        e.close()


@pytest.mark.parametrize("setting", ["graphs", "pipeline", "preemption", "shared"])
def test_unqualified_execution_modes_fail_explicitly(setting):
    options = {
        "graphs": dict(execution_config=ExecutionConfig(cuda_graphs=True)),
        "pipeline": dict(speculative_config=SpeculativeConfig(2, mode="pipeline")),
        "preemption": dict(scheduler_config=SchedulerConfig(enable_preemption=True)),
        "shared": dict(cache_config=CacheConfig(512, 2, layout="shared")),
    }[setting]
    with pytest.raises(ValueError, match="sparse decode attention"):
        engine(**options)


def test_backend_explicitly_declares_sparse_reference_support():
    selected = create_backend(
        "auto",
        torch.device("cpu"),
        torch.float32,
        8,
        2,
        required_capabilities=BackendCapabilities(sparse_decode=True),
    )
    assert selected.capabilities.sparse_decode


@pytest.mark.parametrize("draft", [1, 2])
@pytest.mark.parametrize("accepted", [0, 1, 2])
def test_sync_rejection_retires_all_references_before_replacement(draft, accepted, monkeypatch):
    e = engine(draft=draft, retention=0.5)
    try:
        e.add_request("r", [2, 3, 4], SamplingParams(max_tokens=16, ignore_eos=True))
        while not e.step():
            pass
        request = e.scheduler.requests["r"]
        start = request.position
        original = e.speculative_runner._coda
        calls = 0

        def controlled(hidden):
            nonlocal calls
            logits = torch.full((len(hidden), 64), -100.0)
            if calls < 2:
                logits[:, 10 + calls] = 100
            else:
                for row in range(3):
                    logits[row, 10 + row if row < accepted else 20 + row] = 100
            calls += 1
            return logits

        seen = []
        build = e.reference_store.build_plan

        def record(*args, **kwargs):
            plan = build(*args, **kwargs)
            seen.append(plan)
            return plan

        monkeypatch.setattr(e.reference_store, "build_plan", record)
        monkeypatch.setattr(e.speculative_runner, "_coda", controlled)
        output = e.step()[0]
        expected = [10 + row if row < accepted else 20 + row for row in range(accepted + 1)]
        assert output.token_ids[-(accepted + 1) :] == expected
        assert request.position == start + accepted + 1
        assert e.reference_store.outstanding == 0
        old = seen[0]
        with pytest.raises(RuntimeError, match="stale"):
            e.reference_store.read_reference(old, 0, 0)
        monkeypatch.setattr(e.speculative_runner, "_coda", original)
        e.step()
        assert seen[-1].incarnations[0] > old.incarnations[0]
        assert e.speculative_runner.stats.accepted_tokens >= accepted
    finally:
        e.close()


@pytest.mark.parametrize("draft", [None, 1, 2])
def test_failed_attention_releases_lease_and_allows_fresh_request(draft, monkeypatch):
    e = engine(draft=draft, retention=0.5)
    original = e.cache_manager._attend_prepared

    def fail(layer, kv, q):
        if getattr(kv, "reference_plan", None) is not None:
            raise RuntimeError("injected attention failure")
        return original(layer, kv, q)

    monkeypatch.setattr(e.cache_manager, "_attend_prepared", fail)
    try:
        e.add_request("r", [2, 3], SamplingParams(max_tokens=16, ignore_eos=True))
        with pytest.raises(RuntimeError, match="injected"):
            for _ in range(100):
                e.step()
        assert e.reference_store.outstanding == 0
        assert "r" not in e.scheduler.requests
        assert e.cache_manager.num_used_blocks == 0
        monkeypatch.setattr(e.cache_manager, "_attend_prepared", original)
        generate(e, count=1)
    finally:
        e.close()


@pytest.mark.gpu
@pytest.mark.parametrize("draft", [None, 1, 2])
def test_cuda_sparse_sync_executes_and_drains(draft):
    e = engine(draft=draft, retention=0.5, device="cuda", dtype=torch.bfloat16)
    try:
        generate(e)
        torch.cuda.synchronize()
        assert not bool(e.reference_store.counts.any())
    finally:
        e.close()


@pytest.mark.gpu
def test_cuda_ordinary_retirement_survives_boundary_allocator_reuse(monkeypatch):
    e = engine(device="cuda", dtype=torch.bfloat16)
    try:
        e.add_request("r", [2, 3], SamplingParams(max_tokens=16, ignore_eos=True))
        for _ in range(100):
            e.step()
            if e.reference_store.outstanding:
                break
        store = e.reference_store
        assert store.outstanding == 1
        slot = e.model_runner.attention_references["r"][2].slots[0]
        assert slot != 0
        torch.cuda.synchronize()
        store.counts[0] = 77
        torch.cuda.synchronize()
        clear = store._clear
        pointers = []

        def record(indices):
            assert torch.cuda.current_stream() == e.model_runner.core_stream
            pointers.append(indices.data_ptr())
            clear(indices)

        monkeypatch.setattr(store, "_clear", record)
        # Hold core while boundary allocations try to reuse any released index
        # storage. Correct stream ownership prevents that reuse before clearing.
        with torch.cuda.stream(e.model_runner.core_stream):
            torch.cuda._sleep(20_000_000)
        with torch.cuda.stream(e.model_runner.boundary_stream):
            e.model_runner._release_attention_reference("r")
            poison = [torch.zeros(1, device="cuda", dtype=torch.int64) for _ in range(32)]
        torch.cuda.synchronize()
        assert pointers and len(poison) == 32
        assert not bool(store.counts[slot].any())
        assert bool((store.counts[0] == 77).all())
    finally:
        e.close()


@pytest.mark.gpu
def test_cuda_close_drains_retirement_of_an_active_reference():
    e = engine(device="cuda", dtype=torch.bfloat16)
    e.add_request("r", [2, 3], SamplingParams(max_tokens=16, ignore_eos=True))
    for _ in range(100):
        e.step()
        if e.reference_store.outstanding:
            break
    assert e.reference_store.outstanding == 1
    e.close()
    assert e.reference_store.outstanding == 0
    assert e.cache_manager.num_used_blocks == 0
    assert e.model_runner.core_stream.query()
