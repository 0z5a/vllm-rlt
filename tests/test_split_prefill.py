"""Split prefill: staged chunks and one boundary per recurrence depth.

The scheduler trace tests retain a small mirror of the depth-frontier invariant;
runtime tests exercise the engine transition and synchronous runner path.

Stage order is an engine-side obligation, not a scheduler guard: the scheduler
trusts that a request reaches `PREFILLING_RECURRENT` only after its chunk was
staged, so a stage enqueued out of order produces no usable item rather than a
dedicated error. `test_split_prefill_steps_one_depth_per_chunk_and_keeps_the_
prompt_ordered` pins that order from admission to the last depth.
"""

import pytest
import torch

from tests.helpers import tiny_ouro_config
from vllm_rlt import CacheConfig, ExecutionConfig, ExitConfig, SamplingParams, SchedulerConfig
from vllm_rlt.config import SpeculativeConfig
from vllm_rlt.core.scheduling_policy import (
    NoRefillPolicy,
    RefillPolicy,
    SchedulingPolicy,
    SpeculativePolicy,
    SplitPrefillPolicy,
)
from vllm_rlt.engine.llm_engine import LLMEngine
from vllm_rlt.models import OuroForCausalLM
from vllm_rlt.request import Request, Stage

TOTAL_DEPTHS = 4


def model():
    torch.manual_seed(123)
    return OuroForCausalLM(tiny_ouro_config())


def engine(**kwargs):
    kwargs.setdefault("cache_config", CacheConfig(64, 2))
    kwargs.setdefault("execution_config", ExecutionConfig(atomic_prefill=True))
    kwargs.setdefault(
        "scheduler_config",
        SchedulerConfig(max_num_seqs=2, max_num_batched_tokens=4, prefill_chunk_size=2),
    )
    return LLMEngine(model(), **kwargs)


def split_engine(**kwargs):
    kwargs.setdefault("execution_config", ExecutionConfig(atomic_prefill=False))
    return engine(**kwargs)


def drain_request(e):
    for _ in range(100):
        outputs = e.step()
        if outputs and outputs[-1].finished:
            return outputs[-1]
    pytest.fail("request did not finish within bounded scheduler steps")


def run_request(e, prompt, request_id="run"):
    e.add_request(
        request_id,
        prompt,
        SamplingParams(max_tokens=3, max_loops=2, ignore_eos=True),
    )
    return drain_request(e)


def test_cpu_prefill_segments_gather_and_scatter_contiguous_spans():
    from vllm_rlt.core.scheduler import ScheduledItem, SchedulerOutput

    e = split_engine(
        scheduler_config=SchedulerConfig(
            max_num_seqs=2, max_num_batched_tokens=4, prefill_chunk_size=4
        )
    )
    e.add_request("a", [1, 2, 3, 4], SamplingParams(max_tokens=1))
    e.add_request("b", [5, 6, 7, 8], SamplingParams(max_tokens=1))
    a, b = (e.scheduler.requests[request_id] for request_id in ("a", "b"))
    batch = SchedulerOutput(
        Stage.PREFILLING_PRELUDE,
        [
            ScheduledItem(a, 1, 2, chunk_start=0),
            ScheduledItem(b, 2, 1, chunk_start=1),
        ],
    )
    prepared = e.model_runner.prepare(batch)
    slots = e.model_runner.state_slots
    metadata = prepared.prefill_metadata
    assert metadata.lease.host_segments[:2].tolist() == [
        [slots["a"], 1, 2, 0, 1],
        [slots["b"], 1, 1, 2, 1],
    ]
    assert metadata.token_ids.tolist() == [2, 3, 7]
    assert metadata.lease.host_segments.data_ptr() == metadata.lease.device_segments.data_ptr()

    try:
        hidden = torch.arange(
            3 * e.model.config.hidden_size, dtype=e.model_runner.prefill_states.dtype
        ).view(3, e.model.config.hidden_size)
        e.model_runner._scatter_prefill(hidden, prepared)
        assert torch.equal(e.model_runner._gather_prefill(prepared), hidden)
    finally:
        e.model_runner._release_prepared_metadata(prepared)
    e.close()


def test_recurrent_prepare_uses_absolute_token_start_for_kv_span():
    from vllm_rlt.core.scheduler import ScheduledItem, SchedulerOutput

    e = split_engine()
    e.add_request("a", [1, 2, 3, 4], SamplingParams(max_tokens=1))
    request = e.scheduler.requests["a"]
    assert e.cache_manager.allocate("a", 4)
    batch = SchedulerOutput(
        Stage.PREFILLING_RECURRENT,
        [ScheduledItem(request, token_start=2, token_count=1, depth=1, chunk_start=1)],
    )
    try:
        prepared = e.model_runner.prepare(batch)
        assert prepared.kv.position_ids.tolist() == [2]
        assert prepared.kv.rows[0][2] == 2
        slot = e.model_runner.state_slots["a"]
        assert prepared.prefill_metadata.lease.host_segments[0].tolist() == [slot, 1, 1, 0, 1]
    finally:
        if "prepared" in locals():
            e.model_runner._release_prepared_metadata(prepared)
        e.close()


def test_final_depth_prefill_scatter_retains_only_coda_hidden_per_item():
    from vllm_rlt.core.scheduler import ScheduledItem, SchedulerOutput

    e = split_engine(
        scheduler_config=SchedulerConfig(
            max_num_seqs=2, max_num_batched_tokens=4, prefill_chunk_size=2
        )
    )
    e.add_request("mid", [1, 2], SamplingParams(max_tokens=1))
    e.add_request("final", [3, 4], SamplingParams(max_tokens=1))
    for request_id in ("mid", "final"):
        assert e.cache_manager.allocate(request_id, 2)
    mid, final = (e.scheduler.requests[rid] for rid in ("mid", "final"))
    batch = SchedulerOutput(
        Stage.PREFILLING_RECURRENT,
        [
            ScheduledItem(mid, 0, 2, depth=1, chunk_start=0),
            ScheduledItem(final, 0, 2, depth=3, chunk_start=0, completes_prompt=True),
        ],
    )
    runner = e.model_runner
    prepared = None
    try:
        prepared = runner.prepare(batch)
        mid_slot, final_slot = runner.state_slots["mid"], runner.state_slots["final"]
        runner.prefill_states[mid_slot, :2].fill_(9)
        runner.prefill_states[final_slot, :2].fill_(7)
        hidden = torch.arange(
            4 * e.model.config.hidden_size, dtype=runner.prefill_states.dtype
        ).view(4, e.model.config.hidden_size)

        runner._scatter_prefill(hidden, prepared)
        runner._store_split_prefill_outputs(hidden, prepared)

        assert prepared.prefill_metadata.lease.host_segments[:2].tolist() == [
            [mid_slot, 0, 2, 0, 1],
            [final_slot, 0, 2, 2, 0],
        ]
        assert torch.equal(runner.prefill_states[mid_slot, :2], hidden[:2])
        assert torch.equal(runner.prefill_states[final_slot, :2], torch.full_like(hidden[2:], 7))
        assert torch.equal(final.hidden_state, hidden[-1])
        assert mid.hidden_state is None
    finally:
        if prepared is not None:
            runner._release_prepared_metadata(prepared)
        e.close()


@pytest.mark.gpu
def test_cuda_prefill_segments_gather_and_scatter_variable_spans():
    from vllm_rlt.core.scheduler import ScheduledItem, SchedulerOutput

    e = LLMEngine(
        OuroForCausalLM(tiny_ouro_config()).to("cuda", torch.bfloat16),
        cache_config=CacheConfig(64, 4),
        scheduler_config=SchedulerConfig(
            max_num_seqs=2, max_num_batched_tokens=4, prefill_chunk_size=4
        ),
        execution_config=ExecutionConfig(atomic_prefill=False),
        attention_backend="triton",
    )
    try:
        e.add_request("a", [1, 2, 3, 4], SamplingParams(max_tokens=1))
        e.add_request("b", [5, 6, 7, 8], SamplingParams(max_tokens=1))
        a, b = (e.scheduler.requests[request_id] for request_id in ("a", "b"))
        batch = SchedulerOutput(
            Stage.PREFILLING_PRELUDE,
            [
                ScheduledItem(a, 1, 2, chunk_start=0),
                ScheduledItem(b, 2, 1, chunk_start=1),
            ],
        )
        prepared = e.model_runner.prepare(batch)
        assert tuple(prepared.prefill_metadata.lease.device_segments.shape) == (2, 5)
        assert prepared.prefill_metadata.lease.host_segments.is_pinned()
        assert prepared.prefill_metadata.lease.host_token_ids.is_pinned()
        assert prepared.prefill_metadata.token_ids.tolist() == [2, 3, 7]
        hidden = torch.arange(
            3 * e.model.config.hidden_size, device="cuda", dtype=torch.bfloat16
        ).view(3, e.model.config.hidden_size)
        e.model_runner._scatter_prefill(hidden, prepared)
        torch.cuda.synchronize()
        assert torch.equal(e.model_runner._gather_prefill(prepared), hidden)
    finally:
        if "prepared" in locals():
            torch.cuda.synchronize()
            e.model_runner._release_prepared_metadata(prepared)
        e.close()


@pytest.mark.gpu
def test_cuda_final_depth_prefill_scatter_retains_only_coda_hidden_per_item():
    from vllm_rlt.core.scheduler import ScheduledItem, SchedulerOutput

    e = LLMEngine(
        OuroForCausalLM(tiny_ouro_config()).to("cuda", torch.bfloat16),
        cache_config=CacheConfig(64, 4),
        scheduler_config=SchedulerConfig(
            max_num_seqs=2, max_num_batched_tokens=4, prefill_chunk_size=2
        ),
        execution_config=ExecutionConfig(atomic_prefill=False),
        attention_backend="triton",
    )
    prepared = None
    try:
        e.add_request("mid", [1, 2], SamplingParams(max_tokens=1))
        e.add_request("final", [3, 4], SamplingParams(max_tokens=1))
        for request_id in ("mid", "final"):
            assert e.cache_manager.allocate(request_id, 2)
        mid, final = (e.scheduler.requests[rid] for rid in ("mid", "final"))
        batch = SchedulerOutput(
            Stage.PREFILLING_RECURRENT,
            [
                ScheduledItem(mid, 0, 2, depth=1, chunk_start=0),
                ScheduledItem(final, 0, 2, depth=3, chunk_start=0, completes_prompt=True),
            ],
        )
        runner = e.model_runner
        prepared = runner.prepare(batch)
        mid_slot, final_slot = runner.state_slots["mid"], runner.state_slots["final"]
        runner.prefill_states[mid_slot, :2].fill_(9)
        runner.prefill_states[final_slot, :2].fill_(7)
        hidden = torch.arange(
            4 * e.model.config.hidden_size, device="cuda", dtype=torch.bfloat16
        ).view(4, e.model.config.hidden_size)

        runner._scatter_prefill(hidden, prepared)
        runner._store_split_prefill_outputs(hidden, prepared)
        torch.cuda.synchronize()

        assert torch.equal(runner.prefill_states[mid_slot, :2], hidden[:2])
        assert torch.equal(runner.prefill_states[final_slot, :2], torch.full_like(hidden[2:], 7))
        assert torch.equal(final.hidden_state, hidden[-1])
        assert mid.hidden_state is None
    finally:
        if prepared is not None:
            torch.cuda.synchronize()
            e.model_runner._release_prepared_metadata(prepared)
        e.close()


@pytest.mark.gpu
def test_cuda_prefill_submit_keeps_host_and_device_buffers_until_event():
    from vllm_rlt.core.scheduler import ScheduledItem, SchedulerOutput
    from vllm_rlt.worker.model_runner import PrefillMetadataPool

    e = LLMEngine(
        OuroForCausalLM(tiny_ouro_config()).to("cuda", torch.bfloat16),
        cache_config=CacheConfig(64, 4),
        scheduler_config=SchedulerConfig(
            max_num_seqs=2, max_num_batched_tokens=4, prefill_chunk_size=4
        ),
        execution_config=ExecutionConfig(atomic_prefill=False),
        attention_backend="triton",
    )

    class HeldCudaEvent:
        def __init__(self, event):
            self.event = event
            self.released = False

        def query(self):
            return self.released and self.event.query()

        def synchronize(self):
            self.event.synchronize()
            self.released = True

    try:
        e.add_request("a", [1, 2, 3, 4], SamplingParams(max_tokens=1))
        e.add_request("b", [5, 6, 7, 8], SamplingParams(max_tokens=1))
        a, b = (e.scheduler.requests[rid] for rid in ("a", "b"))
        batches = [
            SchedulerOutput(
                Stage.PREFILLING_PRELUDE,
                [ScheduledItem(request, 0, 2, chunk_start=0)],
            )
            for request in (a, b)
        ]
        runner = e.model_runner
        runner.prefill_metadata_pool = PrefillMetadataPool(2, 4, runner.device, slots=2)

        first_ticket = runner.submit(batches[0])
        first_lease = runner.prefill_metadata_pool.slots[0]
        first_segments = first_lease.host_segments[:1].clone()
        first_tokens = first_lease.host_token_ids[:2].clone()
        held = HeldCudaEvent(first_ticket.event)
        first_lease.event = held

        second_ticket = runner.submit(batches[1])
        second_lease = runner.prefill_metadata_pool.slots[1]
        assert second_lease.host_segments.data_ptr() != first_lease.host_segments.data_ptr()
        assert second_lease.host_token_ids.data_ptr() != first_lease.host_token_ids.data_ptr()
        assert torch.equal(first_lease.host_segments[:1], first_segments)
        assert torch.equal(first_lease.host_token_ids[:2], first_tokens)

        second_ticket.event.synchronize()
        third_ticket = runner.submit(batches[1])
        assert runner.prefill_metadata_pool.slots[1].event is third_ticket.event
        held.synchronize()
        assert torch.equal(first_lease.device_segments[:1].cpu(), first_segments)
        assert torch.equal(first_lease.device_token_ids[:2].cpu(), first_tokens)
        fourth_ticket = runner.submit(batches[0])
        assert runner.prefill_metadata_pool.slots[0].event is fourth_ticket.event
    finally:
        torch.cuda.synchronize()
        e.close()


@pytest.mark.gpu
def test_cuda_kv_metadata_lease_reuses_buffers_after_consumer_event():
    from vllm_rlt.core.kv_cache_manager import KVCacheManager, KVSpan

    cache = KVCacheManager(
        1, 1, 64, 32, 16, 4, device="cuda", dtype=torch.bfloat16, backend="triton"
    )
    assert cache.allocate("a", 4)
    first = cache.acquire_metadata_lease()
    batch = cache._prepare_batch_spans([KVSpan("a", 0, 1, 2)], metadata_lease=first)
    wide_ptr = first.device_wide.data_ptr()
    narrow_ptr = first.device_narrow.data_ptr()
    assert first.host_wide.is_pinned() and first.host_narrow.is_pinned()
    consumed = first.device_wide[:6].sum() + first.device_narrow.sum()
    event = torch.cuda.Event()
    event.record(torch.cuda.current_stream())
    cache.release_metadata_lease(batch.metadata_lease, event)

    second = cache.acquire_metadata_lease()
    assert second is not first
    second_done = torch.cuda.Event()
    second_done.record(torch.cuda.current_stream())
    cache.release_metadata_lease(second, second_done)
    third = cache.acquire_metadata_lease()
    third_done = torch.cuda.Event()
    third_done.record(torch.cuda.current_stream())
    cache.release_metadata_lease(third, third_done)

    event.synchronize()
    assert torch.equal(first.device_wide[:6].cpu(), first.host_wide[:6])
    assert torch.isfinite(consumed)
    reused = cache.acquire_metadata_lease()
    assert reused is first
    assert reused.device_wide.data_ptr() == wide_ptr
    assert reused.device_narrow.data_ptr() == narrow_ptr
    cache.release_metadata_lease(reused)
    torch.cuda.synchronize()


class HeldEvent:
    def __init__(self, before_sync=None):
        self.complete = False
        self.before_sync = before_sync

    def query(self):
        return self.complete

    def synchronize(self):
        if self.before_sync is not None:
            self.before_sync()
        self.complete = True


def test_prefill_metadata_pool_reuses_slots_after_their_events_complete():
    from vllm_rlt.worker.model_runner import PrefillMetadataPool

    pool = PrefillMetadataPool(2, 4, torch.device("cpu"), slots=2)
    first = pool.acquire()
    first_ptr = first.host_segments.data_ptr()
    first_token_ptr = first.host_token_ids.data_ptr()
    pending = HeldEvent()
    pool.release_after(first, pending)

    second = pool.acquire()
    second_ptr = second.host_segments.data_ptr()
    assert second_ptr != first_ptr
    assert second.host_token_ids.data_ptr() != first_token_ptr
    second_done = HeldEvent()
    second_done.complete = True
    pool.release_after(second, second_done)

    reused_second = pool.acquire()
    assert reused_second.host_segments.data_ptr() == second_ptr
    assert reused_second.host_segments.data_ptr() != first_ptr
    pool.release(reused_second)

    pending.complete = True
    reused_first = pool.acquire()
    assert reused_first.host_segments.data_ptr() == first_ptr
    assert reused_first.host_token_ids.data_ptr() == first_token_ptr
    pool.release(reused_first)


def test_prefill_metadata_pool_waits_only_when_all_slots_are_in_flight():
    from vllm_rlt.worker.model_runner import PrefillMetadataPool

    pool = PrefillMetadataPool(1, 1, torch.device("cpu"), slots=2)
    first = pool.acquire()
    first_pending = HeldEvent()
    pool.release_after(first, first_pending)
    second = pool.acquire()
    second_pending = HeldEvent()
    pool.release_after(second, second_pending)

    reused = pool.acquire()
    assert reused is first
    assert first_pending.complete
    assert not second_pending.complete
    pool.release(reused)


@pytest.mark.gpu
def test_submit_transfer_failure_releases_only_prepared_prefill_lease():
    from dataclasses import replace

    from vllm_rlt.core.scheduler import ScheduledItem, SchedulerOutput

    e = LLMEngine(
        OuroForCausalLM(tiny_ouro_config()).to("cuda", torch.bfloat16),
        cache_config=CacheConfig(64, 4),
        scheduler_config=SchedulerConfig(max_num_seqs=1, max_num_batched_tokens=4),
        execution_config=ExecutionConfig(async_scheduling=True, atomic_prefill=False),
        attention_backend="triton",
    )
    e.add_request("a", [1, 2, 3, 4], SamplingParams(max_tokens=1))
    assert e.cache_manager.allocate("a", 4)
    request = e.scheduler.requests["a"]
    batch = SchedulerOutput(
        Stage.PREFILLING_RECURRENT,
        [ScheduledItem(request, token_start=0, token_count=2, depth=0, chunk_start=0)],
    )
    runner = e.model_runner
    original_prepare = runner.prepare

    class FailingRouting:
        imports = ()

        def transfer(self, batch):
            raise RuntimeError("injected metadata transfer failure")

        def record_done(self):
            return None

    def prepare_with_failing_transfer(value):
        prepared = original_prepare(value)
        return replace(prepared, routing=FailingRouting())

    runner.prepare = prepare_with_failing_transfer
    try:
        with pytest.raises(RuntimeError, match="injected metadata transfer failure"):
            runner.submit(batch)
        assert all(not lease.in_use for lease in runner.prefill_metadata_pool.slots)
        assert all(not lease.in_use for lease in runner.cache_manager.metadata_pool.slots)
        runner.prepare = original_prepare
        prepared = runner.prepare(batch)
        torch.cuda.synchronize()
        runner._release_prepared_metadata(prepared)
    finally:
        runner.prepare = original_prepare
        e.close()


def test_metadata_pool_full_failure_does_not_release_existing_leases():
    from vllm_rlt.core.kv_cache_manager import KVMetadataPool

    pool = KVMetadataPool(torch.device("cpu"), slots=2)
    first, second = pool.acquire(), pool.acquire()
    first.stage([1, 2], torch.long)
    second.stage([3, 4], torch.long)
    first_ptr, second_ptr = first.device_wide.data_ptr(), second.device_wide.data_ptr()
    first_event, second_event = HeldEvent(), HeldEvent()
    first.event, second.event = first_event, second_event
    try:
        with pytest.raises(RuntimeError, match="all KV metadata leases are in use"):
            pool.acquire()
        assert first.in_use and second.in_use
        assert first.event is first_event and second.event is second_event
        assert first.device_wide.data_ptr() == first_ptr
        assert second.device_wide.data_ptr() == second_ptr
        pool.release(first)
        third = pool.acquire()
        assert third is first
        pool.release(third)
    finally:
        pool.release(second)


def test_submit_does_not_retry_failed_routing_cleanup(monkeypatch):
    from contextlib import nullcontext
    from dataclasses import replace

    from vllm_rlt.core.scheduler import ScheduledItem, SchedulerOutput

    e = split_engine()
    e.add_request("a", [1, 2], SamplingParams(max_tokens=1))
    request = e.scheduler.requests["a"]
    batch = SchedulerOutput(
        Stage.PREFILLING_PRELUDE,
        [ScheduledItem(request, token_start=0, token_count=2, chunk_start=0)],
    )
    runner = e.model_runner

    class FakeStream:
        def wait_event(self, event):
            return None

    class FailingRouting:
        imports = ()
        ready_event = object()

        def __init__(self):
            self.record_done_calls = 0

        def transfer(self, value):
            return value

        def record_done(self):
            self.record_done_calls += 1
            raise RuntimeError("injected routing cleanup failure")

    routing = FailingRouting()
    original_prepare = runner.prepare
    original_execute = runner._execute

    def fail_execute(batch, prepared):
        raise RuntimeError("injected execute failure")

    runner.core_stream = FakeStream()
    runner.prepare = lambda value: replace(original_prepare(value), routing=routing)
    runner._execute = fail_execute
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: nullcontext())
    try:
        with pytest.raises(RuntimeError, match="injected routing cleanup failure"):
            runner.submit(batch)
        assert routing.record_done_calls == 1
        assert all(not lease.in_use for lease in runner.prefill_metadata_pool.slots)
    finally:
        runner.prepare = original_prepare
        runner._execute = original_execute
        runner.core_stream = None
        e.close()


@pytest.mark.gpu
def test_cuda_submit_readback_failure_releases_metadata_leases():
    from vllm_rlt.core.scheduler import ScheduledItem, SchedulerOutput

    e = LLMEngine(
        OuroForCausalLM(tiny_ouro_config()).to("cuda", torch.bfloat16),
        cache_config=CacheConfig(64, 4),
        scheduler_config=SchedulerConfig(max_num_seqs=1, max_num_batched_tokens=4),
        execution_config=ExecutionConfig(atomic_prefill=False),
        attention_backend="triton",
    )
    e.add_request("a", [1, 2, 3, 4], SamplingParams(max_tokens=1))
    request = e.scheduler.requests["a"]
    batch = SchedulerOutput(
        Stage.PREFILLING_PRELUDE,
        [ScheduledItem(request, token_start=0, token_count=2, chunk_start=0)],
    )
    runner = e.model_runner
    original_readback = runner._readback_slot

    def fail_readback(result):
        raise RuntimeError("injected readback failure")

    runner._readback_slot = fail_readback
    try:
        with pytest.raises(RuntimeError, match="injected readback failure"):
            runner.submit(batch)
        torch.cuda.synchronize()
        assert all(not lease.in_use for lease in runner.prefill_metadata_pool.slots)
        runner._readback_slot = original_readback
        runner.submit(batch)
    finally:
        runner._readback_slot = original_readback
        torch.cuda.synchronize()
        e.close()


def advance_chunk(request, item):
    """Apply one submitted depth run to the request's logical frontiers.

    The run leaves `prefill_depth_ranges[depth]` and extends the right edge of
    the next frontier, or joins the finished prefix at the last depth.
    """
    depth = request.prefill_depth(item.token_start)
    boundary = request.prefill_depth_ranges[depth]
    assert item.token_start == boundary[0], "a depth run must start at its frontier"
    end = item.token_start + item.token_count
    request.prefill_depth_ranges[depth] = [end, boundary[1]]
    if depth + 1 < TOTAL_DEPTHS:
        deeper = request.prefill_depth_ranges[depth + 1]
        request.prefill_depth_ranges[depth + 1] = [deeper[0], end]
    else:
        request.num_prefilled_tokens = end
    return depth


def schedule_stage(scheduler, stage):
    batch = scheduler._take(stage)
    assert batch is not None, f"no {stage.value} batch was selected"
    return batch


def test_split_prefill_steps_one_depth_per_chunk_and_keeps_the_prompt_ordered():
    e = split_engine()
    e.add_request("a", [1, 2, 3, 4, 5], SamplingParams(max_tokens=2, ignore_eos=True))
    request = e.scheduler.requests["a"]
    prompt = request.prompt_token_ids

    # Mirror the pending engine update: each completed stage enqueues the next one.
    def apply(batch):
        item = batch.items[0]
        if batch.stage == Stage.PREFILLING_PRELUDE:
            request.num_preluded_tokens = item.token_start + item.token_count
            e.scheduler.enqueue(request, Stage.PREFILLING_RECURRENT)
            return (item.token_start, item.token_count, None)
        depth = advance_chunk(request, item)
        if depth + 1 < TOTAL_DEPTHS:
            e.scheduler.enqueue(request, Stage.PREFILLING_RECURRENT)
        elif request.num_prefilled_tokens < len(prompt):
            e.scheduler.enqueue(request, Stage.PREFILLING_PRELUDE)
        return (item.token_start, item.token_count, depth)

    trace = []
    for _ in range(40):
        batch = e.scheduler.schedule()
        if batch is None:
            break
        assert len(batch.items) == 1
        trace.append((batch.stage, *apply(batch)))
    assert e.scheduler.schedule() is None, "prefill must drain, not schedule forever"

    assert trace == [
        (Stage.PREFILLING_PRELUDE, 0, 2, None),
        (Stage.PREFILLING_RECURRENT, 0, 2, 0),
        (Stage.PREFILLING_RECURRENT, 0, 2, 1),
        (Stage.PREFILLING_RECURRENT, 0, 2, 2),
        (Stage.PREFILLING_RECURRENT, 0, 2, 3),
        (Stage.PREFILLING_PRELUDE, 2, 2, None),
        (Stage.PREFILLING_RECURRENT, 2, 2, 0),
        (Stage.PREFILLING_RECURRENT, 2, 2, 1),
        (Stage.PREFILLING_RECURRENT, 2, 2, 2),
        (Stage.PREFILLING_RECURRENT, 2, 2, 3),
        (Stage.PREFILLING_PRELUDE, 4, 1, None),
        (Stage.PREFILLING_RECURRENT, 4, 1, 0),
        (Stage.PREFILLING_RECURRENT, 4, 1, 1),
        (Stage.PREFILLING_RECURRENT, 4, 1, 2),
        (Stage.PREFILLING_RECURRENT, 4, 1, 3),
    ]
    assert request.num_prefilled_tokens == 5
    assert all(left == right for left, right in request.prefill_depth_ranges)


def test_engine_split_prefill_transition_advances_the_same_frontier_contract():
    e = split_engine()
    e.add_request("a", [1, 2, 3, 4, 5], SamplingParams(max_tokens=2, ignore_eos=True))
    request = e.scheduler.requests["a"]
    trace = []

    for _ in range(40):
        batch = e.scheduler.schedule()
        if batch is None:
            break
        assert len(batch.items) == 1
        item = batch.items[0]
        progress = e._apply_split_prefill_progress(request, batch.stage, item)
        trace.append((batch.stage, item.token_start, item.token_count, item.depth))
        if progress.prompt_complete:
            break

    assert trace == [
        (Stage.PREFILLING_PRELUDE, 0, 2, None),
        (Stage.PREFILLING_RECURRENT, 0, 2, 0),
        (Stage.PREFILLING_RECURRENT, 0, 2, 1),
        (Stage.PREFILLING_RECURRENT, 0, 2, 2),
        (Stage.PREFILLING_RECURRENT, 0, 2, 3),
        (Stage.PREFILLING_PRELUDE, 2, 2, None),
        (Stage.PREFILLING_RECURRENT, 2, 2, 0),
        (Stage.PREFILLING_RECURRENT, 2, 2, 1),
        (Stage.PREFILLING_RECURRENT, 2, 2, 2),
        (Stage.PREFILLING_RECURRENT, 2, 2, 3),
        (Stage.PREFILLING_PRELUDE, 4, 1, None),
        (Stage.PREFILLING_RECURRENT, 4, 1, 0),
        (Stage.PREFILLING_RECURRENT, 4, 1, 1),
        (Stage.PREFILLING_RECURRENT, 4, 1, 2),
        (Stage.PREFILLING_RECURRENT, 4, 1, 3),
    ]
    assert request.num_prefilled_tokens == len(request.prompt_token_ids)
    assert request.stage is Stage.CODA


@pytest.mark.parametrize(
    ("chunk_size", "token_budget", "prompt"),
    [(1, 3, [1]), (2, 3, [1, 2, 3, 4, 5]), (3, 4, [5, 4, 3, 2, 1, 6, 7])],
)
def test_sync_split_prefill_matches_atomic_tokens_and_exit_depths(chunk_size, token_budget, prompt):
    scheduler = SchedulerConfig(
        max_num_seqs=2,
        max_num_batched_tokens=token_budget,
        prefill_chunk_size=chunk_size,
    )
    atomic = engine(scheduler_config=scheduler)
    split = split_engine(scheduler_config=scheduler)

    expected = run_request(atomic, prompt)
    actual = run_request(split, prompt)

    assert actual.token_ids == expected.token_ids
    assert actual.exit_depths == expected.exit_depths
    assert actual.exit_depths == [TOTAL_DEPTHS, 2, 2]
    assert split.scheduler.requests == {}
    assert split.model_runner.state_slots == {}
    assert split.cache_manager.num_used_blocks == 0


def test_sync_split_prefill_resumes_after_a_cached_prefix():
    prompt = [1, 2, 3, 4, 5]
    cache = CacheConfig(64, 2, enable_prefix_caching=True)
    scheduler = SchedulerConfig(max_num_seqs=2, max_num_batched_tokens=3, prefill_chunk_size=2)
    atomic = engine(cache_config=cache, scheduler_config=scheduler)
    split = split_engine(cache_config=cache, scheduler_config=scheduler)

    expected = run_request(atomic, prompt)
    first = run_request(split, prompt)
    assert first.token_ids == expected.token_ids
    assert len(split.cache_manager.lookup_prefix(prompt)) == 2

    split.add_request(
        "cached",
        prompt,
        SamplingParams(max_tokens=3, max_loops=2, ignore_eos=True),
    )
    split.scheduler._admit()
    assert split.scheduler.requests["cached"].num_prefilled_tokens == 4
    actual = drain_request(split)

    assert actual.token_ids == expected.token_ids
    assert actual.exit_depths == expected.exit_depths
    assert split.model_runner.state_slots == {}


def test_sync_split_prefill_abort_releases_slot_before_request_id_reuse():
    split = split_engine()
    prompt = [1, 2, 3, 4]
    split.add_request("reused", prompt, SamplingParams(max_tokens=2, ignore_eos=True))

    split.step()  # Stage the first prompt chunk.
    split.step()  # Complete depth zero, leaving the chunk resident.
    request = split.scheduler.requests["reused"]
    assert request.num_prefilled_tokens == 0
    assert request.prefill_depth_ranges[0] == [2, 4]
    assert split.model_runner.state_slots == {"reused": 0}

    split.abort_request("reused")
    assert split.model_runner.state_slots == {}
    assert split.cache_manager.num_used_blocks == 0

    actual = run_request(split, [5, 4, 3], request_id="reused")
    expected = run_request(engine(), [5, 4, 3])
    assert actual.token_ids == expected.token_ids
    assert actual.exit_depths == expected.exit_depths


def test_cpu_async_split_prefill_matches_sync_split():
    prompt = [1, 2, 3, 4, 5]
    trace = ExitConfig("trace", depths_by_request={"run": [TOTAL_DEPTHS, 2, 2]})
    sync = split_engine(exit_config=trace)
    asynchronous = split_engine(
        exit_config=trace,
        execution_config=ExecutionConfig(async_scheduling=True, atomic_prefill=False),
    )

    expected = run_request(sync, prompt)
    actual = run_request(asynchronous, prompt)

    assert actual.token_ids == expected.token_ids
    assert actual.exit_depths == expected.exit_depths
    assert asynchronous.cache_manager.num_used_blocks == 0


def test_cpu_async_split_frontier_advances_once_per_submission():
    prompt = [1, 2, 3, 4, 5]
    trace = ExitConfig("trace", depths_by_request={"run": [TOTAL_DEPTHS, 2, 2]})
    e = split_engine(
        exit_config=trace,
        execution_config=ExecutionConfig(async_scheduling=True, atomic_prefill=False),
    )
    e.add_request("run", prompt, SamplingParams(max_tokens=3, max_loops=2, ignore_eos=True))
    request = e.scheduler.requests["run"]
    held = []
    submit = e.model_runner.submit

    def hold_split_ticket(batch):
        ticket = submit(batch)
        if batch.stage in (Stage.PREFILLING_PRELUDE, Stage.PREFILLING_RECURRENT):
            ticket.event = HeldEvent()
            held.append(ticket.event)
            for item in batch.items:
                e.model_runner.events[item.request.request_id] = ticket.event
        return ticket

    e.model_runner.submit = hold_split_ticket
    for _ in range(32):
        e.step()
        if request.stage is Stage.CODA:
            break

    assert request.stage is Stage.CODA
    assert request.num_prefilled_tokens == len(prompt)
    assert all(left == right for left, right in request.prefill_depth_ranges)
    assert len(held) == 15  # Three chunks, each with prelude plus four depths.
    assert e.model_runner.split_prefill_submissions == len(held)
    assert all(not event.complete for event in held)
    for event in held:
        event.complete = True

    output = drain_request(e)
    assert output.exit_depths == [TOTAL_DEPTHS, 2, 2]


def test_split_submission_bound_counts_and_waits_for_oldest_event():
    e = split_engine(
        exit_config=ExitConfig("ouro_delayed"),
        execution_config=ExecutionConfig(async_scheduling=True, atomic_prefill=False),
    )
    e.add_request("wait", [1, 2], SamplingParams(max_tokens=1, ignore_eos=True))
    e.scheduler._admit()
    batch = e.scheduler.schedule()
    runner = e.model_runner
    events = [HeldEvent() for _ in range(3)]
    runner.submission_events.extend(events)

    runner.submit(batch)
    e._update(batch, None)

    assert events[0].complete
    assert len(runner.submission_events) == 2
    assert runner.split_prefill_submissions == 1
    assert runner.split_prefill_submission_waits == 1
    e.abort_request("wait")


def test_runner_slot_release_waits_for_last_request_event():
    e = split_engine()
    e.add_request("slot", [1, 2], SamplingParams(max_tokens=1, ignore_eos=True))
    e.scheduler._admit()
    request = e.scheduler.requests["slot"]
    runner = e.model_runner
    slot = runner._ensure_request_slot(request)

    def verify_slot_is_still_owned():
        assert runner.state_slots["slot"] == slot

    event = HeldEvent(before_sync=verify_slot_is_still_owned)
    runner.events["slot"] = event

    runner.release("slot")

    assert event.complete
    assert "slot" not in runner.state_slots
    assert slot in runner.free_state_slots
    e.abort_request("slot")


@pytest.mark.parametrize("completed_depths", range(TOTAL_DEPTHS))
def test_split_preemption_rewinds_and_recomputes_the_resident_chunk(completed_depths):
    prompt = [1, 2, 3, 4]
    params = SamplingParams(max_tokens=3, max_loops=2, ignore_eos=True)
    scheduler = SchedulerConfig(
        max_num_seqs=2,
        max_num_batched_tokens=4,
        prefill_chunk_size=2,
        enable_preemption=True,
    )
    expected = run_request(split_engine(scheduler_config=scheduler), prompt)
    e = split_engine(scheduler_config=scheduler)
    e.add_request("victim", prompt, params)
    request = e.scheduler.requests["victim"]

    for _ in range(completed_depths + 1):
        e.step()
    assert request.stage is Stage.PREFILLING_RECURRENT

    requester = Request("pressure", [9], SamplingParams(max_tokens=1))
    assert e.preemption.preempt(requester)
    assert e.preemption.snapshots["victim"]["stage"] is Stage.PREFILLING_PRELUDE
    assert request.num_preluded_tokens == request.num_prefilled_tokens
    assert request.prefill_chunk_start == request.num_prefilled_tokens
    assert all(left == request.num_prefilled_tokens for left, _ in request.prefill_depth_ranges)
    assert e.cache_manager.num_used_blocks == 0

    e.scheduler._admit()
    assert request.stage is Stage.PREFILLING_PRELUDE
    actual = drain_request(e)
    assert actual.token_ids == expected.token_ids
    assert actual.exit_depths == expected.exit_depths
    assert e.preemption.resumptions == 1
    assert not e.preemption.snapshots
    assert e.cache_manager.num_used_blocks == 0


def test_priority_admission_preempts_split_prefill_and_matches_reference():
    prompt = [1, 2, 3, 4]
    params = SamplingParams(max_tokens=3, max_loops=2, ignore_eos=True, priority=10)
    trace = ExitConfig("trace", depths_by_request={"reference": [4, 2, 2]})
    scheduler = SchedulerConfig(
        max_num_seqs=1,
        max_num_batched_tokens=4,
        prefill_chunk_size=2,
        policy="priority",
    )
    expected = run_request(
        split_engine(scheduler_config=scheduler, exit_config=trace), prompt, "reference"
    )

    e = split_engine(
        scheduler_config=SchedulerConfig(
            max_num_seqs=1,
            max_num_batched_tokens=4,
            prefill_chunk_size=2,
            policy="priority",
            enable_preemption=True,
        ),
        exit_config=ExitConfig("trace", depths_by_request={"low": [4, 2, 2], "high": [4]}),
    )
    e.add_request("low", prompt, params)
    e.step()
    e.step()
    assert e.scheduler.requests["low"].stage is Stage.PREFILLING_RECURRENT
    e.add_request(
        "high",
        [5, 6],
        SamplingParams(max_tokens=1, max_loops=2, ignore_eos=True, priority=-10),
    )

    finished = {}
    for _ in range(100):
        for output in e.step():
            if output.finished:
                finished[output.request_id] = output
        if len(finished) == 2:
            break
    else:
        pytest.fail("priority preemption did not finish both requests")

    assert e.preemption.preemptions == 1 == e.preemption.resumptions
    assert finished["low"].token_ids == expected.token_ids
    assert finished["low"].exit_depths == expected.exit_depths
    assert len(finished["high"].token_ids) == 1
    assert e.cache_manager.num_used_blocks == 0
    assert e.model_runner.state_slots == {}


@pytest.mark.gpu
@pytest.mark.parametrize("async_scheduling", [False, True])
@pytest.mark.parametrize("failure_site", ["prepare", "_execute"])
def test_cuda_split_failure_drains_writes_before_request_id_reuse(
    monkeypatch, async_scheduling, failure_site
):
    e = LLMEngine(
        OuroForCausalLM(tiny_ouro_config()).to("cuda", torch.bfloat16),
        cache_config=CacheConfig(64, 4),
        scheduler_config=SchedulerConfig(max_num_seqs=1, max_num_batched_tokens=4),
        execution_config=ExecutionConfig(atomic_prefill=False, async_scheduling=async_scheduling),
        exit_config=ExitConfig("trace", depths_by_request={"reused": [4]}),
        attention_backend="triton",
    )
    runner = e.model_runner
    original = getattr(runner, failure_site)
    completed = torch.cuda.Event()

    def fail(batch, *args):
        result = original(batch, *args)
        if batch.stage == Stage.PREFILLING_RECURRENT:
            torch.cuda._sleep(10_000_000)
            completed.record(torch.cuda.current_stream(runner.device))
            raise RuntimeError("injected split failure")
        return result

    try:
        monkeypatch.setattr(runner, failure_site, fail)
        e.add_request("reused", [1, 2, 3], SamplingParams(max_tokens=1, ignore_eos=True))
        e.step()
        with pytest.raises(RuntimeError, match="injected split failure"):
            e.step()
        assert completed.query()
        assert e.cache_manager.num_used_blocks == 0
        assert runner.state_slots == {}

        monkeypatch.setattr(runner, failure_site, original)
        e.add_request("reused", [3, 2, 1], SamplingParams(max_tokens=1, ignore_eos=True))
        result = drain_request(e)
        assert len(result.token_ids) == 1
        assert result.exit_depths == [4]
    finally:
        e.close()


@pytest.mark.gpu
@pytest.mark.parametrize("async_scheduling", [False, True])
def test_cuda_split_prefill_matches_atomic(async_scheduling):
    prompt = [1, 2, 3, 4, 5]
    params = SamplingParams(max_tokens=3, max_loops=2, ignore_eos=True)
    execution = dict(
        async_scheduling=async_scheduling,
        multi_stream=async_scheduling,
        static_buffers=async_scheduling,
        pad_to_power_of_two=async_scheduling,
    )
    torch.manual_seed(321)
    atomic_model = OuroForCausalLM(tiny_ouro_config()).to("cuda", torch.bfloat16)
    atomic = LLMEngine(
        atomic_model,
        cache_config=CacheConfig(128, 2, enable_prefix_caching=True),
        scheduler_config=SchedulerConfig(max_num_seqs=2, max_num_batched_tokens=4),
        execution_config=ExecutionConfig(atomic_prefill=True, **execution),
        exit_config=ExitConfig("ouro_delayed"),
        attention_backend="triton",
    )
    atomic_output = run_request(atomic, prompt, "first")
    atomic.add_request("second", prompt, params)
    atomic.step()
    atomic_cached_output = drain_request(atomic)
    atomic.close()
    del atomic_model
    torch.cuda.empty_cache()

    torch.manual_seed(321)
    split_model = OuroForCausalLM(tiny_ouro_config()).to("cuda", torch.bfloat16)
    split = LLMEngine(
        split_model,
        cache_config=CacheConfig(128, 2, enable_prefix_caching=True),
        scheduler_config=SchedulerConfig(max_num_seqs=2, max_num_batched_tokens=4),
        execution_config=ExecutionConfig(atomic_prefill=False, **execution),
        exit_config=ExitConfig("ouro_delayed"),
        attention_backend="triton",
    )
    split_output = run_request(split, prompt, "first")
    split.add_request("second", prompt, params)
    split.step()
    assert split.cache_manager.prefix_hits > 0
    assert split.scheduler.requests["second"].num_prefilled_tokens > 0
    split_cached_output = drain_request(split)
    split.close()

    assert split_output.token_ids == atomic_output.token_ids
    assert split_output.exit_depths == atomic_output.exit_depths
    assert split_cached_output.token_ids == atomic_cached_output.token_ids
    assert split_cached_output.exit_depths == atomic_cached_output.exit_depths


@pytest.mark.gpu
def test_cuda_split_prefill_survives_decode_graph_capture_and_replay():
    from vllm_rlt.worker.buffers import execution_buffer_bytes

    graph_reserve = 32 * 1024**2

    def run(cuda_graphs):
        torch.manual_seed(765)
        model = OuroForCausalLM(tiny_ouro_config(head_dim=64)).to("cuda", torch.bfloat16)
        cache = CacheConfig(
            block_size=16, gpu_memory_utilization=0.5, memory_reserve_bytes=32 * 1024**2
        )
        scheduler = SchedulerConfig(max_num_seqs=2, max_num_batched_tokens=4, prefill_chunk_size=2)
        execution = ExecutionConfig(
            async_scheduling=True,
            cuda_graphs=cuda_graphs,
            cuda_graph_memory_reserve_bytes=graph_reserve,
            atomic_prefill=False,
        )
        e = LLMEngine(
            model,
            cache_config=cache,
            scheduler_config=scheduler,
            execution_config=execution,
            exit_config=ExitConfig("trace", depths_by_request={"split": [4], "decode": [4, 2, 2]}),
            attention_backend="triton",
        )
        try:
            e.add_request("split", [1, 2, 3, 4], SamplingParams(max_tokens=1, ignore_eos=True))
            e.step()
            e.step()
            split_request = e.scheduler.requests["split"]
            assert split_request.stage is Stage.PREFILLING_RECURRENT
            assert split_request.prefill_depth_ranges[0] == [2, 4]
            # Keep the staged chunk resident while decode triggers graph capture.
            e.scheduler.queues[Stage.PREFILLING_RECURRENT].remove("split")

            e.add_request(
                "decode",
                [5, 6],
                SamplingParams(max_tokens=3, max_loops=2, ignore_eos=True),
            )
            decode_recurrent_runs = 0
            for _ in range(100):
                e.step()
                batch = e.last_schedule
                if batch is not None and batch.stage is Stage.RECURRENT:
                    decode_recurrent_runs += sum(
                        item.request.request_id == "decode" for item in batch.items
                    )
                if decode_recurrent_runs == 2:
                    break
            else:
                pytest.fail("decode request did not reach its second recurrent step")
            assert split_request.stage is Stage.PREFILLING_RECURRENT
            assert split_request.num_prefilled_tokens == 0
            assert split_request.num_preluded_tokens == 2

            graph_counts = None
            if e.model_runner.graphs is not None:
                graph_counts = (
                    e.model_runner.graphs.captures,
                    e.model_runner.graphs.replays,
                    e.model_runner.graphs.fallbacks,
                )
            e.scheduler.enqueue(split_request, Stage.PREFILLING_RECURRENT)
            finished = {}
            for _ in range(200):
                for output in e.step():
                    if output.finished:
                        finished[output.request_id] = output
                if len(finished) == 2:
                    break
            else:
                pytest.fail("split-prefill graph requests did not finish")
            if graph_counts is not None:
                graph_counts = (
                    graph_counts,
                    (
                        e.model_runner.graphs.captures,
                        e.model_runner.graphs.replays,
                        e.model_runner.graphs.fallbacks,
                    ),
                )

            expected_buffers = execution_buffer_bytes(
                model.config, scheduler, cache, execution, next(model.parameters()).element_size()
            )
            return (
                {
                    "decode": (finished["decode"].token_ids, finished["decode"].exit_depths),
                    "split": (finished["split"].token_ids, finished["split"].exit_depths),
                },
                e.memory_plan,
                expected_buffers,
                graph_counts,
            )
        finally:
            e.close()

    eager, graphed = run(False), run(True)
    assert eager[0] == graphed[0]
    assert eager[1]["cuda_graph_reserve_bytes"] == 0
    assert graphed[1]["cuda_graph_reserve_bytes"] == graph_reserve
    assert eager[1]["kv_budget_bytes"] >= graphed[1]["kv_budget_bytes"] + graph_reserve
    assert eager[1]["static_buffer_bytes"] == eager[2] > 0
    assert graphed[1]["static_buffer_bytes"] == graphed[2] > 0
    before_resume, after_resume = graphed[3]
    assert before_resume == (1, 2, 0)
    assert after_resume == (1, 4, 0)


@pytest.mark.gpu
def test_cuda_async_split_preemption_releases_and_reuses_slot():
    prompt = [1, 2, 3, 4]
    params = SamplingParams(max_tokens=3, max_loops=2, ignore_eos=True, priority=10)
    scheduler = SchedulerConfig(max_num_seqs=2, max_num_batched_tokens=4, prefill_chunk_size=2)
    execution = ExecutionConfig(atomic_prefill=False)
    torch.manual_seed(432)
    reference_model = OuroForCausalLM(tiny_ouro_config()).to("cuda", torch.bfloat16)
    reference = LLMEngine(
        reference_model,
        cache_config=CacheConfig(128, 2),
        scheduler_config=scheduler,
        execution_config=execution,
        exit_config=ExitConfig("trace", depths_by_request={"run": [4, 2, 2]}),
        attention_backend="triton",
    )
    expected = run_request(reference, prompt)
    reference.close()
    del reference_model
    torch.cuda.empty_cache()

    torch.manual_seed(432)
    model = OuroForCausalLM(tiny_ouro_config()).to("cuda", torch.bfloat16)
    e = LLMEngine(
        model,
        cache_config=CacheConfig(128, 2),
        scheduler_config=SchedulerConfig(
            max_num_seqs=2,
            max_num_batched_tokens=4,
            prefill_chunk_size=2,
            policy="priority",
            enable_preemption=True,
        ),
        execution_config=ExecutionConfig(async_scheduling=True, atomic_prefill=False),
        exit_config=ExitConfig(
            "trace",
            depths_by_request={
                "victim": [4, 2, 2],
                "pressure-a": [4],
                "pressure-b": [4],
            },
        ),
        attention_backend="triton",
    )
    try:
        e.add_request("victim", prompt, params)
        e.step()
        e.step()
        assert e.scheduler.requests["victim"].stage is Stage.PREFILLING_RECURRENT
        state = e.model_runner.async_state
        slot = state.slots["victim"]
        old_event = e.model_runner.events["victim"]
        assert e.preemption.preempt(
            Request("pressure", [9], SamplingParams(max_tokens=1, priority=-10))
        )
        assert "victim" not in state.slots
        assert slot in state.free
        assert "victim" not in e.model_runner.events

        pressure_params = SamplingParams(max_tokens=1, priority=-10)
        e.add_request("pressure-a", [5, 6], pressure_params)
        e.add_request("pressure-b", [7, 8], pressure_params)
        e.scheduler._admit()
        e.step()
        assert e.scheduler.requests["victim"].stage is Stage.WAITING
        assert slot in state.slots.values()
        assert len(state.slots) == 2
        occupant = next(rid for rid, assigned in state.slots.items() if assigned == slot)
        other = "pressure-b" if occupant == "pressure-a" else "pressure-a"
        e.abort_request(occupant)
        e.abort_request(other)

        e.scheduler._admit()
        e.step()
        assert state.slots["victim"] != slot
        assert e.model_runner.events["victim"] is not old_event
        actual = drain_request(e)

        assert actual.token_ids == expected.token_ids
        assert actual.exit_depths == expected.exit_depths
        assert e.preemption.preemptions == 1 == e.preemption.resumptions
    finally:
        e.close()


@pytest.mark.gpu
@pytest.mark.parametrize("attention_backend", ["triton", "flash_attn_4"])
def test_cuda_split_prefill_batches_mixed_depths(attention_backend):
    torch.manual_seed(654)
    model = OuroForCausalLM(tiny_ouro_config()).to("cuda", torch.bfloat16)
    try:
        e = LLMEngine(
            model,
            cache_config=CacheConfig(128, 16, enable_prefix_caching=True),
            scheduler_config=SchedulerConfig(
                max_num_seqs=2, max_num_batched_tokens=4, prefill_chunk_size=2
            ),
            execution_config=ExecutionConfig(atomic_prefill=False),
            exit_config=ExitConfig("ouro_delayed"),
            attention_backend=attention_backend,
        )
    except (ImportError, OSError, ValueError):
        if attention_backend == "flash_attn_4":
            pytest.skip("FlashAttention-4 is unavailable")
        raise
    try:
        params = SamplingParams(max_tokens=1, ignore_eos=True)
        e.add_request("a", [1, 2], params)
        e.add_request("b", [3, 4], params)
        e.step()  # Prelude for both requests.

        recurrent = e.scheduler.queues[Stage.PREFILLING_RECURRENT]
        recurrent.clear()
        recurrent.append("a")
        e.step()  # Advance only request a to depth one.
        recurrent.append("b")
        e.step()  # Request a is depth one; request b is depth zero.

        assert e.last_schedule.stage is Stage.PREFILLING_RECURRENT
        assert {item.depth for item in e.last_schedule.items} == {0, 1}
    finally:
        e.close()


def test_split_prefill_holds_the_next_prelude_until_the_chunk_is_read():
    """One staged chunk per request: its hidden rows are in use until the last depth."""
    e = split_engine()
    e.add_request("a", [1, 2, 3, 4], SamplingParams(max_tokens=2, ignore_eos=True))
    e.scheduler._admit()
    request = e.scheduler.requests["a"]

    prelude = schedule_stage(e.scheduler, Stage.PREFILLING_PRELUDE)
    item = prelude.items[0]
    request.num_preluded_tokens = item.token_start + item.token_count
    e.scheduler.enqueue(request, Stage.PREFILLING_RECURRENT)
    batch = schedule_stage(e.scheduler, Stage.PREFILLING_RECURRENT)
    advance_chunk(request, batch.items[0])

    # Depth 0 of the first chunk is done, but its hidden rows are still the input
    # of depth 1, so staging the next chunk must wait.
    e.scheduler.enqueue(request, Stage.PREFILLING_PRELUDE)
    assert e.scheduler._take(Stage.PREFILLING_PRELUDE) is None
    assert list(e.scheduler.queues[Stage.PREFILLING_PRELUDE]) == ["a"]
    e.scheduler.queues[Stage.PREFILLING_PRELUDE].clear()
    assert e.scheduler._prelude_window_open(request) is False


def test_split_prefill_reports_no_depth_work_before_the_prelude():
    """Out-of-order staging yields an empty run, not a wrong span.

    The protocol keeps this state unreachable; the assertion documents what the
    scheduler does if it is ever reached, so a broken transition shows up as a
    zero-length item instead of a silently wrong token range.
    """
    e = split_engine()
    e.add_request("a", [1, 2, 3], SamplingParams(max_tokens=2, ignore_eos=True))
    e.scheduler._admit()
    request = e.scheduler.requests["a"]

    item = e.scheduler._make_scheduled_item(request, Stage.PREFILLING_RECURRENT, 4)
    assert item.token_count == 0
    assert request.num_prefilled_tokens == 0


def test_split_prefill_batches_stay_within_the_token_budget():
    e = split_engine(
        scheduler_config=SchedulerConfig(
            max_num_seqs=4, max_num_batched_tokens=3, prefill_chunk_size=2
        )
    )
    for name in ("a", "b", "c"):
        e.add_request(name, [1, 2, 3, 4], SamplingParams(max_tokens=1, ignore_eos=True))
    e.scheduler._admit()
    batch = schedule_stage(e.scheduler, Stage.PREFILLING_PRELUDE)
    assert sum(i.token_count for i in batch.items) <= 3
    assert len(batch.items) == 2


def test_split_policy_honors_decode_due_before_split_prefill():
    from collections import deque
    from types import SimpleNamespace

    e = split_engine(scheduler_config=SchedulerConfig(max_prefill_batches_before_decode=1))
    policy = e.scheduler.policy
    policy.prefill_batches_since_recurrent = 1
    scheduler = SimpleNamespace(
        queues={
            Stage.RECURRENT: deque(["decode"]),
            Stage.PRELUDE: deque(),
            Stage.CODA: deque(),
            Stage.PREFILLING_PRELUDE: deque(["prefill"]),
        },
        _take=lambda stage: stage,
    )
    assert policy.schedule(scheduler) is Stage.RECURRENT
    e.close()


def test_split_prefill_stages_count_toward_decode_fairness():
    e = split_engine(scheduler_config=SchedulerConfig(max_prefill_batches_before_decode=2))
    policy = e.scheduler.policy
    assert policy.PREFILL_STAGES == (
        Stage.PREFILLING_PRELUDE,
        Stage.PREFILLING_RECURRENT,
    )

    policy.record_batch(Stage.PREFILLING_PRELUDE)
    assert not policy.decode_due
    policy.record_batch(Stage.PREFILLING_RECURRENT)
    assert policy.decode_due
    policy.record_batch(Stage.RECURRENT)
    assert not policy.decode_due
    e.close()


def test_split_prefill_stages_after_a_reused_prompt_prefix():
    """A cached prefix is prefilled at every depth: the window opens past it."""
    e = split_engine()
    request = Request(
        "a",
        [1, 2, 3, 4, 5],
        SamplingParams(max_tokens=2, ignore_eos=True),
        prefill_depth_ranges=Request.prefill_boundaries(5, TOTAL_DEPTHS),
    )
    # Admission sets both frontiers to the cached token count.
    request.num_prefilled_tokens = 2
    request.num_preluded_tokens = 2
    assert e.scheduler._prelude_window_open(request)
    item = e.scheduler._make_scheduled_item(request, Stage.PREFILLING_PRELUDE, 4)
    assert (item.token_start, item.token_count) == (2, 2)
    assert request.prefill_depth(item.token_start) == 0


def test_prefill_depth_ranges_track_logical_depths():
    request = Request(
        "a",
        [1, 2, 3],
        SamplingParams(),
        prefill_depth_ranges=Request.prefill_boundaries(3, TOTAL_DEPTHS),
    )
    assert request.prefill_depth_ranges == [[0, 3], [0, 0], [0, 0], [0, 0]]
    assert request.prefill_depth(0) == 0
    assert request.prefill_depth(2) == 0

    request.prefill_depth_ranges[0] = [2, 3]
    request.prefill_depth_ranges[1] = [0, 2]
    assert request.prefill_depth(1) == 1
    assert request.prefill_depth(2) == 0

    with pytest.raises(RuntimeError, match="no prefill depth"):
        Request(
            "b",
            [1],
            SamplingParams(),
            prefill_depth_ranges=Request.prefill_boundaries(1, 1),
        ).prefill_depth(5)


def test_split_prefill_policy_is_selected_only_by_the_atomic_prefill_flag():
    assert RefillPolicy is type(engine().scheduler.policy)
    assert SplitPrefillPolicy is type(split_engine().scheduler.policy)
    assert NoRefillPolicy is type(
        engine(scheduler_config=SchedulerConfig(mode="no_refill")).scheduler.policy
    )


def test_scheduling_policies_declare_their_prefill_stages():
    assert SchedulingPolicy.PREFILL_STAGES == (Stage.PREFILL,)
    assert RefillPolicy.PREFILL_STAGES == (Stage.PREFILL,)
    assert NoRefillPolicy.PREFILL_STAGES == (Stage.PREFILL,)
    assert SpeculativePolicy.PREFILL_STAGES == (Stage.PREFILL,)
    assert SplitPrefillPolicy.PREFILL_STAGES == (
        Stage.PREFILLING_PRELUDE,
        Stage.PREFILLING_RECURRENT,
    )


@pytest.mark.parametrize(
    ("label", "kwargs", "message"),
    [
        (
            "shared layout",
            {"cache_config": CacheConfig(64, 2, "shared")},
            "split prefill requires last_exited",
        ),
        (
            "speculative",
            {"speculative_config": SpeculativeConfig(num_speculative_tokens=2)},
            "split prefill does not support speculative",
        ),
        (
            "no_refill mode",
            {"scheduler_config": SchedulerConfig(mode="no_refill")},
            "split prefill currently implements refill",
        ),
    ],
)
def test_split_prefill_rejects_unimplemented_combinations(label, kwargs, message):
    with pytest.raises(ValueError, match=message):
        split_engine(**kwargs)
