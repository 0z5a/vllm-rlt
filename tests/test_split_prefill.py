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


def advance_chunk(request, item):
    """Apply one completed depth run to the request's depth frontiers.

    The run leaves `running_depth_boundary[depth]` and extends the right edge of
    the next frontier, or joins the finished prefix at the last depth.
    """
    depth = request.prefill_depth(item.token_start)
    boundary = request.running_depth_boundary[depth]
    assert item.token_start == boundary[0], "a depth run must start at its frontier"
    end = item.token_start + item.token_count
    request.running_depth_boundary[depth] = [end, boundary[1]]
    if depth + 1 < TOTAL_DEPTHS:
        deeper = request.running_depth_boundary[depth + 1]
        request.running_depth_boundary[depth + 1] = [deeper[0], end]
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
    assert all(left == right for left, right in request.running_depth_boundary)


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
        progress = e._advance_prefill(request, batch.stage, item)
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
    assert request.running_depth_boundary[0] == [2, 4]
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
        return ticket

    e.model_runner.submit = hold_split_ticket
    for _ in range(32):
        e.step()
        if request.stage is Stage.CODA:
            break

    assert request.stage is Stage.CODA
    assert request.num_prefilled_tokens == len(prompt)
    assert all(left == right for left, right in request.running_depth_boundary)
    assert len(held) == 15  # Three chunks, each with prelude plus four depths.
    assert e.model_runner.split_prefill_submissions == len(held)
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
    slot = runner._slot(request)

    def verify_slot_is_still_owned():
        assert runner.state_slots["slot"] == slot

    event = HeldEvent(before_sync=verify_slot_is_still_owned)
    runner.events["slot"] = event

    runner.release("slot")

    assert event.complete
    assert "slot" not in runner.state_slots
    assert slot in runner.free_state_slots
    e.abort_request("slot")


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


def test_split_prefill_stages_after_a_reused_prompt_prefix():
    """A cached prefix is prefilled at every depth: the window opens past it."""
    e = split_engine()
    request = Request(
        "a",
        [1, 2, 3, 4, 5],
        SamplingParams(max_tokens=2, ignore_eos=True),
        running_depth_boundary=Request.prefill_boundaries(5, TOTAL_DEPTHS),
    )
    # Admission sets both frontiers to the cached token count.
    request.num_prefilled_tokens = 2
    request.num_preluded_tokens = 2
    assert e.scheduler._prelude_window_open(request)
    item = e.scheduler._make_scheduled_item(request, Stage.PREFILLING_PRELUDE, 4)
    assert (item.token_start, item.token_count) == (2, 2)
    assert request.prefill_depth(item.token_start) == 0


def test_prefill_boundaries_track_completed_depths():
    request = Request(
        "a",
        [1, 2, 3],
        SamplingParams(),
        running_depth_boundary=Request.prefill_boundaries(3, TOTAL_DEPTHS),
    )
    assert request.running_depth_boundary == [[0, 3], [0, 0], [0, 0], [0, 0]]
    assert request.prefill_depth(0) == 0
    assert request.prefill_depth(2) == 0

    request.running_depth_boundary[0] = [2, 3]
    request.running_depth_boundary[1] = [0, 2]
    assert request.prefill_depth(1) == 1
    assert request.prefill_depth(2) == 0

    with pytest.raises(RuntimeError, match="no prefill depth"):
        Request(
            "b",
            [1],
            SamplingParams(),
            running_depth_boundary=Request.prefill_boundaries(1, 1),
        ).prefill_depth(5)


def test_split_prefill_policy_is_selected_only_by_the_atomic_prefill_flag():
    assert RefillPolicy is type(engine().scheduler.policy)
    assert SplitPrefillPolicy is type(split_engine().scheduler.policy)
    assert NoRefillPolicy is type(
        engine(scheduler_config=SchedulerConfig(mode="no_refill")).scheduler.policy
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
            "preemption",
            {"scheduler_config": SchedulerConfig(enable_preemption=True)},
            "split prefill does not support preemption",
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
