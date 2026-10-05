"""Capture and replay CPU-visible admission, batch and coda-delivery decisions."""

from experiments.loopkv.schedule_oracle import signature
from vllm_rlt import ExitConfig
from vllm_rlt.core.scheduler import ScheduledItem, SchedulerOutput
from vllm_rlt.engine.llm_engine import LLMEngine
from vllm_rlt.request import Stage


def ticket_key(ticket):
    return [[item.request.request_id for item in ticket.batch.items], list(ticket.output_indices)]


class Capture:
    """Observe the existing engine; preserve its natural policy and GPU overlap."""

    def __init__(self, engine: LLMEngine):
        assert engine.execution_config.async_scheduling
        assert not engine.scheduler.config.enable_preemption
        assert not engine.cache_manager.enable_prefix_caching
        self.engine = engine
        self.events = []
        self.admissions = []
        self.deliveries = []
        self.commit = engine.scheduler._commit_admission
        self.schedule = engine.scheduler.schedule
        self.collect = engine._collect_coda
        self.deliver = engine._deliver_coda
        engine.scheduler._commit_admission = self.record_admission
        engine.scheduler.schedule = self.record_schedule
        engine._collect_coda = self.record_collect
        engine._deliver_coda = self.record_delivery

    def record_admission(self, request, plan):
        admitted = self.commit(request, plan)
        if admitted:
            self.admissions.append(request.request_id)
        return admitted

    def record_schedule(self, *, prefer_recurrent=False):
        batch = self.schedule(prefer_recurrent=prefer_recurrent)
        bucket = (
            self.engine.model_runner._size(len(batch.items))
            if batch is not None and batch.stage == Stage.RECURRENT
            else None
        )
        self.events.append(
            {
                "kind": "schedule",
                "admitted": self.admissions,
                "batch": signature(batch),
                "graph_bucket": bucket,
            }
        )
        self.admissions = []
        return batch

    def record_delivery(self, ticket):
        self.deliveries.append(ticket_key(ticket))
        return self.deliver(ticket)

    def record_collect(self, wait=False):
        outputs = self.collect(wait=wait)
        self.events.append({"kind": "collect", "wait": wait, "delivered": self.deliveries})
        self.deliveries = []
        return outputs


class Replay:
    """Force logical decisions while retaining the runner's real gate computation."""

    def __init__(self, engine: LLMEngine, events: list[dict], exits: dict[str, list[int]]):
        assert engine.execution_config.async_scheduling
        assert not engine.scheduler.config.enable_preemption
        assert not engine.cache_manager.enable_prefix_caching
        assert not engine.has_unfinished_requests()
        assert engine.model_runner.exit_config.mode == "ouro_delayed"
        self.engine, self.events, self.index = engine, events, 0
        # Only the engine's policy is forced. Runner/Graph gate kernels stay enabled.
        engine.exit_config = ExitConfig("trace", depths_by_request=exits)
        engine._exit_traces = {key: tuple(values) for key, values in exits.items()}
        engine.scheduler.schedule = self.replay_schedule
        engine._collect_coda = self.replay_collect

    def next_event(self, kind):
        event = self.events[self.index]
        assert event["kind"] == kind, (self.index, event["kind"], kind)
        self.index += 1
        return event

    def replay_collect(self, wait=False):
        event = self.next_event("collect")
        assert event["wait"] == wait
        outputs = []
        for key in event["delivered"]:
            matches = [t for t in self.engine._pending_coda if ticket_key(t) == key]
            assert len(matches) == 1, key
            ticket = matches[0]
            outputs.extend(self.engine._deliver_coda(ticket))
            self.engine._pending_coda.remove(ticket)
        return outputs

    def replay_schedule(self, *, prefer_recurrent=False):
        event = self.next_event("schedule")
        scheduler = self.engine.scheduler
        scheduler.selected_request_ids.clear()
        for rid in event["admitted"]:
            request = scheduler.requests[rid]
            assert request.stage == Stage.WAITING
            scheduler.queues[Stage.WAITING].remove(rid)
            plan = scheduler._plan_admission(request, len(self.engine.cache_manager._allocations))
            assert scheduler._commit_admission(request, plan), rid
        logical = event["batch"]
        if logical is None:
            return None
        stage, rows = Stage(logical[0]), logical[1]
        items = []
        for rid, position, depth, output_index, start, count in rows:
            request = scheduler.requests[rid]
            assert request.stage == stage
            assert (request.position, request.loops_done, request.num_scheduled_outputs) == (
                position,
                depth,
                output_index,
            )
            scheduler.queues[stage].remove(rid)
            if stage != Stage.CODA:
                frontier = start + count if stage == Stage.PREFILL else position + 1
                assert scheduler._ensure_execution_capacity(request, frontier)
            scheduler.selected_request_ids.add(rid)
            items.append(ScheduledItem(request, start, count))
        if stage == Stage.RECURRENT:
            assert self.engine.model_runner._size(len(items)) == event["graph_bucket"]
        scheduler.policy.record_batch(stage)
        return SchedulerOutput(stage, items)

    def assert_drained(self):
        assert self.index == len(self.events)
        assert not self.engine.has_unfinished_requests()
        assert not self.engine._pending_coda
        assert self.engine.cache_manager.num_free_blocks == self.engine.cache_manager.num_blocks
