"""One cold prefill per queued prompt group, with independent continuations."""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING
from uuid import uuid4

import torch

from vllm_rlt.request import Request, Stage

if TYPE_CHECKING:
    from vllm_rlt.core.scheduler import Scheduler
    from vllm_rlt.worker.model_runner import ModelRunner


class ForkAdmission(Enum):
    NORMAL = auto()
    DEFER = auto()
    FORKED = auto()


@dataclass
class _Group:
    key: tuple[int, tuple[int, ...]]
    waiting: dict[str, Request] = field(default_factory=dict)
    leader: Request | None = None
    snapshot_id: str | None = None
    hidden: torch.Tensor | None = None


class GroupPrefillFork:
    def __init__(self, scheduler: "Scheduler", runner: "ModelRunner", depth: int):
        self.scheduler, self.runner, self.depth = scheduler, runner, depth
        self.cache = scheduler.cache_manager
        self.groups: dict[tuple[int, tuple[int, ...]], _Group] = {}
        self.by_request: dict[str, _Group] = {}
        self.forked_requests = self.reused_tokens = self.capacity_fallbacks = 0

    def add(self, request: Request) -> None:
        key = (request.weight_version, tuple(request.prompt_token_ids))
        group = self.groups.setdefault(key, _Group(key))
        group.waiting[request.request_id] = request
        self.by_request[request.request_id] = group

    def started(self, request: Request) -> None:
        group = self.by_request.get(request.request_id)
        if group is not None:
            group.waiting.pop(request.request_id)
            group.leader = request

    def _drop(self, group: _Group) -> None:
        if group.snapshot_id is not None:
            self.cache.free(group.snapshot_id)
        self.groups.pop(group.key)
        if group.leader is not None:
            self.by_request.pop(group.leader.request_id, None)
        for request_id in group.waiting:
            self.by_request.pop(request_id)

    def remove(self, request: Request) -> None:
        group = self.by_request.pop(request.request_id, None)
        if group is None:
            return
        group.waiting.pop(request.request_id, None)
        if group.leader is request:
            group.leader = None
        if not group.waiting:
            self._drop(group)

    def _reserved(self) -> int:
        return self.scheduler._reserved_growth_blocks(
            reserve_outputs=self.cache.incremental_allocation
            and self.scheduler.preempt_callback is None
        )

    def reclaim(self) -> bool:
        """Pending followers may recompute when their snapshot blocks admission."""
        for group in self.groups.values():
            if group.snapshot_id is not None:
                self.capacity_fallbacks += 1
                self._drop(group)
                return True
        return False

    @torch.inference_mode()
    def capture(self, request: Request) -> None:
        group = self.by_request.get(request.request_id)
        if group is None or group.leader is not request:
            return
        if not group.waiting:
            self._drop(group)
            return
        length = len(request.prompt_token_ids)
        # Sharing the leader's partial page adds a future private-page demand.
        extra = (
            self.cache.storage_depths
            if length % self.cache.block_size and request.sampling_params.max_tokens > 1
            else 0
        )
        if extra + self._reserved() > self.cache.num_free_blocks:
            self.capacity_fallbacks += 1
            self._drop(group)
            return
        snapshot = "group-prefill-" + uuid4().hex
        assert self.cache.fork_prefix(request.request_id, snapshot, length, length)
        assert request.hidden_state is not None
        if self.runner.device.type == "cuda":
            stream = torch.cuda.current_stream(self.runner.device)
            event = self.runner.events.get(request.request_id)
            if event is not None:
                stream.wait_event(event)
            request.hidden_state.record_stream(stream)
        group.hidden = request.hidden_state.clone()
        if self.runner.device.type == "cuda":
            event = torch.cuda.Event()
            event.record(stream)
            self.runner.events[request.request_id] = event
        group.snapshot_id = snapshot
        group.leader = None
        self.by_request.pop(request.request_id)

    @torch.inference_mode()
    def admit(self, request: Request, active_count: int) -> ForkAdmission:
        group = self.by_request.get(request.request_id)
        if group is None:
            return ForkAdmission.NORMAL
        if group.snapshot_id is None:
            return ForkAdmission.DEFER if group.leader is not None else ForkAdmission.NORMAL
        length = len(request.prompt_token_ids)
        capacity = length + request.sampling_params.max_tokens - 1
        budget = (
            length
            if self.cache.incremental_allocation and self.scheduler.preempt_callback is not None
            else capacity
        )
        prefix_pages = (length + self.cache.block_size - 1) // self.cache.block_size
        required = self.cache.required_blocks(budget) - prefix_pages * self.cache.storage_depths
        if length % self.cache.block_size and budget > length:
            required += self.cache.storage_depths
        headroom = self.cache.watermark_blocks if active_count else 0
        if required + self._reserved() + headroom > self.cache.num_free_blocks:
            if active_count == 0:
                self.capacity_fallbacks += 1
                self._drop(group)
                return ForkAdmission.NORMAL
            return ForkAdmission.DEFER
        if not self.cache.fork_prefix(group.snapshot_id, request.request_id, length, capacity):
            return ForkAdmission.DEFER
        assert group.hidden is not None
        if self.runner.device.type == "cuda":
            group.hidden.record_stream(torch.cuda.current_stream(self.runner.device))
        self.runner._save(request, group.hidden)
        request.num_prefilled_tokens = length
        request.loops_done = self.depth
        self.scheduler.enqueue(request, Stage.CODA)
        self.forked_requests += 1
        self.reused_tokens += length
        group.waiting.pop(request.request_id)
        self.by_request.pop(request.request_id)
        if not group.waiting:
            self._drop(group)
        return ForkAdmission.FORKED
