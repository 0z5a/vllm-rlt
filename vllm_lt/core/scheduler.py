"""CPU scheduling at stage/loop boundaries, independent of model execution."""

from collections import deque
from dataclasses import dataclass

from vllm_lt.config import SchedulerConfig
from vllm_lt.request import Request, Stage


@dataclass(frozen=True)
class ScheduledItem:
    request: Request
    # Used by prefill only; decode always schedules one position per request.
    token_start: int = 0
    token_count: int = 1


@dataclass(frozen=True)
class SchedulerOutput:
    stage: Stage
    items: list[ScheduledItem]

    @property
    def num_tokens(self) -> int:
        return sum(item.token_count for item in self.items)


class Scheduler:
    def __init__(self, config: SchedulerConfig, cache_manager):
        self.config = config
        self.cache_manager = cache_manager
        self.requests: dict[str, Request] = {}
        self.queues: dict[Stage, deque[str]] = {s: deque() for s in Stage}
        self.protected = set()
        self.preempt_callback = None
        self.resume_callback = None
        self._no_refill_phase = "fill"
        self._prefill_batches_since_recurrent = 0

    def add_request(self, request: Request):
        if request.request_id in self.requests:
            raise ValueError(f"duplicate request ID: {request.request_id}")
        self.requests[request.request_id] = request
        self.queues[Stage.WAITING].append(request.request_id)

    def enqueue(self, request: Request, stage: Stage):
        request.stage = stage
        self.queues[stage].append(request.request_id)

    def finish(self, request: Request, reason: str):
        # A delayed EOS can stop a request already queued for its next stage.
        for queue in self.queues.values():
            while request.request_id in queue:
                queue.remove(request.request_id)
        self.cache_manager.free(request.request_id)
        request.stage = Stage.FINISHED
        request.finish_reason = reason
        request.hidden_state = None
        request.input_token_tensor = None
        request.num_output_placeholders = 0
        request.generator = None
        self.requests.pop(request.request_id)

    def abort(self, request_id: str) -> Request:
        request = self.requests[request_id]
        for queue in self.queues.values():
            try:
                queue.remove(request_id)
            except ValueError:
                pass
        self.finish(request, "abort")
        return request

    @property
    def has_unfinished_requests(self) -> bool:
        return bool(self.requests)

    def _admit(self):
        active = sum(
            r.stage not in (Stage.WAITING, Stage.RECEIVING) for r in self.requests.values()
        )
        waiting = self.queues[Stage.WAITING]
        if self.config.policy == "priority":
            ordered = sorted(waiting, key=lambda rid: self.requests[rid].sampling_params.priority)
            waiting.clear()
            waiting.extend(ordered)
        deferred = []
        # Scan a bounded window, preserving relative order of waiting items.
        for _ in range(min(len(waiting), self.config.admission_scan_limit)):
            if active >= self.config.max_num_seqs:
                if not (
                    self.config.policy == "priority"
                    and self.preempt_callback
                    and waiting
                    and self.preempt_callback(self.requests[waiting[0]], priority_only=True)
                ):
                    break
                active -= 1
            request_id = waiting.popleft()
            request = self.requests[request_id]
            capacity = len(request.prompt_token_ids) + request.sampling_params.max_tokens - 1
            cache = self.cache_manager
            restored = self.resume_callback(request) if self.resume_callback else None
            if restored is not None:
                if restored:
                    active += 1
                else:
                    deferred.append(request_id)
                continue
            prefix = cache.lookup_prefix(request.prompt_token_ids)
            hit = len(prefix) * cache.block_size
            initial = min(len(request.prompt_token_ids), hit + self.config.prefill_chunk_size)
            if not cache.incremental_allocation:
                initial = capacity
            # Without pressure preemption, reserve future growth for all admitted
            # requests; with it, preserve at least every in-flight prompt.
            reserve_outputs = cache.incremental_allocation and not self.preempt_callback
            reserved = sum(
                max(
                    0,
                    cache.required_blocks(
                        len(r.prompt_token_ids)
                        + (r.sampling_params.max_tokens - 1 if reserve_outputs else 0)
                    )
                    - len(cache._get_allocation(r.request_id).block_tables[0])
                    * cache.storage_depths,
                )
                for r in self.requests.values()
                if r.stage == Stage.PREFILL
                or (reserve_outputs and r.stage not in (Stage.WAITING, Stage.RECEIVING))
            )
            required = (
                cache.required_blocks(
                    len(request.prompt_token_ids)
                    if cache.incremental_allocation and not reserve_outputs
                    else capacity
                )
                - len(prefix) * cache.storage_depths
            )
            headroom = cache.watermark if active else 0
            cached_claim = sum(cache._refs[b] == 1 for group in prefix for b in group)
            fits = required + reserved + headroom + cached_claim <= cache.num_free_blocks
            if not fits or not cache.allocate(
                request_id, capacity, initial_tokens=initial, prefix=prefix
            ):
                deferred.append(request_id)
                if request.admission_bypasses >= self.config.max_admission_bypasses:
                    break  # Protect this request while existing work drains.
                continue
            for blocked_id in deferred:
                self.requests[blocked_id].admission_bypasses += 1
            request.num_prefilled_tokens = hit
            self.enqueue(request, Stage.PREFILL)
            active += 1
            if any(
                self.requests[rid].admission_bypasses >= self.config.max_admission_bypasses
                for rid in deferred
            ):
                break
        waiting.extendleft(reversed(deferred))

    def _take(self, stage: Stage) -> SchedulerOutput:
        if stage == Stage.PREFILL:
            self._prefill_batches_since_recurrent += 1
        elif stage == Stage.RECURRENT:
            self._prefill_batches_since_recurrent = 0
        budget = self.config.max_num_batched_tokens
        items = []
        queue = self.queues[stage]
        remaining = len(queue)
        while queue and budget and len(items) < self.config.max_num_seqs and remaining:
            remaining -= 1
            request = self.requests[queue.popleft()]
            if stage == Stage.PREFILL:
                start = request.num_prefilled_tokens
                count = min(
                    budget, self.config.prefill_chunk_size, len(request.prompt_token_ids) - start
                )
            else:
                start, count = 0, 1
            frontier = start + count if stage == Stage.PREFILL else request.position + 1
            if stage in (Stage.PREFILL, Stage.PRELUDE, Stage.RECURRENT):
                if not self.cache_manager.ensure_capacity(request.request_id, frontier):
                    if not (self.preempt_callback and self.preempt_callback(request)):
                        queue.append(request.request_id)
                        continue
                    if not self.cache_manager.ensure_capacity(request.request_id, frontier):
                        queue.append(request.request_id)
                        continue
            self.protected.add(request.request_id)
            items.append(ScheduledItem(request, start, count))
            budget -= count
        return SchedulerOutput(stage, items) if items else None

    def schedule(self, *, prefer_recurrent=False) -> SchedulerOutput | None:
        self.protected.clear()
        if not self.requests:
            return None
        q = self.queues
        if self.config.mode == "no_refill":
            if self._no_refill_phase == "core":
                if q[Stage.RECURRENT]:
                    return self._take(Stage.RECURRENT)
                self._no_refill_phase = "coda"
            if self._no_refill_phase == "coda":
                if q[Stage.CODA]:
                    return self._take(Stage.CODA)
                self._no_refill_phase = "fill"
            # Continuous short arrivals must not indefinitely postpone an
            # existing decode. Allow at most one prefill batch between loops.
            if (
                self._prefill_batches_since_recurrent
                >= self.config.max_prefill_batches_before_decode
            ):
                if q[Stage.PRELUDE]:
                    return self._take(Stage.PRELUDE)
                if q[Stage.RECURRENT]:
                    self._no_refill_phase = "core"
                    return self._take(Stage.RECURRENT)
            self._admit()
            if q[Stage.PREFILL]:
                return self._take(Stage.PREFILL)
            if q[Stage.CODA]:  # first output after full-depth prompt prefill
                return self._take(Stage.CODA)
            if q[Stage.PRELUDE]:
                return self._take(Stage.PRELUDE)
            if q[Stage.RECURRENT]:
                self._no_refill_phase = "core"
                return self._take(Stage.RECURRENT)
        else:
            # In multi-stream execution, give an independent core batch one
            # turn alongside newly submitted boundary work, then refill it.
            if prefer_recurrent and q[Stage.RECURRENT]:
                return self._take(Stage.RECURRENT)
            # A prelude created by coda runs immediately, returning tokens to the core.
            if q[Stage.PRELUDE]:
                return self._take(Stage.PRELUDE)
            if q[Stage.CODA] and (
                len(q[Stage.CODA]) >= self.config.min_coda_batch_size or not q[Stage.RECURRENT]
            ):
                return self._take(Stage.CODA)
            if (
                self._prefill_batches_since_recurrent
                >= self.config.max_prefill_batches_before_decode
                and q[Stage.RECURRENT]
            ):
                return self._take(Stage.RECURRENT)
            self._admit()
            if q[Stage.PREFILL]:
                return self._take(Stage.PREFILL)
            if q[Stage.RECURRENT]:
                return self._take(Stage.RECURRENT)
        return None
