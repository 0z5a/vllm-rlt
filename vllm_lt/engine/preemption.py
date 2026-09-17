"""Lossless pressure preemption for recurrent-depth KV.

CPU snapshots retain exact KV planes, hidden state and request RNG ownership.
Full-depth replay of generated tokens is deliberately avoided: it changes RLT
semantics. Snapshots are bounded by the admitted request population.
"""

from copy import deepcopy

import torch

from vllm_lt.request import Stage


class PreemptionManager:
    def __init__(self, engine):
        self.engine = engine
        self.snapshots = {}
        self.preemptions = self.resumptions = 0

    def preempt(self, requester, *, priority_only=False):
        e = self.engine
        candidates = [
            r
            for r in e.scheduler.requests.values()
            if r is not requester
            and r.stage not in (Stage.WAITING, Stage.RECEIVING)
            and not r.num_output_placeholders
            and r.request_id not in e.scheduler.protected
            and not e.cache_manager._get_allocation(r.request_id).transfer_leases
        ]
        if priority_only:
            candidates = [
                r
                for r in candidates
                if r.sampling_params.priority > requester.sampling_params.priority
            ]
        if not candidates:
            return False
        victim = max(
            candidates,
            key=lambda r: (
                r.sampling_params.priority,
                e.cache_manager.required_blocks(r.position + 1),
            ),
        )
        e.model_runner.synchronize()
        cache = e.cache_manager
        allocation = cache._get_allocation(victim.request_id)
        blocks = [b for table in allocation.block_tables for b in table]
        # Copy views one page at a time: a pressure recovery must not allocate
        # another request-sized temporary on an already full GPU.
        keys = torch.empty((len(blocks), *cache.key_cache.shape[1:]), dtype=cache.dtype)
        values = torch.empty_like(keys)
        for row, block in enumerate(blocks):
            keys[row].copy_(cache.key_cache[block])
            values[row].copy_(cache.value_cache[block])
        snapshot = dict(
            stage=victim.stage,
            maximum=allocation.max_tokens,
            pages=len(allocation.block_tables[0]),
            written=deepcopy(allocation.written),
            keys=keys,
            values=values,
            hidden=None if victim.hidden_state is None else victim.hidden_state.cpu().clone(),
            token=None
            if victim.input_token_tensor is None
            else victim.input_token_tensor.cpu().clone(),
        )
        e.model_runner.release(victim.request_id)
        cache.poll_prefixes()
        cache.free(victim.request_id)
        for q in e.scheduler.queues.values():
            while victim.request_id in q:
                q.remove(victim.request_id)
        victim.hidden_state = victim.input_token_tensor = None
        self.snapshots[victim.request_id] = snapshot
        e.scheduler.enqueue(victim, Stage.WAITING)
        self.preemptions += 1
        return True

    def resume(self, request):
        state = self.snapshots.get(request.request_id)
        if state is None:
            return None
        e, cache = self.engine, self.engine.cache_manager
        frontier = min(state["maximum"], state["pages"] * cache.block_size)
        if not cache.allocate(request.request_id, state["maximum"], initial_tokens=frontier):
            return False
        allocation = cache._get_allocation(request.request_id)
        blocks = [b for table in allocation.block_tables for b in table]
        for row, block in enumerate(blocks):
            cache.key_cache[block].copy_(state["keys"][row])
            cache.value_cache[block].copy_(state["values"][row])
        allocation.written = state["written"]
        request.hidden_state = None if state["hidden"] is None else state["hidden"].to(cache.device)
        request.input_token_tensor = (
            None if state["token"] is None else state["token"].to(cache.device)
        )
        # Restored buffers become visible to every runner stream before resumption.
        if cache.device.type == "cuda":
            torch.cuda.current_stream(cache.device).synchronize()
        e.scheduler.enqueue(request, state["stage"])
        del self.snapshots[request.request_id]
        self.resumptions += 1
        return True
