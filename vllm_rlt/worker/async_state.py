"""Stable request slots and event-owned routing banks for asynchronous CUDA work.

Hot state and page tables remain on device. Small CPU routing descriptions live
in mapped pinned memory (UVA); the H2D fallback uses the same bank ownership.
No bank or request slot may be reused until its last GPU reader completes.
"""

import math
from dataclasses import replace

import torch

from vllm_rlt.core.alias_kv_cache import AliasKVCacheManager
from vllm_rlt.core.compact_kv_cache import CompactKVCacheManager
from vllm_rlt.core.kv_cache_manager import _PreparedKVBatch


class RoutingBank:
    def __init__(self, owner, rows, *, control_only=False):
        self.owner = owner
        columns = 6 if owner.compact else 4 if owner.alias else 3
        self.host = torch.empty((rows, columns), dtype=torch.int64, pin_memory=True)
        self.done = self.ready_event = None
        self.uploads = []
        self.imports = []
        self.dependencies = []
        self.descriptor = None
        if control_only:
            return
        device = owner.cache.device
        self.slots = torch.empty(rows, dtype=torch.int64, device=device)
        self.positions = torch.empty_like(self.slots)
        self.blocks = torch.empty_like(self.slots)
        self.offsets = torch.empty_like(self.slots)
        self.lengths = torch.empty(rows, dtype=torch.int32, device=device)
        self.tables = torch.empty((rows, owner.width), dtype=torch.int32, device=device)
        self.depth_tables = self.depths = self.alias_starts = None
        self.map_pointers = self.map_widths = None
        if owner.compact:
            self.map_pointers = torch.empty_like(self.slots)
            self.map_widths = torch.empty_like(self.lengths)
            self.depths = torch.empty_like(self.lengths)
        if owner.alias:
            self.depth_tables = torch.empty(
                (rows, owner.planes, owner.width), dtype=torch.int32, device=device
            )
            self.depths = torch.empty(rows, dtype=torch.int32, device=device)
            self.alias_starts = torch.zeros(rows, dtype=torch.int32, device=device)
        self.hidden = torch.empty(
            (rows, owner.hidden.shape[1]), dtype=owner.hidden.dtype, device=device
        )
        self.tokens = torch.empty((rows, 1), dtype=torch.int64, device=device)
        self.descriptor = None

    def acquire(self):
        if self.done is not None:
            self.done.synchronize()
        self.uploads.clear()
        self.imports.clear()
        self.dependencies.clear()
        self.descriptor = None

    def transfer(self, batch=None):
        from vllm_rlt.kernels.routing import compact_metadata_kernel, metadata_kernel

        for ready in self.dependencies:
            torch.cuda.current_stream(self.owner.cache.device).wait_event(ready)
        for slot, table in self.uploads:
            self.owner.tables[slot, :, : table.shape[1]].copy_(table, non_blocking=True)
        for slot, hidden in self.imports:
            hidden.record_stream(torch.cuda.current_stream(self.owner.cache.device))
            self.owner.hidden[slot].copy_(hidden)
        self.descriptor = (
            self.host
            if self.owner.use_uva
            else self.host[: self.count].to(self.owner.cache.device, non_blocking=True)
        )
        if self.owner.compact:
            compact_metadata_kernel[(self.size,)](
                self.descriptor,
                self.slots,
                self.positions,
                self.lengths,
                self.blocks,
                self.offsets,
                self.depths,
                self.map_pointers,
                self.map_widths,
                self.count,
                self.owner.cache.block_size,
                batch is not None,
            )
        else:
            metadata_kernel[(self.size,)](
                self.descriptor,
                self.owner.tables,
                self.slots,
                self.positions,
                self.lengths,
                self.blocks,
                self.offsets,
                self.tables,
                self.depth_tables,
                self.depths,
                self.alias_starts,
                self.count,
                self.size,
                self.owner.width,
                self.owner.cache.block_size,
                self.owner.planes,
                self.width,
                256,
                self.owner.alias and batch is not None,
                self.host.shape[1],
            )
        self.ready_event = torch.cuda.Event()
        self.ready_event.record(torch.cuda.current_stream(self.owner.cache.device))
        if batch is None:
            return None
        return replace(
            batch,
            position_ids=self.positions[: self.size],
            context_lengths=self.lengths[: self.size],
            write_blocks=self.blocks[: self.count],
            write_offsets=self.offsets[: self.count],
            block_tables=self.tables[: self.size, : self.width],
            depth_block_tables=self.depth_tables[: self.size, :, : self.width]
            if self.owner.alias
            else None,
            query_depths=self.depths[: self.size]
            if self.owner.alias or self.owner.compact
            else None,
            alias_starts=self.alias_starts[: self.size] if self.owner.alias else None,
            record_map_pointers=self.map_pointers[: self.size] if self.owner.compact else None,
            record_map_widths=self.map_widths[: self.size] if self.owner.compact else None,
        )

    def record_done(self):
        self.done = torch.cuda.Event()
        self.done.record(torch.cuda.current_stream(self.owner.cache.device))

    def gather(self, tokens=False):
        from vllm_rlt.kernels.routing import gather_kernel

        pool = self.owner.tokens if tokens else self.owner.hidden
        output = self.tokens if tokens else self.hidden
        h = pool.shape[1]
        gather_kernel[(self.size, math.ceil(h / 256))](
            pool, self.slots, output, self.count, h, pool.stride(0), output.stride(0), 256
        )
        return output[: self.size, 0] if tokens else output[: self.size]

    def scatter(self, values, tokens=False):
        from vllm_rlt.kernels.routing import scatter_kernel

        pool = self.owner.tokens if tokens else self.owner.hidden
        values = values.reshape(-1, 1) if tokens else values
        h = pool.shape[1]
        scatter_kernel[(self.count, math.ceil(h / 256))](
            values, self.slots, pool, h, values.stride(0), pool.stride(0), 256
        )


class AsyncState:
    def __init__(self, cache, config, scheduler, rows, *, use_uva=True):
        self.cache = cache
        self.compact = isinstance(cache, CompactKVCacheManager)
        self.alias = isinstance(cache, AliasKVCacheManager) and not self.compact
        self.use_uva = use_uva
        self.width = (
            0 if self.compact else math.ceil(config.max_position_embeddings / cache.block_size)
        )
        self.planes = config.total_ut_steps if cache.layout == "last_exited" else 1
        self.hidden = torch.empty(
            (scheduler.max_num_seqs, config.hidden_size),
            dtype=cache.key_cache.dtype,
            device=cache.device,
        )
        self.tokens = torch.empty(
            (scheduler.max_num_seqs, 1), dtype=torch.int64, device=cache.device
        )
        self.tables = torch.zeros(
            (scheduler.max_num_seqs, self.planes, self.width),
            dtype=torch.int32,
            device=cache.device,
        )
        self.slots = {}
        self.table_versions = {}
        self.owners = {}
        self.free = list(reversed(range(scheduler.max_num_seqs)))
        self.banks = [RoutingBank(self, rows) for _ in range(4)]
        self.finalize_banks = [
            RoutingBank(self, scheduler.max_num_seqs, control_only=True) for _ in range(2)
        ]
        self.index = self.finalize_index = 0

    def ensure_slot(self, request, bank):
        rid = request.request_id
        allocation = self.cache._get_allocation(rid)
        if rid in self.slots:
            if self.owners[rid] != (id(request), id(allocation)):
                raise RuntimeError("request slot reused before retiring its previous owner")
            slot = self.slots[rid]
            if not self.compact and self.table_versions.get(rid) != allocation.block_tables:
                table = torch.tensor(allocation.block_tables, dtype=torch.int32).pin_memory()
                bank.uploads.append((slot, table))
                self.table_versions[rid] = allocation.block_tables
            return slot
        if not self.free:
            raise RuntimeError("no retired request state slots")
        slot = self.free.pop()
        self.slots[rid] = slot
        self.owners[rid] = (id(request), id(allocation))
        if self.alias or self.compact:
            bank.dependencies.append(self.cache._allocation_ready[id(allocation)])
        if not self.compact:
            table = torch.tensor(allocation.block_tables, dtype=torch.int32).pin_memory()
            bank.uploads.append((slot, table))
            self.table_versions[rid] = allocation.block_tables
        if request.hidden_state is not None:
            bank.imports.append((slot, request.hidden_state))
            request.hidden_state = self.hidden[slot]
        return slot

    def prepare(self, requests, depths, positions, size, *, recurrent=False, finalize=False):
        if finalize:
            bank = self.finalize_banks[self.finalize_index]
            self.finalize_index = (self.finalize_index + 1) % len(self.finalize_banks)
        else:
            bank = self.banks[self.index]
            self.index = (self.index + 1) % len(self.banks)
        bank.acquire()
        bank.count, bank.size = len(requests), size
        bank.width = 0 if self.compact else max(p // self.cache.block_size + 1 for p in positions)
        ids = [r.request_id for r in requests]
        rows = tuple(self.cache._validate_rows(ids, depths, positions)) if recurrent else ()
        descriptors = [
            (self.ensure_slot(r, bank), d if recurrent or finalize else 0, p)
            for r, d, p in zip(requests, depths, positions)
        ]
        if self.alias:
            descriptors = [
                (
                    *row,
                    self.cache._alias_starts[id(self.cache._get_allocation(request.request_id))][
                        depth
                    ]
                    if recurrent
                    else 0,
                )
                for request, depth, row in zip(requests, depths, descriptors)
            ]
        if self.compact:
            records = self.cache._reserve_records(rows) if recurrent else [-1] * len(requests)
            allocations = [self.cache._get_allocation(rid) for rid in ids]
            descriptors = [
                (*row, self.cache._maps[id(a)].data_ptr(), a.max_tokens, record)
                for row, a, record in zip(descriptors, allocations, records)
            ]
        # A normal list is CPU-owned; only the contiguous pinned snapshot is GPU-readable.
        bank.host.numpy()[: len(descriptors)] = descriptors
        batch = None
        if recurrent:
            if not self.compact:
                addresses = [
                    (
                        a.block_tables[self.cache._plane(d)][p // self.cache.block_size],
                        p % self.cache.block_size,
                    )
                    for a, d, p in rows
                ]
                if len(set(addresses)) != len(addresses):
                    raise ValueError("duplicate KV write addresses")
            batch = _PreparedKVBatch(
                owner=self.cache,
                rows=rows,
                allocations=tuple(dict(zip(ids, (a for a, _, _ in rows))).items()),
                position_ids=bank.host[: len(rows), 2],
                write_blocks=bank.blocks[: len(rows)],
                write_offsets=bank.offsets[: len(rows)],
                block_tables=bank.tables[:size, : bank.width],
                context_lengths=bank.lengths[:size],
                writable=True,
            )
        return bank, batch

    def release(self, rid):
        self.table_versions.pop(rid, None)
        self.owners.pop(rid, None)
        slot = self.slots.pop(rid, None)
        if slot is not None:
            self.free.append(slot)
