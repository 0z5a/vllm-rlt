"""Reusable stage inputs and metadata; padding never acquires a KV address."""

import math
from dataclasses import replace

import torch

from vllm_lt.core.kv_cache_manager import _PreparedKVBatch


def execution_buffer_bytes(config, scheduler, cache, execution, element_size):
    if not execution.static_buffers:
        return 0
    rows = scheduler.max_num_batched_tokens
    if execution.pad_to_power_of_two:
        rows = 1 << (rows - 1).bit_length()
    width = math.ceil(config.max_position_embeddings / cache.block_size)
    # Two banks each for core and boundary execution, plus persistent request states.
    return 4 * rows * (config.hidden_size * element_size + 4 * width + 36) + (
        scheduler.max_num_seqs * config.hidden_size * element_size
    )


class Workspace:
    def __init__(self, rows, width, hidden_size, device, dtype):
        self.hidden = torch.empty((rows, hidden_size), device=device, dtype=dtype)
        self.device = device
        self.event = None
        shapes = dict(
            tokens=(rows,),
            positions=(rows,),
            blocks=(rows,),
            offsets=(rows,),
            tables=(rows, width),
            lengths=(rows,),
        )
        self.host = {}
        self.gpu = {}
        for name, shape in shapes.items():
            kind = torch.int32 if name in ("tables", "lengths") else torch.int64
            self.host[name] = torch.empty(shape, dtype=kind, pin_memory=device.type == "cuda")
            self.gpu[name] = torch.empty(shape, dtype=kind, device=device)

    def acquire(self):
        if self.event is not None:
            self.event.synchronize()  # Protect host DMA inputs as well as device scratch.

    def release(self):
        if self.device.type == "cuda":
            self.event = torch.cuda.Event()
            self.event.record(torch.cuda.current_stream(self.device))

    def tokens(self, ids, size):
        host = self.host["tokens"][:size]
        host.zero_()
        host[: len(ids)] = torch.tensor(ids, dtype=host.dtype)
        self.gpu["tokens"][:size].copy_(host, non_blocking=True)
        return self.gpu["tokens"][:size]

    def prepare(self, cache, ids, depths, positions, size):
        rows = tuple(cache._validate_rows(ids, depths, positions))
        addresses = [
            (a.block_tables[cache._plane(d)][p // cache.block_size], p % cache.block_size)
            for a, d, p in rows
        ]
        if len(set(addresses)) != len(addresses):
            raise ValueError("duplicate KV write addresses")
        n = len(rows)
        for name in ("positions", "lengths", "tables"):
            self.host[name][:size].zero_()
        for index, ((allocation, depth, pos), (block, offset)) in enumerate(zip(rows, addresses)):
            self.host["positions"][index] = pos
            self.host["lengths"][index] = pos + 1
            self.host["blocks"][index] = block
            self.host["offsets"][index] = offset
            table = allocation.block_tables[cache._plane(depth)]
            self.host["tables"][index, : len(table)] = torch.tensor(table, dtype=torch.int32)
        for name in ("positions", "lengths", "tables", "blocks", "offsets"):
            count = n if name in ("blocks", "offsets") else size
            self.gpu[name][:count].copy_(self.host[name][:count], non_blocking=True)
        return _PreparedKVBatch(
            owner=cache,
            rows=rows,
            allocations=tuple(dict(zip(ids, (a for a, _, _ in rows))).items()),
            position_ids=self.gpu["positions"][:size],
            write_blocks=self.gpu["blocks"][:n],
            write_offsets=self.gpu["offsets"][:n],
            block_tables=self.gpu["tables"][:size],
            context_lengths=self.gpu["lengths"][:size],
            writable=True,
        )


class InputStaging:
    """Pinned host DMA inputs, independent of static device execution buffers.

    Each submission owns fresh device metadata. A host bank can be overwritten
    after its H2D event, without waiting for the model forward to complete.
    """

    def __init__(self, rows, width):
        self.event = None
        # Two contiguous copies instead of one transfer per metadata field.
        self.longs = torch.empty(3 * rows, dtype=torch.int64, pin_memory=True)
        self.ints = torch.empty(rows * (width + 1), dtype=torch.int32, pin_memory=True)

    def prepare(self, cache, ids, depths, positions, size):
        if self.event is not None:
            self.event.synchronize()  # Only H2D ownership, never the forward event.
        rows = tuple(cache._validate_rows(ids, depths, positions))
        addresses = [
            (a.block_tables[cache._plane(d)][p // cache.block_size], p % cache.block_size)
            for a, d, p in rows
        ]
        if len(set(addresses)) != len(addresses):
            raise ValueError("duplicate KV write addresses")
        n = len(rows)
        width = max((p // cache.block_size + 1 for _, _, p in rows), default=0)
        longs = [p for _, _, p in rows] + [0] * (size - n)
        longs.extend(b for b, _ in addresses)
        longs.extend(o for _, o in addresses)
        tables = []
        for a, d, _ in rows:
            table = a.block_tables[cache._plane(d)][:width]
            tables.extend(table)
            tables.extend([-1] * (width - len(table)))
        tables.extend([-1] * ((size - n) * width))
        ints = tables + [p + 1 for _, _, p in rows] + [0] * (size - n)
        self.longs[: len(longs)].copy_(torch.tensor(longs, dtype=torch.int64))
        self.ints[: len(ints)].copy_(torch.tensor(ints, dtype=torch.int32))
        return _PreparedKVBatch(
            owner=cache,
            rows=rows,
            allocations=tuple(dict(zip(ids, (a for a, _, _ in rows))).items()),
            position_ids=self.longs[:size],
            write_blocks=self.longs[size : size + n],
            write_offsets=self.longs[size + n : size + 2 * n],
            block_tables=self.ints[: size * width].view(size, width),
            context_lengths=self.ints[size * width : size * (width + 1)],
            writable=True,
        )

    def transfer(self, batch, device):
        size, width = batch.block_tables.shape
        n = len(batch.rows)
        longs = self.longs[: size + 2 * n].to(device, non_blocking=True)
        ints = self.ints[: size * (width + 1)].to(device, non_blocking=True)
        result = replace(
            batch,
            position_ids=longs[:size],
            write_blocks=longs[size : size + n],
            write_offsets=longs[size + n : size + 2 * n],
            block_tables=ints[: size * width].view(size, width),
            context_lengths=ints[size * width :],
        )
        self.event = torch.cuda.Event()
        self.event.record(torch.cuda.current_stream(device))
        return result
