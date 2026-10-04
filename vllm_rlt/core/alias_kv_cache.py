"""Experimental rectangular LAST_EXITED cache with metadata-only finalization."""

from dataclasses import replace

import torch

from vllm_rlt.core.kv_cache_manager import KVCacheManager, _WrittenPositions


class AliasKVCacheManager(KVCacheManager):
    """Keep every executed version; skipped depths resolve to its canonical source.

    The payload pool remains rectangular. This mode removes promotion writes,
    but does not reduce reserved payload memory. Only eager, non-transfer
    execution is supported until asynchronous descriptors are adapted.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.layout != "last_exited" or self.backend not in ("torch", "triton"):
            raise ValueError("alias storage requires LAST_EXITED with torch or Triton")
        if self.enable_prefix_caching or self.incremental_allocation:
            raise ValueError("alias storage currently requires fixed unshared allocations")
        self.source_depths = torch.full(
            (self.num_blocks, self.block_size), -1, dtype=torch.int32, device=self.device
        )
        self._readable: dict[int, list[list[_WrittenPositions]]] = {}
        self._exits: dict[int, dict[int, int]] = {}
        self._alias_starts: dict[int, list[int]] = {}
        self._allocation_ready: dict[int, torch.cuda.Event] = {}
        self.promotion_copy_bytes = 0
        self.materialization_bytes = 0

    def allocate(self, request_id, max_tokens, *, initial_tokens=None, prefix=()):
        if prefix or (initial_tokens is not None and initial_tokens != max_tokens):
            raise ValueError("alias storage requires a complete private reservation")
        if not super().allocate(request_id, max_tokens):
            return False
        allocation = self._get_allocation(request_id)
        self._readable[id(allocation)] = [
            [_WrittenPositions() for _ in range(self.num_layers)] for _ in range(self.max_loops)
        ]
        self._exits[id(allocation)] = {}
        self._alias_starts[id(allocation)] = [max_tokens] * self.max_loops
        for block in allocation.block_tables[0]:
            self.source_depths[block].fill_(-1)
        if self.device.type == "cuda":
            ready = torch.cuda.Event()
            ready.record(torch.cuda.current_stream(self.device))
            self._allocation_ready[id(allocation)] = ready
        return True

    def free(self, request_id):
        allocation = self._allocations.get(request_id)
        super().free(request_id)
        if allocation is not None:
            self._readable.pop(id(allocation))
            self._exits.pop(id(allocation))
            self._alias_starts.pop(id(allocation))
            self._allocation_ready.pop(id(allocation), None)

    def pin_transfer(self, request_id, transfer_id):
        raise ValueError("alias KV transfer is not implemented")

    def truncate_suffix(self, request_id, frontier):
        raise ValueError("alias speculative rollback is not implemented")

    def _prepare_batches(self, request_ids, depth_sets, positions, **kwargs):
        batches = super()._prepare_batches(request_ids, depth_sets, positions, **kwargs)
        if not batches[0].rows:
            return batches
        width = batches[0].block_tables.shape[1]
        plans = []
        for index, batch in enumerate(batches):
            starts = [self._alias_starts[id(a)][d] for a, d, _ in batch.rows]
            if any(start <= p for start, (_, _, p) in zip(starts, batch.rows)):
                plans.append((index, starts))
        if not plans:
            return batches
        metadata = []
        for allocation, _, _ in batches[0].rows:
            for table in allocation.block_tables:
                row = table[:width]
                metadata.extend((*row, *([-1] * (width - len(row)))))
        table_size = len(metadata)
        for index, starts in plans:
            metadata.extend(depth for _, depth, _ in batches[index].rows)
            metadata.extend(starts)
        staged = self._stage(metadata, torch.int32)
        device_tables = staged[:table_size].reshape(-1, self.max_loops, width)
        result = list(batches)
        offset, rows = table_size, len(batches[0].rows)
        for index, _ in plans:
            result[index] = replace(
                batches[index],
                depth_block_tables=device_tables,
                query_depths=staged[offset : offset + rows],
                alias_starts=staged[offset + rows : offset + 2 * rows],
            )
            offset += 2 * rows
        return tuple(result)

    def _write_prepared(self, layer, batch, k, v):
        self._validate_layer(layer)
        self._require_live_batch(batch)
        boundary = (
            len(self.recurrent_layers) != self.num_layers and layer not in self.recurrent_layers
        )
        for allocation, depth, position in batch.rows:
            if boundary and depth != 0:
                raise ValueError("boundary KV must use depth zero")
            if position in self._exits[id(allocation)] and (
                not boundary or position in allocation.written[0][layer]
            ):
                raise ValueError("cannot overwrite a finalized alias source")
        super()._write_prepared(layer, batch, k, v)
        for allocation, depth, position in batch.rows:
            self._readable[id(allocation)][depth][layer].add(position)

    def _require_prefix(self, allocation, layer, depth, length):
        if length > self._readable[id(allocation)][depth][layer].prefix:
            raise RuntimeError("uninitialized logical KV history")

    def finalize_token(self, request_id, position, exit_depth):
        self.finalize_tokens(((request_id, position, exit_depth),))

    def finalize_tokens(self, tokens):
        targets = []
        for request_id, position, exit_depth in tokens:
            target = self._finalize_metadata(request_id, position, exit_depth)
            if target is not None:
                targets.append((*target, exit_depth))
        if not targets:
            return
        if self.device.type == "cuda":
            from vllm_rlt.kernels.record_map import publish_exits

            count = len(targets)
            packed = self._stage(
                [tensor.data_ptr() + index * tensor.element_size() for tensor, index, _ in targets]
                + [depth for _, _, depth in targets],
                torch.int64,
            )
            publish_exits(packed[:count], packed[count:])
        else:
            for tensor, index, depth in targets:
                tensor.view(-1)[index] = depth

    def _finalize_metadata(self, request_id, position, exit_depth):
        allocation = self._get_allocation(request_id)
        self._validate_depth(exit_depth)
        self._validate_position(allocation, position)
        for depth in range(exit_depth + 1):
            for layer in self.recurrent_layers:
                if position not in allocation.written[depth][layer]:
                    raise RuntimeError("cannot finalize before every executed version is written")
        if position in self._exits[id(allocation)]:
            raise ValueError("position already finalized")
        self._exits[id(allocation)][position] = exit_depth
        for depth in range(exit_depth + 1, self.max_loops):
            starts = self._alias_starts[id(allocation)]
            starts[depth] = min(starts[depth], position)
            for layer in self.recurrent_layers:
                self._readable[id(allocation)][depth][layer].add(position)
        if exit_depth + 1 < self.max_loops:
            block = allocation.block_tables[0][position // self.block_size]
            return self.source_depths, block * self.block_size + position % self.block_size
        return None

    def _attend_prepared(self, layer, batch, q):
        self._validate_layer(layer)
        self._require_live_batch(batch)
        self._validate_tensor(q, len(batch.position_ids), "q", query=True)
        if not batch.rows:
            return torch.empty_like(q)
        for allocation, depth, position in batch.rows:
            self._require_prefix(allocation, layer, depth, position + 1)
        if self.backend == "triton":
            from vllm_rlt.kernels.triton_attention import paged_attention

            return paged_attention(
                q,
                self.key_cache[:, layer],
                self.value_cache[:, layer],
                batch.block_tables,
                batch.context_lengths,
                source_depths=self.source_depths if batch.alias_starts is not None else None,
                depth_block_tables=batch.depth_block_tables,
                query_depths=batch.query_depths,
                alias_starts=batch.alias_starts,
            )
        # CPU verification only. The GPU candidate loads aliased payload directly.
        output = torch.empty_like(q)
        groups = q.shape[1] // self.num_kv_heads
        for row, (allocation, depth, position) in enumerate(batch.rows):
            request_id = next(rid for rid, a in batch.allocations if a is allocation)
            keys, values = self.read(layer, request_id, depth, position + 1)
            keys = keys.repeat_interleave(groups, dim=1).float()
            values = values.repeat_interleave(groups, dim=1).float()
            scores = torch.einsum("hd,thd->ht", q[row].float(), keys) * self.head_dim**-0.5
            output[row] = torch.einsum("ht,thd->hd", scores.softmax(-1), values).to(q.dtype)
        return output

    def read(self, layer, request_id, depth, length=None):
        self._validate_layer(layer)
        self._validate_depth(depth)
        allocation = self._get_allocation(request_id)
        if length is None:
            length = self._readable[id(allocation)][depth][layer].prefix
        if type(length) is not int or not 0 <= length <= allocation.max_tokens:
            raise ValueError("read length exceeds the reserved token budget")
        self._require_prefix(allocation, layer, depth, length)
        exits = self._exits[id(allocation)]
        blocks = self._stage(
            [
                allocation.block_tables[min(depth, exits.get(p, depth))][p // self.block_size]
                for p in range(length)
            ],
            torch.long,
        )
        offsets = torch.arange(length, device=self.device) % self.block_size
        self.materialization_bytes += (
            2 * length * self.num_kv_heads * self.head_dim * self.key_cache.element_size()
        )
        return self.key_cache[blocks, layer, offsets], self.value_cache[blocks, layer, offsets]
