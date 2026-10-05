"""Eager record-granular KV experiment with worst-future admission credits.

The arena uses the existing [page, layer, offset, head, dim] strides. Records
are individual offsets across all layers; unused offsets are globally reusable.
Prompt and decode positions share this arena without page-boundary rounding.
Future positions reserve every depth; optional credit reclamation applies only
to finalized historical positions whose skipped versions have no record.
"""

import torch

from vllm_rlt.core.alias_kv_cache import AliasKVCacheManager
from vllm_rlt.core.kv_cache_manager import _Allocation, _PreparedKVBatch, _WrittenPositions


class CompactKVCacheManager(AliasKVCacheManager):
    def __init__(self, *args, reclaim_skipped_credits=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.reclaim_skipped_credits = reclaim_skipped_credits
        self._free_records = list(reversed(range(self.num_blocks * self.block_size)))
        self._reserved_records = 0
        self._credits: dict[int, int] = {}
        self._records: dict[int, dict[tuple[int, int], int]] = {}
        self._maps: dict[int, torch.Tensor] = {}
        self.peak_live_records = 0
        self.peak_metadata_bytes = 0

    @property
    def num_free_blocks(self):
        """Uncommitted worst-future credits, not unused physical records."""
        return (self.num_blocks * self.block_size - self._reserved_records) // self.block_size

    @property
    def live_records(self):
        return self.num_blocks * self.block_size - len(self._free_records)

    def allocate(self, request_id, max_tokens, *, initial_tokens=None, prefix=()):
        if request_id in self._allocations:
            raise ValueError("request already has an allocation")
        if prefix or (initial_tokens is not None and initial_tokens != max_tokens):
            raise ValueError("compact storage requires a complete private reservation")
        required = self.required_blocks(max_tokens)
        if required > self.num_free_blocks:
            return False
        allocation = _Allocation(
            max_tokens,
            tuple(() for _ in range(self.max_loops)),
            [[_WrittenPositions() for _ in range(self.num_layers)] for _ in range(self.max_loops)],
        )
        key = id(allocation)
        self._allocations[request_id] = allocation
        self._readable[key] = [
            [_WrittenPositions() for _ in range(self.num_layers)] for _ in range(self.max_loops)
        ]
        self._exits[key], self._records[key] = {}, {}
        # Executed versions occupy R planes; the final plane holds exit depth.
        self._maps[key] = torch.full(
            (self.max_loops + 1, max_tokens), -1, dtype=torch.int32, device=self.device
        )
        self._credits[key] = required * self.block_size
        self._reserved_records += self._credits[key]
        self.peak_metadata_bytes = max(
            self.peak_metadata_bytes,
            sum(t.numel() * t.element_size() for t in self._maps.values())
            + self.source_depths.numel() * self.source_depths.element_size(),
        )
        return True

    def ensure_capacity(self, request_id, tokens):
        if not 0 < tokens <= self._get_allocation(request_id).max_tokens:
            raise ValueError("KV growth exceeds the reservation")
        return True

    def get_block_table(self, request_id, depth):
        raise ValueError("compact records cannot be exported as a rectangular block table")

    def free(self, request_id):
        allocation = self._allocations.pop(request_id, None)
        if allocation is None:
            return
        key = id(allocation)
        self._free_records.extend(self._records.pop(key).values())
        self._reserved_records -= self._credits.pop(key)
        del self._maps[key], self._readable[key], self._exits[key]

    def _prepare_batches(
        self, request_ids, depth_sets, positions, *, for_write=True, packed_prefill=False
    ):
        if packed_prefill:
            raise ValueError("compact packed prefill is not implemented")
        if not depth_sets:
            raise ValueError("at least one depth assignment is required")
        batches = []
        for depths in depth_sets:
            rows = tuple(self._validate_rows(request_ids, depths, positions))
            addresses = [(id(a), depth, position) for a, depth, position in rows]
            if for_write and len(set(addresses)) != len(rows):
                raise ValueError("duplicate compact write address")
            records = []
            for allocation, depth, position in rows:
                key = id(allocation)
                versions = self._records[key]
                if for_write and position in self._exits[key]:
                    # Coda may fill an unwritten boundary layer after recurrent
                    # finalization. Its depth-zero record already exists.
                    if (
                        len(self.recurrent_layers) == self.num_layers
                        or depth != 0
                        or (depth, position) not in versions
                    ):
                        raise ValueError("cannot overwrite a finalized source")
                if for_write and (depth, position) not in versions:
                    versions[depth, position] = self._free_records.pop()
                records.append(versions.get((depth, position), -1))
            n = len(rows)
            wide = self._stage(
                [self._maps[id(a)].data_ptr() for a, _, _ in rows]
                + [p for _, _, p in rows]
                + records
                + [record // self.block_size for record in records]
                + [record % self.block_size for record in records],
                torch.int64,
            )
            narrow = self._stage(
                [a.max_tokens for a, _, _ in rows]
                + [depth for _, depth, _ in rows]
                + [p + 1 for _, _, p in rows],
                torch.int32,
            )
            pointers, position_ids, record_ids = wide[:n], wide[n : 2 * n], wide[2 * n : 3 * n]
            widths, query_depths = narrow[:n], narrow[n : 2 * n]
            if for_write and rows:
                if self.device.type == "cuda":
                    from vllm_rlt.kernels.record_map import publish_records

                    publish_records(pointers, widths, query_depths, position_ids, record_ids)
                else:
                    for (allocation, depth, position), record in zip(rows, records):
                        self._maps[id(allocation)][depth, position] = record
            batches.append(
                _PreparedKVBatch(
                    owner=self,
                    rows=rows,
                    allocations=tuple(dict(zip(request_ids, (a for a, _, _ in rows))).items()),
                    position_ids=position_ids,
                    write_blocks=wide[3 * n : 4 * n],
                    write_offsets=wide[4 * n : 5 * n],
                    block_tables=torch.empty((len(rows), 0), dtype=torch.int32, device=self.device),
                    context_lengths=narrow[2 * n : 3 * n],
                    writable=for_write,
                    query_depths=query_depths,
                    record_map_pointers=pointers,
                    record_map_widths=widths,
                )
            )
        self.peak_live_records = max(self.peak_live_records, self.live_records)
        return tuple(batches)

    def _finalize_metadata(self, request_id, position, exit_depth):
        allocation = self._get_allocation(request_id)
        self._validate_depth(exit_depth)
        self._validate_position(allocation, position)
        key = id(allocation)
        if position in self._exits[key]:
            raise ValueError("position already finalized")
        for depth in range(exit_depth + 1):
            for layer in self.recurrent_layers:
                if position not in allocation.written[depth][layer]:
                    raise RuntimeError("cannot finalize an unwritten executed version")
        self._exits[key][position] = exit_depth
        if self.reclaim_skipped_credits:
            released = sum(
                (depth, position) not in self._records[key]
                for depth in range(exit_depth + 1, self.max_loops)
            )
            self._credits[key] -= released
            self._reserved_records -= released
        for depth in range(exit_depth + 1, self.max_loops):
            for layer in self.recurrent_layers:
                self._readable[key][depth][layer].add(position)
        return self._maps[key], self.max_loops * allocation.max_tokens + position

    def _attend_prepared(self, layer, batch, q):
        if self.backend != "triton":
            return super()._attend_prepared(layer, batch, q)
        self._validate_layer(layer)
        self._require_live_batch(batch)
        self._validate_tensor(q, len(batch.position_ids), "q", query=True)
        for allocation, depth, position in batch.rows:
            self._require_prefix(allocation, layer, depth, position + 1)
        from vllm_rlt.kernels.triton_attention import paged_attention

        return paged_attention(
            q,
            self.key_cache[:, layer],
            self.value_cache[:, layer],
            batch.block_tables,
            batch.context_lengths,
            query_depths=batch.query_depths,
            record_map_pointers=batch.record_map_pointers,
            record_map_widths=batch.record_map_widths,
            max_loops=self.max_loops,
        )

    def read(self, layer, request_id, depth, length=None):
        self._validate_layer(layer)
        self._validate_depth(depth)
        allocation = self._get_allocation(request_id)
        key = id(allocation)
        if length is None:
            length = self._readable[key][depth][layer].prefix
        if type(length) is not int or not 0 <= length <= allocation.max_tokens:
            raise ValueError("read length exceeds the reservation")
        self._require_prefix(allocation, layer, depth, length)
        records = self._stage(
            [
                self._records[key][min(depth, self._exits[key].get(p, depth)), p]
                for p in range(length)
            ],
            torch.int64,
        )
        self.materialization_bytes += (
            2 * length * self.num_kv_heads * self.head_dim * self.key_cache.element_size()
        )
        blocks, offsets = records // self.block_size, records % self.block_size
        return self.key_cache[blocks, layer, offsets], self.value_cache[blocks, layer, offsets]
