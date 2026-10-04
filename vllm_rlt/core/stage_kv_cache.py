"""Recurrent page bundles with one copy of each boundary-layer KV plane."""

from collections.abc import Sequence

import torch

from vllm_rlt.core.kv_cache_manager import KVCacheManager, _PreparedKVBatch
from vllm_rlt.core.kv_group import KVGroupSpec


class StageKVCacheManager(KVCacheManager):
    """Keep legacy depth-page accounting, but pack boundary planes once.

    A bundle owns R core page IDs ``bundle * R + depth`` and one boundary
    page. Admission, growth and release acquire/release complete bundles.
    Thus the scheduler's existing depth-page budget still counts the same
    logical capacity, without permitting boundary/core fragmentation.
    """

    def __init__(
        self,
        *,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        num_blocks: int,
        block_size: int,
        max_loops: int,
        recurrent_layers: Sequence[int],
        groups: tuple[KVGroupSpec, ...],
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
        backend: str = "torch",
        layout: str = "last_exited",
        enable_prefix_caching: bool = False,
        incremental_allocation: bool = False,
        watermark_ratio: float = 0.0,
    ):
        if layout != "last_exited" or enable_prefix_caching:
            raise ValueError("stage KV requires last_exited without cross-request prefix sharing")
        if max_loops < 1 or num_blocks < max_loops:
            raise ValueError("stage KV requires at least one complete recurrence page bundle")
        recurrent = tuple(recurrent_layers)
        if (
            not recurrent
            or len(set(recurrent)) != len(recurrent)
            or any(not 0 <= layer < num_layers for layer in recurrent)
        ):
            raise ValueError("recurrent layers must be nonempty, unique and inside the model")
        covered = tuple(layer for group in groups for layer in group.layers)
        if sorted(covered) != list(range(num_layers)) or any(
            group.repeats != (max_loops if layer in recurrent else 1)
            for group in groups
            for layer in group.layers
        ):
            raise ValueError(
                "KV groups must partition all layers with their execution multiplicity"
            )
        bundles = num_blocks // max_loops
        super().__init__(
            num_layers=len(recurrent),
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            num_blocks=bundles * max_loops,
            block_size=block_size,
            max_loops=max_loops,
            device=device,
            dtype=dtype,
            backend=backend,
            incremental_allocation=incremental_allocation,
            watermark_ratio=watermark_ratio,
        )
        self.groups = groups
        self.num_layers = num_layers
        self.recurrent_layers = recurrent
        self._core_layers = {layer: index for index, layer in enumerate(recurrent)}
        self._boundary_layers = {
            layer: index
            for index, layer in enumerate(
                layer for layer in range(num_layers) if layer not in recurrent
            )
        }
        shape = (bundles, len(self._boundary_layers), block_size, num_kv_heads, head_dim)
        self.boundary_keys = torch.empty(shape, device=self.device, dtype=dtype)
        self.boundary_values = torch.empty_like(self.boundary_keys)
        self._free_bundles = list(reversed(range(bundles)))
        self._free_blocks = []

    @property
    def num_free_blocks(self) -> int:
        return len(self._free_bundles) * self.max_loops

    @property
    def allocated_bytes(self) -> int:
        return (
            2
            * (self.key_cache.numel() + self.boundary_keys.numel())
            * self.key_cache.element_size()
        )

    @property
    def bytes_per_block(self) -> float:
        # Average per scheduler depth-page; only complete bundles are allocated.
        return self.allocated_bytes / self.num_blocks

    def _claim(self, count: int) -> list[int] | None:
        assert count % self.max_loops == 0
        bundles = count // self.max_loops
        if bundles > len(self._free_bundles):
            return None
        pages = [self._free_bundles.pop() for _ in range(bundles)]
        blocks = [
            page * self.max_loops + depth for depth in range(self.max_loops) for page in pages
        ]
        for block in blocks:
            assert self._refs[block] == 0
            self._refs[block] = 1
        return blocks

    def _drop_refs(self, blocks):
        bundles = set()
        for block in blocks:
            self._refs[block] -= 1
            assert self._refs[block] == 0
            bundles.add(block // self.max_loops)
        for bundle in sorted(bundles, reverse=True):
            begin = bundle * self.max_loops
            assert not any(self._refs[begin : begin + self.max_loops])
            self._free_bundles.append(bundle)

    def allocate(self, request_id: str, max_tokens: int, *, initial_tokens=None, prefix=()) -> bool:
        if prefix:
            raise ValueError("stage KV does not share request page bundles")
        return super().allocate(request_id, max_tokens, initial_tokens=initial_tokens)

    def layer_cache(self, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
        if layer in self._core_layers:
            index = self._core_layers[layer]
            return self.key_cache[:, index], self.value_cache[:, index]
        index = self._boundary_layers[layer]
        return self.boundary_keys[:, index], self.boundary_values[:, index]

    def layer_blocks(self, layer: int, blocks: torch.Tensor) -> torch.Tensor:
        return (
            blocks
            if layer in self._core_layers
            else blocks.div(self.max_loops, rounding_mode="floor")
        )

    def _require_boundary_depth(self, layer: int, batch: _PreparedKVBatch) -> None:
        if layer not in self._core_layers and any(depth != 0 for _, depth, _ in batch.rows):
            raise ValueError("boundary layers use depth zero")

    def _write_prepared(self, layer, batch, k, v):
        self._require_boundary_depth(layer, batch)
        super()._write_prepared(layer, batch, k, v)

    def _attend_prepared(self, layer, batch, q):
        self._require_boundary_depth(layer, batch)
        return super()._attend_prepared(layer, batch, q)

    def read(self, layer: int, request_id: str, depth: int, length: int | None = None):
        if layer not in self._core_layers and depth != 0:
            raise ValueError("boundary layers use depth zero")
        return super().read(layer, request_id, depth, length)

    def storage_pages(
        self, blocks: list[int]
    ) -> tuple[tuple[torch.Tensor, torch.Tensor, list[int]], ...]:
        boundaries = list(dict.fromkeys(block // self.max_loops for block in blocks))
        return (
            (self.key_cache, self.value_cache, blocks),
            (self.boundary_keys, self.boundary_values, boundaries),
        )
