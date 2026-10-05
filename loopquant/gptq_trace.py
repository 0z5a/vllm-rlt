"""Native first/all/matched-row Hessians with explicit recurrent row ownership."""

import hashlib
import io

import torch
from torch import nn

from vllm_rlt.core.kv_cache_manager import _PreparedKVBatch
from vllm_rlt.layers import RMSNorm

from .gptq import HessianStats
from .trace import ActivationTrace


def shared_input_projection(name: str) -> str:
    """Ouro's QKV and gate/up consumers see the same norm output."""
    parent, field = name.rsplit(".", 1)
    return (
        parent
        + "."
        + {"k_proj": "q_proj", "v_proj": "q_proj", "up_proj": "gate_proj"}.get(field, field)
    )


class HessianTrace(ActivationTrace):
    """Collect three registered baselines without storing activation histories.

    The matched policy chooses exactly one loop for each request/position using
    a seeded hash. With complete fixed-depth execution its total row count equals
    first-loop calibration, while all-loop aggregation retains every valid row.
    QKV and gate/up share a Gram matrix but each physical weight is packed once.
    """

    def __init__(self, prompt_lengths: dict[str, int], loops: int, seed: int = 17) -> None:
        super().__init__(io.StringIO(), prompt_lengths)
        if loops < 1:
            raise ValueError("calibration needs a positive fixed loop depth")
        self.loops, self.seed = loops, seed
        self.selection: dict[tuple[str, int], list[int]] = {}
        self.hessian_indices: dict[tuple[str, int], torch.Tensor] = {}
        self.hessians: dict[tuple[str, str], HessianStats] = {}
        self.aliases: dict[str, str] = {}

    def prepare(self, batch: _PreparedKVBatch, size: int) -> None:
        super().prepare(batch, size)
        self.selection = {}
        for index, row in enumerate(self.rows):
            if not row.valid_mask:
                continue
            if row.loop_id not in range(self.loops):
                raise ValueError("observed loop outside the registered calibration depth")
            self.selection.setdefault(("all", row.loop_id), []).append(index)
            if row.loop_id == 0:
                self.selection.setdefault(("first", 0), []).append(index)
            digest = hashlib.sha256(
                f"{self.seed}:{row.request_id}:{row.token_position}".encode()
            ).digest()
            if row.loop_id == int.from_bytes(digest[:8], "big") % self.loops:
                self.selection.setdefault(("matched", row.loop_id), []).append(index)
        self.hessian_indices = {}

    def record(
        self, name: str, site: str, module: nn.Linear | RMSNorm, values: torch.Tensor
    ) -> None:
        if not self.rows or site != "input" or not isinstance(module, nn.Linear):
            return
        canonical = shared_input_projection(name)
        self.aliases[name] = canonical
        if name != canonical:
            return
        if values.ndim != 2 or values.shape[0] != len(self.rows):
            raise ValueError("Hessian inputs must align with the prepared native rows")
        if not self.hessian_indices:
            self.hessian_indices = {
                key: torch.tensor(rows, device=values.device)
                for key, rows in self.selection.items()
            }
        for (policy, loop), indices in self.hessian_indices.items():
            key = (name, policy)
            if key not in self.hessians:
                self.hessians[key] = HessianStats.create(values.shape[-1], values.device)
            selected = values.index_select(0, indices)
            valid = torch.ones(selected.shape[0], device=values.device, dtype=torch.bool)
            self.hessians[key].add(selected, valid, loop)
        self.calls += 1

    def summary(self) -> list[dict]:
        result = []
        for name in sorted(set(self.aliases.values())):
            counts = {
                policy: sum(self.hessians[name, policy].loop_rows.values())
                for policy in ("first", "all", "matched")
            }
            if (
                counts["all"] != self.loops * counts["first"]
                or counts["matched"] != counts["first"]
            ):
                raise ValueError("calibration did not complete the declared fixed-depth row budget")
            for policy in ("first", "all", "matched"):
                stats = self.hessians[name, policy]
                result.append(
                    dict(
                        physical_projections=sorted(
                            alias for alias, target in self.aliases.items() if target == name
                        ),
                        producer=name,
                        policy=policy,
                        total_rows=counts[policy],
                        loop_rows=stats.loop_rows,
                        loop_energy=stats.loop_energy,
                        gram_bytes=stats.gram.numel() * stats.gram.element_size(),
                    )
                )
        return result
