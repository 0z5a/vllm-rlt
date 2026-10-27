"""Sparse decode attention arithmetic and its per-row plan (RFC #156, section 3.2).

The sparse decode algorithm keeps the attention output of a reference loop and
replaces only the contribution of the retained keys in a later loop. The
reference loop is loop 2 in the RFC's one-based numbering, which is depth index
1 at runtime.

This module holds the arithmetic and the plan that describes one submission:

``reference_statistics``
    the dense reference attention of one row, plus the retained state
    ``(S, m, O_ref, C_ref)``,
``sparse_correction``
    the corrected output of a later loop, ``O = O_ref - C_ref + m * C_new``,
``SparseDecodePlan``
    the row modes and reference slots of one submission.

The arithmetic is deliberately plain PyTorch. The retained state is small, so a
straightforward implementation is easier to check against the worked example in
the RFC than an optimized kernel, and it keeps the contract that the attention
kernels must satisfy visible in one place.

The plan is the named view between the runner and the attention layer. The
runner owns the reference slots and their lifetime; the attention layer only
borrows the plan for the rows of one submission. Row modes and slots are
host-known: the runner decides both before it submits, so a submission never
depends on a device readback.
"""

from dataclasses import dataclass

import torch

# Row modes for one recurrent submission.
DENSE = 0
CAPTURE = 1
CONSUME = 2

# The reference loop is loop 2 in the RFC's one-based numbering, which is
# zero-based depth index 1 at runtime.
REFERENCE_DEPTH = 1


def reference_mode(depth: int) -> int:
    """Row mode of a recurrent step at the given zero-based loop depth."""
    if depth == REFERENCE_DEPTH:
        return CAPTURE
    if depth > REFERENCE_DEPTH:
        return CONSUME
    return DENSE


@dataclass(frozen=True)
class SparseDecodePlan:
    """The reference plan of one recurrent submission.

    ``modes`` and ``slots`` are host-known: the runner decides both before it
    submits, so a submission never depends on a device readback. ``slots`` holds
    one reference slot per row, or ``-1`` for a row that owns no reference.
    ``incarnations`` identifies the token that occupies each slot; a plan built
    for a superseded incarnation is stale and must be rejected.

    A dense row runs the ordinary attention path. A capture row is the
    reference loop: it computes the dense attention and publishes the retained
    state. A consume row is a later loop: it corrects the reference output with
    the retained keys.

    ``store`` is the runner-owned reference pool, borrowed by the attention
    layer through duck typing so that model code needs no import of the worker
    package.
    """

    modes: tuple[int, ...]
    slots: tuple[int, ...]
    incarnations: tuple[int, ...]
    key_retention: float
    key_bucket: int | None
    slot_ids: torch.Tensor | None = None
    store: object | None = None

    def groups(self) -> dict[int, list[int]]:
        """Rows per mode, in submission order."""
        grouped: dict[int, list[int]] = {}
        for row, mode in enumerate(self.modes):
            grouped.setdefault(mode, []).append(row)
        return {mode: rows for mode, rows in grouped.items() if rows}

    @property
    def reference_depth(self) -> int:
        """The zero-based depth that captures the reference state."""
        return REFERENCE_DEPTH

    def dense(self) -> bool:
        """Whether the submission has no sparse work at all."""
        return all(mode == DENSE for mode in self.modes)


def retained_count(length: int, key_retention: float, key_bucket: int | None) -> int:
    """How many keys a query with ``length`` legal keys retains.

    The retention fraction is rounded to the nearest key. ``key_bucket`` caps
    the retained count; ``None`` leaves the retention budget unbounded so that
    ``key_retention=1.0`` stays a true full-retention control.
    """
    if key_retention <= 0 or key_retention > 1:
        raise ValueError("key_retention must be in (0, 1]")
    if key_bucket is not None and key_bucket < 1:
        raise ValueError("key_bucket must be positive when set")
    if length < 1:
        return 0
    budget = max(1, int(length * key_retention + 0.5))
    if key_bucket is not None:
        budget = min(budget, key_bucket)
    return min(budget, length)


def expand_kv(tensor: torch.Tensor, groups: int) -> torch.Tensor:
    """Repeat KV heads for grouped-query attention."""
    if groups == 1:
        return tensor
    return tensor.repeat_interleave(groups, dim=1)


def select_keys(probabilities: torch.Tensor, count: int) -> torch.Tensor:
    """Pick the retained key positions of every head.

    ``probabilities`` is ``[heads, length]``. Equal probabilities are broken by
    the smaller absolute position, so the selection does not depend on the order
    of the rows in a batch.
    """
    if count < 1:
        raise ValueError("a selection needs at least one key")
    if count > probabilities.shape[1]:
        raise ValueError("cannot retain more keys than the query has")
    # A stable descending sort keeps equal probabilities in position order.
    order = torch.argsort(probabilities, dim=-1, descending=True, stable=True)
    return order[:, :count].sort(dim=-1).values


def reference_statistics(
    q: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    *,
    key_retention: float,
    key_bucket: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Dense reference attention and the retained state of one row.

    ``q`` is ``[heads, dim]``, ``keys`` and ``values`` are
    ``[length, kv_heads, dim]``. Returns ``(output, contribution, mass,
    positions)`` where output and contribution are ``[heads, dim]``, mass is
    ``[heads]`` and positions are the retained absolute key positions.
    """
    heads, dim = q.shape
    length = keys.shape[0]
    groups = heads // keys.shape[1]
    scale = 1.0 / dim**0.5
    expanded_keys = expand_kv(keys, groups).float()
    expanded_values = expand_kv(values, groups).float()
    scores = torch.einsum("hd,lhd->hl", q.float(), expanded_keys) * scale
    probabilities = torch.softmax(scores, dim=-1)
    output = torch.einsum("hl,lhd->hd", probabilities, expanded_values)
    count = retained_count(length, key_retention, key_bucket)
    positions = select_keys(probabilities, count)
    gathered = probabilities.gather(1, positions)
    # Index the per-head value rows: positions is [heads, count], so the head
    # index broadcasts against it and selects each head's own key rows.
    head_index = torch.arange(heads, device=q.device)[:, None]
    selected_values = expanded_values[positions, head_index]
    contribution = torch.einsum("hk,hkd->hd", gathered, selected_values)
    mass = gathered.sum(dim=-1)
    return output, contribution, mass, positions


def sparse_correction(
    q: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    mass: torch.Tensor,
    output: torch.Tensor,
    contribution: torch.Tensor,
) -> torch.Tensor:
    """Correct the reference output with the retained keys of a later loop.

    ``keys`` and ``values`` are ``[heads, retained, dim]`` and already carry the
    keys of the later loop. ``mass``, ``output`` and ``contribution`` are the
    retained reference state.
    """
    dim = q.shape[-1]
    scale = 1.0 / dim**0.5
    scores = torch.einsum("hd,hkd->hk", q.float(), keys.float()) * scale
    probabilities = torch.softmax(scores, dim=-1)
    new_contribution = torch.einsum("hk,hkd->hd", probabilities, values.float())
    return output - contribution + mass.float().unsqueeze(-1) * new_contribution
