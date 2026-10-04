"""Logical layer groups and their physical recurrence multiplicity."""

from dataclasses import dataclass


@dataclass(frozen=True)
class KVGroupSpec:
    name: str
    layers: tuple[int, ...]
    repeats: int

    @property
    def planes(self) -> int:
        return len(self.layers) * self.repeats
