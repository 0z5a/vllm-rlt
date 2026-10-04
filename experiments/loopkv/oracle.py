"""Independent dense-copy and canonical-record CPU address specifications.

This executable contract does not model CUDA ordering or prove engine speed.
"""

from dataclasses import dataclass, field


@dataclass
class DenseOracle:
    loops: int
    layers: int
    payload: dict[tuple[int, int, int], bytes] = field(default_factory=dict)

    def write(self, position: int, depth: int, layer: int, value: bytes) -> None:
        self.payload[position, depth, layer] = value

    def finalize(self, position: int, loops_done: int) -> None:
        # Deliberately performs physical copies; never calls the candidate resolver.
        for layer in range(self.layers):
            last = self.payload[position, loops_done - 1, layer]
            for depth in range(loops_done, self.loops):
                self.payload[position, depth, layer] = bytes(bytearray(last))


def resolve_source(query_depth: int, exit_depth: int, loops: int) -> int:
    if not 0 <= query_depth < loops or not 0 <= exit_depth < loops:
        raise ValueError("depth index must be in [0, max_loops)")
    return min(query_depth, exit_depth)


@dataclass
class AddressOracle:
    loops: int
    layers: int
    generation: int = 0
    records: dict[tuple[int, int, int], bytes] = field(default_factory=dict)
    exits: dict[int, int] = field(default_factory=dict)

    def write(self, position: int, depth: int, layer: int, value: bytes) -> None:
        if position < 0 or not 0 <= depth < self.loops or not 0 <= layer < self.layers:
            raise ValueError("invalid position, depth, or layer")
        if position in self.exits:
            raise ValueError("cannot overwrite a finalized position")
        self.records[position, depth, layer] = value

    def finalize(self, position: int, loops_done: int) -> None:
        if not 1 <= loops_done <= self.loops:
            raise ValueError("loops_done must be in [1, max_loops]")
        if position in self.exits:
            raise ValueError("position already finalized")
        for depth in range(loops_done):
            for layer in range(self.layers):
                if (position, depth, layer) not in self.records:
                    raise ValueError("cannot publish an unwritten version")
        self.exits[position] = loops_done - 1

    def read(
        self, position: int, depth: int, layer: int, generation: int, *, current: bool = False
    ) -> bytes:
        if generation != self.generation:
            raise ValueError("stale request generation")
        if not 0 <= depth < self.loops or not 0 <= layer < self.layers:
            raise ValueError("invalid depth or layer")
        if current:
            source = depth
        else:
            if position not in self.exits:
                raise ValueError("unfinalized history")
            source = resolve_source(depth, self.exits[position], self.loops)
        return self.records[position, source, layer]

    def release(self) -> None:
        self.records.clear()
        self.exits.clear()
        self.generation += 1
