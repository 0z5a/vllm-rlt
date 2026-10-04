"""Payload accounting; estimates are not observed device memory."""

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class Geometry:
    layers: int
    kv_heads: int
    head_dim: int
    loops: int
    element_bytes: int = 2
    page_tokens: int = 16

    def __post_init__(self):
        if any(type(v) is not int or v <= 0 for v in asdict(self).values()):
            raise ValueError("geometry dimensions must be positive integers")

    @property
    def record_bytes(self) -> int:
        return 2 * self.layers * self.kv_heads * self.head_dim * self.element_bytes


def account(shape: Geometry, prompt: int, exits: list[int]) -> dict[str, int]:
    """Count written positions, including full-depth prompt and actual loop counts."""
    if type(prompt) is not int or prompt < 0:
        raise ValueError("prompt must be a nonnegative integer")
    if any(type(r) is not int or not 1 <= r <= shape.loops for r in exits):
        raise ValueError("executed loops must be in [1, max_loops]")
    positions = prompt + len(exits)
    records = shape.loops * prompt + sum(exits)
    dense = shape.record_bytes * shape.loops * positions
    unique = shape.record_bytes * records
    pages = (positions + shape.page_tokens - 1) // shape.page_tokens
    physical = pages * shape.page_tokens * shape.loops * shape.record_bytes
    return {
        "cached_decode_positions": len(exits),
        "logical_kv_bytes": dense,
        "unique_live_payload_bytes": unique,
        "avoidable_promotion_payload_bytes": dense - unique,
        "rectangular_physical_bytes": physical,
        "rectangular_fragmentation_bytes": physical - dense,
        "compact_payload_lower_bound_bytes": unique,
        "compact_record_map_int32_bytes": len(exits) * shape.loops * 4,
        "compact_exit_int32_bytes": len(exits) * 4,
        "worst_future_record_bytes_per_position": shape.loops * shape.record_bytes,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    result = account(Geometry(**config["geometry"]), config["prompt_tokens"], config["loops_done"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps({"scope": "payload_estimate_not_device_measurement", **result}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
