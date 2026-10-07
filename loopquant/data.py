"""Document ownership and token-window manifests."""

import hashlib
import random
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class Window:
    document_id: str
    source_revision: str
    split: str
    start: int
    token_ids: tuple[int, ...]
    content_sha256: str


def document_split(document_id: str, seed: int = 17) -> str:
    bucket = (
        int.from_bytes(hashlib.sha256(f"{seed}:{document_id}".encode()).digest()[:8], "big") % 100
    )
    if bucket < 70:
        return "calibration"
    if bucket < 85:
        return "dev"
    return "locked_test"


def validate_windows(windows: list[Window]) -> None:
    owners: dict[str, set[str]] = {}
    seen: set[tuple[str, str, int]] = set()
    for window in windows:
        if len(window.source_revision) != 40 or any(
            ch not in "0123456789abcdef" for ch in window.source_revision
        ):
            raise ValueError("each dataset must have a full immutable revision")
        if window.split not in {"calibration", "qat_train", "dev", "locked_test", "serving"}:
            raise ValueError("unknown data split")
        if window.start < 0 or len(window.token_ids) < 2:
            raise ValueError("invalid window position or token count")
        identity = (window.document_id, window.split, window.start)
        if identity in seen:
            raise ValueError("duplicate window")
        seen.add(identity)
        owners.setdefault(window.document_id, set()).add(window.split)
        if len(owners[window.document_id]) > 1 and not owners[window.document_id] <= {
            "calibration",
            "qat_train",
        }:
            raise ValueError(f"document crosses independent data splits: {window.document_id}")


def training_windows(windows: list[Window], length: int) -> list[Window]:
    """Full, disjoint microbatches from registered training/calibration documents.

    Short tails are omitted rather than joined across document boundaries. The
    caller reports both the retained pool and repeated tokens consumed by QAT.
    """
    validate_windows(windows)
    if length < 2 or any(row.split not in {"calibration", "qat_train"} for row in windows):
        raise ValueError("QAT requires training documents and at least two tokens per sequence")
    result = [
        replace(
            row,
            split="qat_train",
            start=row.start + offset,
            token_ids=row.token_ids[offset : offset + length],
        )
        for row in windows
        for offset in range(0, len(row.token_ids) - length + 1, length)
    ]
    if not result:
        raise ValueError("training pool has no full sequences")
    validate_windows(result)
    return result


def training_order(size: int, seed: int, position: int, count: int) -> list[int]:
    """Epoch shuffles indexed by consumed microbatches, including after resume."""
    if size < 1 or position < 0 or count < 1:
        raise ValueError("invalid training pool size or sampling interval")
    result = []
    while len(result) < count:
        epoch, offset = divmod(position + len(result), size)
        order = list(range(size))
        random.Random(f"{seed}:{epoch}").shuffle(order)
        result.extend(order[offset : offset + count - len(result)])
    return result
