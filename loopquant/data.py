"""Document ownership and token-window manifests."""

import hashlib
from dataclasses import dataclass


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
