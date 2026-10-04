"""Match ASCII text stops using a separately qualified ByteLevel token table."""

from collections.abc import Sequence


class ByteStops:
    def __init__(self, pieces: Sequence[bytes], stops: Sequence[str]):
        self.pieces = pieces
        self.stops = tuple((s, s.encode("ascii")) for s in stops)
        if not self.stops or any(not value for _, value in self.stops):
            raise ValueError("text stops must be nonempty ASCII strings")
        self.overlap = max(len(value) for _, value in self.stops) - 1
        self.consumed = 0
        self.generated = b""

    def update(self, token_ids: Sequence[int]) -> tuple[str, int] | None:
        start = max(0, len(self.generated) - self.overlap)
        self.generated += b"".join(self.pieces[t] for t in token_ids[self.consumed :])
        self.consumed = len(token_ids)
        matches = [
            (offset, name)
            for name, value in self.stops
            if (offset := self.generated.find(value, start)) >= 0
        ]
        if not matches:
            return None
        offset, name = min(matches)
        return name, offset
