import math
from dataclasses import dataclass


@dataclass(frozen=True)
class PDConfig:
    prefill_devices: tuple[int, ...] = (0,)
    decode_devices: tuple[int, ...] = (1,)
    max_pending_requests: int = 256
    transfer_chunk_bytes: int = 64 * 1024**2
    max_inflight_bytes: int = 256 * 1024**2
    max_transfer_descriptors: int = 256
    max_control_messages: int = 32
    startup_timeout: float = 300.0
    request_timeout: float = 300.0
    shutdown_timeout: float = 30.0
    backend: str = "UCX"
    max_receiving_requests: int = 32
    max_draining_requests: int = 8
    pair_ranks: tuple[tuple[int, int, int], ...] = ()

    def __post_init__(self):
        devices = (*self.prefill_devices, *self.decode_devices)
        if not self.prefill_devices or not self.decode_devices:
            raise ValueError("PD requires nonempty prefill and decode device pools")
        if any(type(d) is not int or d < 0 for d in devices) or len(set(devices)) != len(devices):
            raise ValueError("PD devices must be distinct nonnegative CUDA device indices")
        for name, value in (
            ("max_receiving_requests", self.max_receiving_requests),
            ("max_draining_requests", self.max_draining_requests),
        ):
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        for name, value in (
            ("max_pending_requests", self.max_pending_requests),
            ("transfer_chunk_bytes", self.transfer_chunk_bytes),
            ("max_inflight_bytes", self.max_inflight_bytes),
            ("max_transfer_descriptors", self.max_transfer_descriptors),
            ("max_control_messages", self.max_control_messages),
        ):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.transfer_chunk_bytes > self.max_inflight_bytes:
            raise ValueError("transfer_chunk_bytes exceeds max_inflight_bytes")
        for name, value in (
            ("startup_timeout", self.startup_timeout),
            ("request_timeout", self.request_timeout),
            ("shutdown_timeout", self.shutdown_timeout),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if not self.backend:
            raise ValueError("NIXL backend must be specified")
        pairs = set()
        for prefill, decode, rank in self.pair_ranks:
            if (
                prefill not in self.prefill_devices
                or decode not in self.decode_devices
                or type(rank) is not int
                or rank < 0
                or (prefill, decode) in pairs
            ):
                raise ValueError("pair_ranks must contain unique configured P/D pairs and ranks")
            pairs.add((prefill, decode))
