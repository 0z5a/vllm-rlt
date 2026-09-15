import math
from dataclasses import dataclass


def _positive(name, value):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class CacheConfig:
    # None profiles CUDA memory; CPU retains a bounded 256-block default.
    num_blocks: int | None = None
    block_size: int = 16
    layout: str = "last_exited"
    gpu_memory_utilization: float = 0.9
    kv_cache_memory_bytes: int | None = None
    memory_reserve_bytes: int = 256 * 1024 * 1024

    def __post_init__(self):
        if self.num_blocks is not None:
            _positive("num_blocks", self.num_blocks)
        _positive("block_size", self.block_size)
        if self.layout not in ("last_exited", "shared"):
            raise ValueError("layout must be 'last_exited' or 'shared'")
        if (
            not math.isfinite(self.gpu_memory_utilization)
            or not 0 < self.gpu_memory_utilization <= 1
        ):
            raise ValueError("gpu_memory_utilization must be in (0, 1]")
        if self.kv_cache_memory_bytes is not None:
            _positive("kv_cache_memory_bytes", self.kv_cache_memory_bytes)
            if self.num_blocks is not None:
                raise ValueError("specify num_blocks or kv_cache_memory_bytes, not both")
        if type(self.memory_reserve_bytes) is not int or self.memory_reserve_bytes < 0:
            raise ValueError("memory_reserve_bytes must be a nonnegative integer")


@dataclass(frozen=True)
class SchedulerConfig:
    max_num_seqs: int = 8
    max_num_batched_tokens: int = 128
    mode: str = "refill"
    min_coda_batch_size: int = 1
    admission_scan_limit: int = 64
    max_admission_bypasses: int = 8
    prefill_chunk_size: int = 128
    max_prefill_batches_before_decode: int = 1

    def __post_init__(self):
        for name in (
            "max_num_seqs",
            "max_num_batched_tokens",
            "min_coda_batch_size",
            "admission_scan_limit",
            "max_admission_bypasses",
            "prefill_chunk_size",
            "max_prefill_batches_before_decode",
        ):
            _positive(name, getattr(self, name))
        if self.mode not in ("refill", "no_refill"):
            raise ValueError("mode must be 'refill' or 'no_refill'")


@dataclass(frozen=True)
class ExitConfig:
    # random_lookahead predicts exit AFTER one additional loop.
    # ouro_delayed delays the original cumulative-hazard decision by one loop.
    mode: str = "ouro"
    seed: int = 0
    depths_by_request: dict[str, list[int]] | None = None

    def __post_init__(self):
        if self.mode not in ("ouro", "ouro_delayed", "random_lookahead", "trace"):
            raise ValueError("exit mode must be ouro, ouro_delayed, random_lookahead or trace")
        if self.mode == "trace" and not isinstance(self.depths_by_request, dict):
            raise ValueError("trace mode requires depths_by_request")
        if self.mode != "trace" and self.depths_by_request is not None:
            raise ValueError("depths_by_request is only valid in trace mode")
        if type(self.seed) is not int or not 0 <= self.seed < 2**63:
            raise ValueError("exit seed must be an integer in [0, 2**63)")


@dataclass(frozen=True)
class ExecutionConfig:
    async_scheduling: bool = False
    multi_stream: bool = True
    static_buffers: bool = False
    pad_to_power_of_two: bool = False

    def __post_init__(self):
        for name in ("async_scheduling", "multi_stream", "static_buffers", "pad_to_power_of_two"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a boolean")
        if self.pad_to_power_of_two and not self.static_buffers:
            raise ValueError("padding requires static_buffers")
