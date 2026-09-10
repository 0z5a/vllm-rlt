from dataclasses import dataclass


@dataclass(frozen=True)
class CacheConfig:
    # Total physical pages, including all loop depths. Each page holds all layers.
    num_blocks: int = 256
    block_size: int = 16

    def __post_init__(self):
        for name in ("num_blocks", "block_size"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class SchedulerConfig:
    max_num_seqs: int = 8
    max_num_batched_tokens: int = 128
    mode: str = "refill"
    min_coda_batch_size: int = 1

    def __post_init__(self):
        for name in ("max_num_seqs", "max_num_batched_tokens", "min_coda_batch_size"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.mode not in ("refill", "no_refill"):
            raise ValueError("mode must be 'refill' or 'no_refill'")
