"""CPU submission counters shared by diagnostic and timed engine captures."""

import math
from collections import Counter
from dataclasses import dataclass, field

from vllm_rlt.core.scheduler import SchedulerOutput
from vllm_rlt.request import Stage


@dataclass
class WorkCounters:
    recurrent_rows: Counter[int] = field(default_factory=Counter)
    recurrent_depth_rows: Counter[int] = field(default_factory=Counter)
    prefill_tokens: int = 0
    peak_residents: int = 0

    def observe(self, batch: SchedulerOutput | None, residents: int):
        self.peak_residents = max(self.peak_residents, residents)
        if batch is None:
            return
        if batch.stage == Stage.PREFILL:
            self.prefill_tokens += batch.num_tokens
        elif batch.stage == Stage.RECURRENT:
            self.recurrent_rows[batch.num_tokens] += 1
            for item in batch.items:
                self.recurrent_depth_rows[item.request.loops_done] += item.token_count

    def summary(self):
        calls = self.recurrent_rows.total()
        rows = sum(size * count for size, count in self.recurrent_rows.items())
        quantiles = {}
        for percentile in (10, 50, 90):
            rank, cumulative = math.ceil(calls * percentile / 100), 0
            value = None
            for size, count in sorted(self.recurrent_rows.items()):
                cumulative += count
                if cumulative >= rank:
                    value = size
                    break
            quantiles[f"p{percentile}"] = value
        return {
            "recurrent_batch_histogram": dict(sorted(self.recurrent_rows.items())),
            "submitted_recurrent_depth_rows": dict(sorted(self.recurrent_depth_rows.items())),
            "submitted_recurrent_rows": rows,
            "recurrent_calls": calls,
            "effective_batch_mean": rows / calls if calls else None,
            "effective_batch_nearest_rank_quantiles": quantiles,
            "submitted_prefill_tokens": self.prefill_tokens,
            "observed_peak_residents_after_step": self.peak_residents,
        }
