"""Stage selection and fairness state for loop-level scheduling."""

from enum import Enum, auto
from typing import TYPE_CHECKING

from vllm_rlt.config import SchedulerConfig, SpeculativeConfig
from vllm_rlt.request import Stage

if TYPE_CHECKING:
    from vllm_rlt.core.scheduler import Scheduler, SchedulerOutput


class NoRefillPhase(Enum):
    FILL = auto()
    CORE = auto()
    CODA = auto()


class SchedulingPolicy:
    # Every stage that advances prompt prefill counts against the decode bound,
    # whether the whole chunk or one recurrence depth forms the batch boundary.
    PREFILL_STAGES = (
        Stage.PREFILL,
        Stage.PREFILLING_PRELUDE,
        Stage.PREFILLING_RECURRENT,
    )

    def __init__(self, config: SchedulerConfig):
        self.config = config
        self.prefill_batches_since_recurrent = 0

    @property
    def decode_due(self) -> bool:
        return self.prefill_batches_since_recurrent >= self.config.max_prefill_batches_before_decode

    def record_batch(self, stage: Stage) -> None:
        """Count nonempty scheduled batches, not attempts or GPU completions."""
        if stage in self.PREFILL_STAGES:
            self.prefill_batches_since_recurrent += 1
        elif stage in (Stage.RECURRENT, Stage.SPECULATIVE):
            self.prefill_batches_since_recurrent = 0


class RefillPolicy(SchedulingPolicy):
    def schedule(
        self, scheduler: "Scheduler", *, prefer_recurrent: bool = False
    ) -> "SchedulerOutput | None":
        q = scheduler.queues
        # Give independent core work an overlap opportunity before refilling.
        if prefer_recurrent and q[Stage.RECURRENT]:
            return scheduler._take(Stage.RECURRENT)
        if q[Stage.PRELUDE]:
            return scheduler._take(Stage.PRELUDE)
        if q[Stage.CODA] and (
            len(q[Stage.CODA]) >= self.config.min_coda_batch_size or not q[Stage.RECURRENT]
        ):
            return scheduler._take(Stage.CODA)
        if self.decode_due and q[Stage.RECURRENT]:
            return scheduler._take(Stage.RECURRENT)
        scheduler._admit()
        if q[Stage.PREFILL]:
            return scheduler._take(Stage.PREFILL)
        # Admission can restore a snapshot directly to a boundary stage.
        if q[Stage.PRELUDE]:
            return scheduler._take(Stage.PRELUDE)
        if q[Stage.CODA] and (
            len(q[Stage.CODA]) >= self.config.min_coda_batch_size or not q[Stage.RECURRENT]
        ):
            return scheduler._take(Stage.CODA)
        if q[Stage.RECURRENT]:
            return scheduler._take(Stage.RECURRENT)
        return None


class NoRefillPolicy(SchedulingPolicy):
    def __init__(self, config: SchedulerConfig):
        super().__init__(config)
        self.phase = NoRefillPhase.FILL

    def schedule(
        self, scheduler: "Scheduler", *, prefer_recurrent: bool = False
    ) -> "SchedulerOutput | None":
        q = scheduler.queues
        if self.phase == NoRefillPhase.CORE:
            if q[Stage.RECURRENT]:
                return scheduler._take(Stage.RECURRENT)
            self.phase = NoRefillPhase.CODA
        if self.phase == NoRefillPhase.CODA:
            if q[Stage.CODA]:
                return scheduler._take(Stage.CODA)
            self.phase = NoRefillPhase.FILL
        # Bound prefill runs so continuous arrivals cannot starve decode.
        if self.decode_due:
            if q[Stage.PRELUDE]:
                return scheduler._take(Stage.PRELUDE)
            if q[Stage.RECURRENT]:
                self.phase = NoRefillPhase.CORE
                return scheduler._take(Stage.RECURRENT)
        scheduler._admit()
        if q[Stage.PREFILL]:
            return scheduler._take(Stage.PREFILL)
        if q[Stage.CODA]:  # First output after full-depth prompt prefill.
            return scheduler._take(Stage.CODA)
        if q[Stage.PRELUDE]:
            return scheduler._take(Stage.PRELUDE)
        if q[Stage.RECURRENT]:
            self.phase = NoRefillPhase.CORE
            return scheduler._take(Stage.RECURRENT)
        return None


class SpeculativePolicy(SchedulingPolicy):
    """Schedule atomic draft/verify rounds with bounded prefill interleaving."""

    def schedule(self, scheduler, *, prefer_recurrent=False):
        q = scheduler.queues
        if q[Stage.CODA]:
            return scheduler._take(Stage.CODA)
        if self.decode_due and q[Stage.SPECULATIVE]:
            return scheduler._take(Stage.SPECULATIVE)
        scheduler._admit()
        if q[Stage.PREFILL]:
            return scheduler._take(Stage.PREFILL)
        if q[Stage.SPECULATIVE]:
            return scheduler._take(Stage.SPECULATIVE)
        return None


class SplitPrefillPolicy(SchedulingPolicy):
    """Refill ordering where each recurrence depth is its own batch boundary."""

    def schedule(self, scheduler: "Scheduler", *, prefer_recurrent: bool = False):
        q = scheduler.queues
        if prefer_recurrent and q[Stage.RECURRENT]:
            return scheduler._take(Stage.RECURRENT)
        if q[Stage.PRELUDE]:
            return scheduler._take(Stage.PRELUDE)
        if q[Stage.CODA] and (
            len(q[Stage.CODA]) >= self.config.min_coda_batch_size or not q[Stage.RECURRENT]
        ):
            return scheduler._take(Stage.CODA)
        if self.decode_due and q[Stage.RECURRENT]:
            return scheduler._take(Stage.RECURRENT)
        scheduler._admit()
        for stage in (Stage.PREFILLING_PRELUDE, Stage.PREFILLING_RECURRENT, Stage.RECURRENT):
            if q[stage]:
                # A stage can hold only blocked requests: the staged-chunk window
                # may still be busy, or KV growth may be waiting for capacity.
                batch = scheduler._take(stage)
                if batch is not None:
                    return batch
        return None


class SpeculativeSplitPrefillPolicy(SchedulingPolicy):
    """Placeholder for split prefill with speculative decoding.

    Speculation needs CODA/SPECULATIVE rounds ordered against staged prompt
    depths, including one bounded output per request, so it stays unreachable
    from the factory until that ordering exists.
    """

    def schedule(self, scheduler: "Scheduler", *, prefer_recurrent: bool = False):
        raise NotImplementedError(
            "split prefill does not support speculative decoding yet; "
            "use ExecutionConfig(atomic_prefill=True)"
        )


class SchedulingPolicyFactory:
    @staticmethod
    def create_from_configs(
        scheduler_config: SchedulerConfig,
        speculative_config: SpeculativeConfig | None,
        atomic_prefill: bool = True,
    ) -> SchedulingPolicy:
        """Pick one policy; reject combinations no policy implements yet.

        Failing here keeps an unimplemented mode from silently changing
        semantics or returning an empty schedule forever.
        """
        if speculative_config is not None:
            if not atomic_prefill:
                raise ValueError(
                    "speculative decoding requires atomic prefill; "
                    "set ExecutionConfig(atomic_prefill=True)"
                )
            return SpeculativePolicy(scheduler_config)
        if atomic_prefill:
            policy_cls = RefillPolicy if scheduler_config.mode == "refill" else NoRefillPolicy
            return policy_cls(scheduler_config)
        if scheduler_config.mode != "refill":
            raise ValueError("split prefill currently implements refill scheduling only")
        return SplitPrefillPolicy(scheduler_config)
