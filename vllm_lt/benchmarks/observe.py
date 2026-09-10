"""Host-only benchmark accounting and instance-scoped observation.

No device queries or synchronization occur here. Resolve summaries, history
validation and gate checks after the caller's final timing boundary.
"""

import math
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from time import perf_counter_ns


def _integer(value, name, *, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def summarize_records(requests: list[dict], *, arrival_ns: int, synchronized_ns: int) -> dict:
    """Recompute metrics from final host records; all raw timestamps are absolute ns.

    This pure function raises ValueError for invalid histories. Undefined metrics
    are None and have a dotted-path explanation in ``unavailable``.
    """
    _integer(arrival_ns, "arrival_ns")
    _integer(synchronized_ns, "synchronized_ns", minimum=arrival_ns)
    per_request, unavailable = {}, {}
    all_times, token_count, decode_depths, tpot_numerator, gaps_count = [], 0, [], 0, 0

    def interval(row, start, end, label):
        first, last = row.get(start), row.get(end)
        if first is None or last is None:
            unavailable[label] = "required lifecycle timestamp was not observed"
            return None
        if last < first:
            raise ValueError(f"{label} has reversed lifecycle timestamps")
        return last - first

    for row in requests:
        request_id = row.get("request_id")
        if not isinstance(request_id, str) or not request_id or request_id in per_request:
            raise ValueError("request records require unique nonempty IDs")
        tokens, depths, timestamps = (
            row.get("token_ids"),
            row.get("exit_depths"),
            row.get("token_timestamps_ns"),
        )
        if not all(isinstance(value, list) for value in (tokens, depths, timestamps)):
            raise ValueError("token IDs, depths and timestamps must be arrays")
        if not len(tokens) == len(depths) == len(timestamps):
            raise ValueError("token, depth and timestamp histories must have equal lengths")
        for token in tokens:
            _integer(token, "token ID")
        for depth in depths:
            _integer(depth, "exit depth", minimum=1)
        for stamp in timestamps:
            _integer(stamp, "token timestamp", minimum=arrival_ns)
            if stamp > synchronized_ns:
                raise ValueError("token timestamp exceeds final synchronization")
        if timestamps != sorted(timestamps):
            raise ValueError("token timestamps must be monotonic")
        for name in ("admitted_ns", "enqueue_start_ns", "enqueue_end_ns", "first_prefill_ns"):
            stamp = row.get(name)
            if stamp is not None:
                _integer(stamp, name, minimum=arrival_ns)
                if stamp > synchronized_ns:
                    raise ValueError(f"{name} exceeds final synchronization")
        if type(row.get("finished")) is not bool:
            raise ValueError("finished must be boolean")
        base = f"per_request.{request_id}"
        metrics = {
            "ttft_ns": timestamps[0] - arrival_ns if timestamps else None,
            "completion_latency_ns": (
                timestamps[-1] - arrival_ns if timestamps and row["finished"] else None
            ),
            "tpot_ns": (timestamps[-1] - timestamps[0]) / (len(tokens) - 1)
            if len(tokens) > 1
            else None,
            "token_gaps_ns": [right - left for left, right in zip(timestamps, timestamps[1:])],
            "admission_queue_ns": row["admitted_ns"] - arrival_ns
            if row.get("admitted_ns") is not None
            else None,
            "enqueue_ns": interval(row, "enqueue_start_ns", "enqueue_end_ns", f"{base}.enqueue_ns"),
            "admitted_to_first_prefill_ns": interval(
                row, "admitted_ns", "first_prefill_ns", f"{base}.admitted_to_first_prefill_ns"
            ),
        }
        if not timestamps:
            unavailable[f"{base}.ttft_ns"] = "no output was observed"
        if metrics["completion_latency_ns"] is None:
            unavailable[f"{base}.completion_latency_ns"] = "no completed output history"
        if len(tokens) < 2:
            unavailable[f"{base}.tpot_ns"] = "fewer than two outputs"
        if metrics["admission_queue_ns"] is None:
            unavailable[f"{base}.admission_queue_ns"] = "admission was not observed"
        per_request[request_id] = metrics
        all_times.extend(timestamps)
        token_count += len(tokens)
        decode_depths.extend(depths[1:])
        if len(tokens) > 1:
            tpot_numerator += timestamps[-1] - timestamps[0]
            gaps_count += len(tokens) - 1
    delivery = max(all_times) - arrival_ns if all_times else None
    throughput = token_count * 1_000_000_000 / delivery if delivery else None
    if delivery is None:
        unavailable["delivery_wall_ns"] = "no output was observed"
    if throughput is None:
        unavailable["generated_tokens_per_second"] = "no positive delivery interval"
    if not decode_depths:
        unavailable["mean_decode_depth"] = "no subsequent outputs"
    if not gaps_count:
        unavailable["token_weighted_tpot_ns"] = "no subsequent outputs"
    return {
        "delivery_wall_ns": delivery,
        "synchronized_wall_ns": synchronized_ns - arrival_ns,
        "generated_tokens": token_count,
        "generated_tokens_per_second": throughput,
        "mean_decode_depth": sum(decode_depths) / len(decode_depths) if decode_depths else None,
        "token_weighted_tpot_ns": tpot_numerator / gaps_count if gaps_count else None,
        "per_request": per_request,
        "unavailable": unavailable,
    }


class RunCollector:
    def __init__(self, request_ids: list[str], *, arrival_ns: int, max_events: int | None = None):
        _integer(arrival_ns, "arrival_ns")
        if not request_ids or any(not isinstance(key, str) or not key for key in request_ids):
            raise ValueError("request IDs must be nonempty strings")
        if len(set(request_ids)) != len(request_ids):
            raise ValueError("duplicate request IDs")
        if max_events is not None:
            _integer(max_events, "max_events", minimum=1)
        self.arrival_ns = arrival_ns
        self.max_events = max_events
        self.events = []
        self.step_returns = {}
        self._last_return_ns = arrival_ns
        self._histories = {key: [] for key in request_ids}
        self._records = {
            key: {
                "request_id": key,
                "token_ids": [],
                "exit_depths": [],
                "token_timestamps_ns": [],
                "admitted_ns": None,
                "enqueue_start_ns": None,
                "enqueue_end_ns": None,
                "first_prefill_ns": None,
                "finished": False,
                "finish_reason": None,
            }
            for key in request_ids
        }

    def start(self, arrival_ns: int) -> None:
        """Set the timing origin after instrumentation setup, before any observation."""
        _integer(arrival_ns, "arrival_ns")
        observed = self.events or self.step_returns or any(self._histories.values())
        lifecycle_started = any(
            row[name] is not None
            for row in self._records.values()
            for name in ("admitted_ns", "enqueue_start_ns", "enqueue_end_ns", "first_prefill_ns")
        )
        if observed or lifecycle_started:
            raise ValueError("collector timing cannot restart after observation")
        self.arrival_ns = self._last_return_ns = arrival_ns

    def _record(self, request_id):
        try:
            return self._records[request_id]
        except KeyError:
            raise ValueError(f"unregistered request ID {request_id!r}") from None

    def _event(self, kind, stamp, **fields):
        _integer(stamp, "event timestamp", minimum=self.arrival_ns)
        if self.max_events is not None and len(self.events) >= self.max_events:
            raise ValueError("benchmark event storage limit exceeded")
        self.events.append(
            {
                "schema_version": 1,
                "artifact_type": "event",
                "event_seq": len(self.events),
                "kind": kind,
                "host_offset_ns": stamp - self.arrival_ns,
                **fields,
            }
        )

    def observe_submission(self, request_id, *, started_ns, ended_ns):
        row = self._record(request_id)
        if row["enqueue_start_ns"] is not None:
            raise ValueError("duplicate request submission")
        _integer(started_ns, "enqueue start", minimum=self.arrival_ns)
        _integer(ended_ns, "enqueue end", minimum=started_ns)
        row["enqueue_start_ns"], row["enqueue_end_ns"] = started_ns, ended_ns
        self._event(
            "submitted",
            ended_ns,
            request_id=request_id,
            logical_arrival_offset_ns=0,
            enqueue_start_offset_ns=started_ns - self.arrival_ns,
            enqueue_end_offset_ns=ended_ns - self.arrival_ns,
        )

    def observe_admission(self, request_id, *, admitted_ns, reserved_pages):
        row = self._record(request_id)
        if row["admitted_ns"] is not None:
            raise ValueError("duplicate request admission")
        row["admitted_ns"] = admitted_ns
        self._event("admitted", admitted_ns, request_id=request_id, reserved_pages=reserved_pages)

    def observe_prefill(self, request_id, *, dispatched_ns, step_id):
        row = self._record(request_id)
        if row["first_prefill_ns"] is None:
            row["first_prefill_ns"] = dispatched_ns
            self._event("first_prefill", dispatched_ns, request_id=request_id, step_id=step_id)

    def observe_outputs(self, outputs, *, step_id: int, returned_ns: int):
        _integer(step_id, "step_id")
        _integer(returned_ns, "returned_ns", minimum=self._last_return_ns)
        if step_id in self.step_returns:
            raise ValueError("duplicate observed step")
        self.step_returns[step_id] = returned_ns
        self._last_return_ns = returned_ns
        seen = set()
        for output in outputs:
            key = output.request_id
            if key in seen:
                raise ValueError("multiple outputs for one request in one step")
            seen.add(key)
            row = self._record(key)
            count = len(row["token_ids"])
            length = len(output.token_ids)
            if length != len(output.exit_depths) or length - count not in (0, 1):
                raise ValueError("incompatible cumulative output accounting")
            if row["finished"] and length != count:
                raise ValueError("new output after request completion")
            # RequestOutput owns copies of its histories. Retain those host arrays
            # for deferred prefix checking; never retain the live Request/tensors.
            self._histories[key].append((output.token_ids, output.exit_depths))
            if length > count:
                token, depth = output.token_ids[-1], output.exit_depths[-1]
                row["token_ids"].append(token)
                row["exit_depths"].append(depth)
                row["token_timestamps_ns"].append(returned_ns)
                self._event(
                    "token_emitted",
                    returned_ns,
                    request_id=key,
                    step_id=step_id,
                    output_index=count,
                    token_id=token,
                    exit_depth=depth,
                )
            if output.finished and not row["finished"]:
                self._event(
                    "request_finished",
                    returned_ns,
                    request_id=key,
                    step_id=step_id,
                    finish_reason=output.finish_reason,
                )
                row["finished"], row["finish_reason"] = True, output.finish_reason

    def finish(self, *, synchronized_ns: int) -> dict:
        _integer(synchronized_ns, "synchronized_ns", minimum=self._last_return_ns)
        for key, histories in self._histories.items():
            row = self._records[key]
            for tokens, depths in histories:
                size = len(tokens)
                if tokens != row["token_ids"][:size] or depths != row["exit_depths"][:size]:
                    raise ValueError(f"cumulative output prefix changed for {key!r}")
        result = self.snapshot(synchronized_ns=synchronized_ns)
        if result["metrics_error"] is not None:
            raise ValueError(result["metrics_error"])
        return result

    def snapshot(self, *, synchronized_ns: int | None) -> dict:
        """Preserve partial host evidence without validating cumulative prefixes.

        A failed or unavailable synchronization boundary cannot yield valid
        timing metrics. Raw request/event records are still returned unchanged.
        """
        requests = [
            {key: list(value) if isinstance(value, list) else value for key, value in row.items()}
            for row in self._records.values()
        ]
        metrics, error = None, None
        if synchronized_ns is None:
            error = "final synchronization boundary is unavailable"
        else:
            try:
                _integer(synchronized_ns, "synchronized_ns", minimum=self._last_return_ns)
                metrics = summarize_records(
                    requests, arrival_ns=self.arrival_ns, synchronized_ns=synchronized_ns
                )
            except ValueError as exc:
                error = str(exc)
        return {
            "requests": requests,
            "metrics": metrics,
            "metrics_error": error,
            "events": [dict(event) for event in self.events],
        }


def populated_page_stats(cache) -> dict:
    """Count host-tracked initialized pages, for profiling only.

    Layer completeness describes initialized-slot bookkeeping, not GPU stream
    completion. _WrittenPositions has canonical prefix/pending representation.
    """
    pages = set()
    all_layer_complete = True
    for allocation in cache._allocations.values():
        for table, layers in zip(allocation.block_tables, allocation.written):
            first = layers[0]
            all_layer_complete &= all(
                row.prefix == first.prefix and row.pending == first.pending for row in layers[1:]
            )
            for row in layers:
                logical_pages = set(range((row.prefix + cache.block_size - 1) // cache.block_size))
                logical_pages.update(position // cache.block_size for position in row.pending)
                pages.update(table[index] for index in logical_pages)
    return {"populated_pages": len(pages), "all_layer_complete": all_layer_complete}


@dataclass(frozen=True)
class RowSnapshot:
    request_id: str
    output_index: int
    input_position: int | None
    next_depth_index: int | None
    token_start: int
    token_count: int


@dataclass(frozen=True)
class BatchSnapshot:
    step_id: int
    stage: str
    rows: tuple[RowSnapshot, ...]


class EngineInstrumentation:
    def __init__(self, collector, *, max_steps=None):
        if max_steps is not None:
            _integer(max_steps, "max_steps", minimum=1)
        self.collector = collector
        self.max_steps = max_steps
        self.step_id = -1
        self.stage_counts = Counter()
        self.stage_tokens = Counter()
        self.recurrent_depth_counts = Counter()
        self.recurrent_occupancy = Counter()
        self.request_work = {}
        self.gate_probabilities = []
        self.logical_copy_bytes = 0
        self.snapshots = []

    def validate_gate_probabilities(self):
        if any(
            not math.isfinite(value) or not 0 <= value <= 1 for value in self.gate_probabilities
        ):
            raise ValueError("nonfinite or invalid actual gate probability")
        return {"count": len(self.gate_probabilities), "finite": True}

    def summary(self):
        """Call only after timing; snapshots contain host values, never Requests."""
        snapshots = []
        for snapshot in self.snapshots:
            row = dict(snapshot)
            row["batch"] = asdict(row["batch"])
            row["step_return_ns"] = self.collector.step_returns.get(row["batch"]["step_id"])
            snapshots.append(row)
        nonfinite = {
            str(index): repr(value)
            for index, value in enumerate(self.gate_probabilities)
            if not math.isfinite(value)
        }
        return {
            "stage_counts": dict(self.stage_counts),
            "stage_tokens": dict(self.stage_tokens),
            "recurrent_depth_counts": dict(self.recurrent_depth_counts),
            "recurrent_occupancy": dict(self.recurrent_occupancy),
            "request_work": {key: dict(value) for key, value in self.request_work.items()},
            "gate_probabilities": [
                value if math.isfinite(value) else None for value in self.gate_probabilities
            ],
            "nonfinite_gate_probabilities": nonfinite,
            "logical_copy_bytes": self.logical_copy_bytes,
            "snapshots": snapshots,
        }


@contextmanager
def instrument_engine(engine, collector, *, clock=perf_counter_ns, profile=False, max_steps=None):
    """Observe only this engine instance; restore even when execution raises.

    No CUDA events are inserted. The outer loop timestamps step return before
    calling collector.observe_outputs(..., step_id=observation.step_id, ...).
    """
    observed = EngineInstrumentation(collector, max_steps=max_steps)
    saved = []

    def replace(instance, name, function):
        saved.append((instance, name, name in vars(instance), vars(instance).get(name)))
        setattr(instance, name, function)

    original_schedule = engine.scheduler.schedule
    original_execute = engine.model_runner.execute
    original_allocate = engine.cache_manager.allocate
    original_finalize = engine.cache_manager.finalize_token

    def scan_pages():
        # Called only in profile mode; expose observer overhead as its own scope.
        import torch

        with torch.profiler.record_function("vllm_lt::observer_page_scan"):
            return populated_page_stats(engine.cache_manager)

    def allocate(request_id, max_tokens):
        allocated = original_allocate(request_id, max_tokens)
        stamp = clock()
        if allocated:
            collector.observe_admission(
                request_id,
                admitted_ns=stamp,
                reserved_pages=engine.cache_manager.required_blocks(max_tokens),
            )
        return allocated

    def schedule():
        started = clock() if profile else None
        batch = original_schedule()
        if batch is None:
            return batch
        ended = clock()
        observed.step_id += 1
        if max_steps is not None and observed.step_id >= max_steps:
            raise ValueError("benchmark scheduler-step limit exceeded")
        stage = batch.stage.value
        observed.stage_counts[stage] += 1
        observed.stage_tokens[stage] += batch.num_tokens
        if stage == "recurrent":
            observed.recurrent_occupancy[str(len(batch.items))] += 1
        rows = []
        for item in batch.items:
            request = item.request
            work = observed.request_work.setdefault(request.request_id, Counter())
            work[stage] += item.token_count
            if stage == "recurrent":
                observed.recurrent_depth_counts[str(request.loops_done + 1)] += 1
            elif stage == "prefill":
                work["prefill_token_traversals"] += (
                    item.token_count * engine.model.config.total_ut_steps
                )
                collector.observe_prefill(
                    request.request_id, dispatched_ns=ended, step_id=observed.step_id
                )
            if profile:
                rows.append(
                    RowSnapshot(
                        request.request_id,
                        len(request.generated_token_ids),
                        None if stage == "prefill" else request.position,
                        request.loops_done if stage == "recurrent" else None,
                        item.token_start,
                        item.token_count,
                    )
                )
        if profile:
            observed.snapshots.append(
                {
                    "batch": BatchSnapshot(observed.step_id, stage, tuple(rows)),
                    "schedule_start_ns": started,
                    "schedule_end_ns": ended,
                }
            )
        return batch

    def execute(batch):
        if profile:
            observed.snapshots[-1]["execute_start_ns"] = clock()
        result = original_execute(batch)
        if profile:
            observed.snapshots[-1]["execute_end_ns"] = clock()
            observed.snapshots[-1]["kv_after_execute"] = scan_pages()
        if batch.stage.value == "recurrent":
            if len(result) != len(batch.items):
                raise ValueError("gate result count differs from scheduled rows")
            observed.gate_probabilities.extend(result)
        return result

    def finalize(request_id, position, exit_depth):
        result = original_finalize(request_id, position, exit_depth)
        cache = engine.cache_manager
        observed.logical_copy_bytes += (cache.max_loops - exit_depth - 1) * (
            cache.bytes_per_block // cache.block_size
        )
        if profile:
            observed.snapshots[-1]["kv_after_finalize"] = scan_pages()
        return result

    try:
        replace(engine.cache_manager, "allocate", allocate)
        replace(engine.cache_manager, "finalize_token", finalize)
        replace(engine.scheduler, "schedule", schedule)
        replace(engine.model_runner, "execute", execute)
        yield observed
    finally:
        for instance, name, existed, value in reversed(saved):
            if existed:
                setattr(instance, name, value)
            else:
                delattr(instance, name)
