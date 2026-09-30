"""Inference profiling and per-rank background artifact processing."""

import csv
import hashlib
import json
import logging
import os
import queue
import re
import shutil
import socket
import tarfile
import threading
import time
import uuid
from collections import Counter
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


def profile_timestamp(output_dir):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    sequence = 1
    while True:
        name = stamp if sequence == 1 else f"{stamp}-{sequence}"
        try:
            # Reserve the shared manifest atomically, including across processes.
            with (root / f"manifest-{name}.json").open("x") as stream:
                json.dump({"capture_time": name, "ranks": {}}, stream)
            return name
        except FileExistsError:
            sequence += 1


@dataclass(frozen=True)
class ProfileConfig:
    enabled: bool = False
    output_dir: str | None = None
    activities: tuple[str, ...] | None = None
    record_shapes: bool = False
    with_stack: bool = False
    profile_memory: bool = False
    with_flops: bool = False
    wait: int = 0
    warmup: int = 1
    active: int = 10
    repeat: int = 1

    def __post_init__(self):
        for name in ("enabled", "record_shapes", "with_stack", "profile_memory", "with_flops"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be boolean")
        for name in ("wait", "warmup", "active", "repeat"):
            value = getattr(self, name)
            if type(value) is not int or value < (1 if name == "active" else 0):
                raise ValueError(f"invalid profile {name}")
        if self.enabled and (not isinstance(self.output_dir, str) or not self.output_dir):
            raise ValueError("profiling requires output_dir")
        if self.activities is not None:
            values = tuple(self.activities)
            if (
                not values
                or len(set(values)) != len(values)
                or any(a not in ("cpu", "cuda") for a in values)
            ):
                raise ValueError("profile activities must be unique cpu/cuda values")
            object.__setattr__(self, "activities", values)

    def resolve_activities(self, device):
        names = self.activities or (("cpu", "cuda") if str(device).startswith("cuda") else ("cpu",))
        values = [getattr(torch.profiler.ProfilerActivity, name.upper()) for name in names]
        if not set(values) <= torch.profiler.supported_activities():
            raise ValueError(f"unsupported profile activities: {names}")
        return values


def _json(path, data):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _digest(path):
    result = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


class _JSONStream:
    """Decode one JSON value at a time, keeping only the current event in memory."""

    def __init__(self, source):
        self.source, self.buffer = source, ""
        self.decoder = json.JSONDecoder()
        self.eof = False

    def fill(self):
        chunk = self.source.read(65536)
        self.buffer += chunk
        self.eof = not chunk

    def peek(self):
        self.buffer = self.buffer.lstrip()
        while not self.buffer and not self.eof:
            self.fill()
            self.buffer = self.buffer.lstrip()
        return self.buffer[:1]

    def take(self, character):
        if self.peek() != character:
            raise ValueError(f"malformed trace: expected {character!r}")
        self.buffer = self.buffer[1:]

    def value(self):
        self.peek()
        while True:
            try:
                value, end = self.decoder.raw_decode(self.buffer)
                # A numeric value may extend into the next input chunk.
                if end == len(self.buffer) and not self.eof:
                    self.fill()
                    continue
                self.buffer = self.buffer[end:]
                return value
            except json.JSONDecodeError:
                if self.eof:
                    raise
                self.fill()


def _trace_events(path):
    with open(path) as source:
        stream = _JSONStream(source)
        stream.take("{")
        found = False
        while stream.peek() != "}":
            key = stream.value()
            stream.take(":")
            if key == "traceEvents":
                found = True
                stream.take("[")
                while stream.peek() != "]":
                    event = stream.value()
                    if not isinstance(event, dict):
                        raise ValueError("trace event must be an object")
                    yield event
                    if stream.peek() == "]":
                        break
                    stream.take(",")
                stream.take("]")
            else:
                stream.value()
            if stream.peek() == "}":
                break
            stream.take(",")
        stream.take("}")
        if not found or stream.peek():
            raise ValueError("missing traceEvents or trailing trace data")


def parse_trace(path, output, metadata):
    """Aggregate exported events, without claiming summed durations are wall time."""
    groups = {}
    counts = Counter()
    low, high = None, None
    memory = {"events": 0, "allocated_bytes": 0, "freed_bytes": 0}
    for event in _trace_events(path):
        category = str(event.get("cat", ""))
        counts[category] += 1
        args = event.get("args") or {}
        if event.get("name") == "[memory]":
            memory["events"] += 1
            size = args.get("Bytes", 0)
            if isinstance(size, (int, float)):
                memory["allocated_bytes" if size >= 0 else "freed_bytes"] += abs(size)
        if event.get("ph") != "X":
            continue
        start, duration = event.get("ts"), event.get("dur")
        if not isinstance(start, (int, float)) or not isinstance(duration, (int, float)):
            continue
        low = start if low is None else min(low, start)
        high = start + duration if high is None else max(high, start + duration)
        if category not in (
            "cpu_op",
            "kernel",
            "gpu_memcpy",
            "gpu_memset",
            "cuda_runtime",
            "cuda_driver",
            "user_annotation",
            "python_function",
        ):
            continue
        shape = args.get("Input Dims")
        stack = args.get("Call stack")
        attribution = "device" if category in ("kernel", "gpu_memcpy", "gpu_memset") else "cpu"
        key = (
            event.get("name", ""),
            category,
            str(event.get("pid", "")),
            str(event.get("tid", "")),
            str(args.get("device", "")),
            str(args.get("stream", "")),
            json.dumps(shape),
            json.dumps(stack),
        )
        if key not in groups:
            groups[key] = dict(
                zip(
                    ("name", "category", "pid", "tid", "device", "stream", "input_shapes", "stack"),
                    key,
                )
            )
            groups[key].update(
                attribution=attribution,
                count=0,
                inclusive_duration_us=0,
                self_duration_us=None,
                flops=None,
            )
        row = groups[key]
        row["count"] += 1
        row["inclusive_duration_us"] += duration
        flops = args.get("flops", args.get("FLOPs"))
        if isinstance(flops, (int, float)):
            row["flops"] = (row["flops"] or 0) + flops
    fields = (
        "name",
        "category",
        "attribution",
        "pid",
        "tid",
        "device",
        "stream",
        "count",
        "inclusive_duration_us",
        "self_duration_us",
        "input_shapes",
        "stack",
        "flops",
    )
    with open(output / "operators.csv", "w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        writer.writerows(groups.values())
    summary = dict(
        metadata,
        event_counts=dict(counts),
        observed_duration_us=None if low is None else high - low,
        memory=memory if metadata["config"]["profile_memory"] else None,
        notes=[
            "Durations are inclusive event times; their sum is not wall time.",
            "Self time and unavailable native event fields are null.",
            "Memory events describe visible allocations, not total device memory.",
        ],
    )
    if "cuda" in (metadata["config"].get("activities") or ()) and not any(
        counts[c] for c in ("kernel", "gpu_memcpy", "gpu_memset")
    ):
        summary["notes"].append("No CUDA device events recorded; device coverage is unverified.")
    _json(output / "summary.json", summary)


def _parse_native_details(staging):
    """Aggregate immutable native event snapshots on the artifact thread."""
    metrics = {}
    path = staging / "events.jsonl"
    if path.exists():
        with path.open() as source:
            for line in source:
                event = json.loads(line)
                key = (
                    event["name"],
                    event["device_type"],
                    event["thread"],
                    json.dumps(event["input_shapes"]),
                    json.dumps(event["stack"]),
                )
                if key not in metrics:
                    metrics[key] = dict(
                        zip(("name", "device_type", "thread", "input_shapes", "stack"), key)
                    )
                    metrics[key].update(
                        count=0,
                        cpu_time_us=0,
                        self_cpu_time_us=0,
                        device_time_us=0,
                        self_device_time_us=0,
                        flops=None,
                    )
                row = metrics[key]
                row["count"] += 1
                for field in (
                    "cpu_time_us",
                    "self_cpu_time_us",
                    "device_time_us",
                    "self_device_time_us",
                    "flops",
                ):
                    if event[field] is not None:
                        row[field] = (row[field] or 0) + event[field]
        fields = (
            "name",
            "device_type",
            "thread",
            "input_shapes",
            "stack",
            "count",
            "cpu_time_us",
            "self_cpu_time_us",
            "device_time_us",
            "self_device_time_us",
            "flops",
        )
        with (staging / "operator_metrics.csv").open("w", newline="") as destination:
            writer = csv.DictWriter(destination, fieldnames=fields)
            writer.writeheader()
            writer.writerows(metrics.values())
    memory = staging / "memory.json"
    if memory.exists():
        # Native memory timeline is [timestamps, category-size rows]. Read each
        # row separately; traces can contain millions of allocation changes.
        with memory.open() as source:
            stream = _JSONStream(source)
            stream.take("[")
            stream.take("[")
            samples = 0
            first, last = None, None
            while stream.peek() != "]":
                last = stream.value()
                first = last if first is None else first
                samples += 1
                if stream.peek() == "]":
                    break
                stream.take(",")
            stream.take("]")
            stream.take(",")
            stream.take("[")
            rows, peak = 0, 0
            while stream.peek() != "]":
                values = stream.value()
                peak = max(peak, sum(values))
                rows += 1
                if stream.peek() == "]":
                    break
                stream.take(",")
            stream.take("]")
            stream.take("]")
            if rows != samples or stream.peek():
                raise ValueError("invalid memory timeline")
        _json(
            staging / "memory_summary.json",
            dict(
                samples=samples,
                first_timestamp=first,
                last_timestamp=last,
                peak_visible_category_bytes=peak,
                note="Native timeline categories, not total device memory.",
            ),
        )
    stacks = staging / "stacks.txt"
    if stacks.exists():
        entries, weight = 0, 0.0
        with stacks.open() as source:
            for line in source:
                if line.strip():
                    _, value = line.rsplit(" ", 1)
                    weight += float(value)
                    entries += 1
        _json(
            staging / "stack_summary.json",
            dict(
                entries=entries,
                self_cpu_time_us=weight,
                note="An empty export may reflect native stack-collection limitations.",
            ),
        )


def _package(staging, archive):
    metadata = json.loads((staging / "metadata.json").read_text())
    parse_trace(staging / "trace.json", staging, metadata)
    _parse_native_details(staging)
    inventory = {
        p.name: {"bytes": p.stat().st_size, "sha256": _digest(p)}
        for p in staging.iterdir()
        if p.is_file() and p.name != "inventory.json"
    }
    _json(staging / "inventory.json", inventory)
    temporary = archive.with_suffix(archive.suffix + ".tmp")
    with tarfile.open(temporary, "w:gz") as bundle:
        for name in (*inventory, "inventory.json"):
            bundle.add(staging / name, arcname=name, recursive=False)
    with tarfile.open(temporary, "r:gz") as bundle:
        for name, item in inventory.items():
            result = hashlib.sha256()
            size = 0
            with bundle.extractfile(name) as member:
                for chunk in iter(lambda: member.read(1024 * 1024), b""):
                    result.update(chunk)
                    size += len(chunk)
            if size != item["bytes"] or result.hexdigest() != item["sha256"]:
                raise ValueError(f"archive verification failed: {name}")
    temporary.replace(archive)
    checksum = _digest(archive)
    cleanup_error = None
    try:
        shutil.rmtree(staging)
    except OSError as exc:
        cleanup_error = str(exc)
    return checksum, cleanup_error


class ProfileSession:
    def __init__(
        self,
        config,
        device,
        *,
        scheduled=False,
        session_id=None,
        capture_time=None,
        rank=None,
        role="engine",
        managed_manifest=True,
    ):
        self.config = config
        activities = config.resolve_activities(device)
        self.session_id = session_id or uuid.uuid4().hex
        if not re.fullmatch(r"[A-Za-z0-9_-]+", self.session_id):
            raise ValueError("invalid profile session ID")
        if rank is None:
            rank = (
                torch.distributed.get_rank()
                if (torch.distributed.is_available() and torch.distributed.is_initialized())
                else int(os.environ.get("RANK", 0))
            )
        if type(rank) is not int or rank < 0:
            raise ValueError("profile rank must be a nonnegative integer")
        self.rank = rank
        self.capture_time = capture_time or profile_timestamp(config.output_dir)
        if not re.fullmatch(r"\d{8}T\d{6}Z(?:-[1-9]\d*)?", self.capture_time):
            raise ValueError("invalid profile capture timestamp")
        self.root = Path(config.output_dir).expanduser().resolve()
        self.directory = self.root / f"rank-{rank:05d}-{self.capture_time}"
        self.directory.mkdir(parents=True, exist_ok=False)
        self.metadata = dict(
            session_id=self.session_id,
            capture_time=self.capture_time,
            rank=rank,
            role=role,
            hostname=socket.gethostname(),
            pid=os.getpid(),
            device=str(device),
            config=asdict(config),
            scheduled=scheduled,
            torch_version=torch.__version__,
            activities=[activity.name.lower() for activity in activities],
        )
        self.managed_manifest = managed_manifest
        self.jobs = {}
        self.errors = []
        self.lock = threading.Lock()
        self.queue = queue.Queue(maxsize=16)
        self.done = threading.Event()
        self.steps = 0
        self.cycles = 0
        self.recording = False
        self.owner = threading.get_ident()
        self.scheduled = scheduled
        self.profiler = torch.profiler.profile(
            activities=activities,
            schedule=torch.profiler.schedule(
                wait=config.wait, warmup=config.warmup, active=config.active, repeat=config.repeat
            )
            if scheduled
            else None,
            on_trace_ready=self._export,
            record_shapes=config.record_shapes,
            with_stack=config.with_stack,
            profile_memory=config.profile_memory,
            with_flops=config.with_flops,
            # Kineto needs verbose export to retain operator call stacks.
            experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True)
            if config.with_stack
            else None,
        )
        self.thread = threading.Thread(
            target=self._process, name=f"profile-rank-{rank}", daemon=False
        )
        self.profiler.start()
        self.recording = True
        self.metadata["started_at_ns"] = time.time_ns()
        self.thread.start()
        self._write_status_safely()

    def _check_owner(self):
        if threading.get_ident() != self.owner:
            raise RuntimeError("profiler control must run on the engine owner thread")

    def step(self):
        self._check_owner()
        if self.recording:
            self.steps += 1
            self.profiler.step()
            length = self.config.wait + self.config.warmup + self.config.active
            if self.scheduled and self.config.repeat and self.steps >= length * self.config.repeat:
                self.stop()

    def _export(self, profiler):
        cycle = self.cycles
        self.cycles += 1
        staging = self.directory / f".cycle-{cycle:05d}.raw"
        archive = self.directory / f"cycle-{cycle:05d}.tar.gz"
        try:
            staging.mkdir()
            profiler.export_chrome_trace(str(staging / "trace.json"))
            wait = getattr(profiler, "wait_for_exports", None)
            if wait is not None:
                wait()
            # Snapshot native events while this cycle still owns them. Native
            # post-processing is part of export; custom aggregation runs below
            # on the background thread, without a reference to the profiler.
            with (staging / "events.jsonl").open("w") as destination:
                for event in profiler.events():
                    snapshot = dict(
                        name=event.name,
                        device_type=str(event.device_type),
                        thread=event.thread,
                        input_shapes=event.input_shapes if self.config.record_shapes else None,
                        stack=event.stack if self.config.with_stack else None,
                        cpu_time_us=event.cpu_time_total,
                        self_cpu_time_us=event.self_cpu_time_total,
                        device_time_us=event.device_time_total,
                        self_device_time_us=event.self_device_time_total,
                        flops=event.flops if self.config.with_flops else None,
                    )
                    destination.write(json.dumps(snapshot) + "\n")
            if self.config.with_stack:
                profiler.export_stacks(str(staging / "stacks.txt"))
            if self.config.profile_memory and self.config.record_shapes and self.config.with_stack:
                profiler.export_memory_timeline(
                    str(staging / "memory.json"), device=self.metadata["device"]
                )
            expected = self.config.wait + self.config.warmup + self.config.active
            metadata = dict(
                self.metadata,
                cycle=cycle,
                steps=self.steps,
                incomplete_window=self.scheduled and self.steps < expected * (cycle + 1),
            )
            _json(staging / "metadata.json", metadata)
            with self.lock:
                self.jobs[cycle] = dict(
                    state="queued",
                    staging=str(staging),
                    archive=str(archive),
                    incomplete_window=metadata["incomplete_window"],
                )
            try:
                self.queue.put_nowait(cycle)
            except queue.Full:
                pass  # Worker scans queued descriptors after consuming its bounded queue.
        except Exception as exc:
            with self.lock:
                self.jobs[cycle] = dict(state="failed", staging=str(staging), error=str(exc))
            logger.exception("profile export failed")

    def _process(self):
        while True:
            try:
                cycle = self.queue.get(timeout=0.1)
            except queue.Empty:
                with self.lock:
                    cycle = next((i for i, j in self.jobs.items() if j["state"] == "queued"), None)
                if cycle is None:
                    if self.done.is_set():
                        break
                    continue
            with self.lock:
                job = self.jobs[cycle]
                if job["state"] != "queued":
                    continue
                job["state"] = "processing"
                staging, archive = Path(job["staging"]), Path(job["archive"])
            try:
                checksum, cleanup_error = _package(staging, archive)
                update = dict(state="ready", sha256=checksum, cleanup_error=cleanup_error)
            except Exception as exc:
                update = dict(state="failed", error=str(exc))
                logger.exception("profile processing failed: %s", staging)
            with self.lock:
                job.update(update)
            self._write_status_safely()
        self._write_status_safely()

    def status(self):
        with self.lock:
            return dict(
                self.metadata,
                recording=self.recording,
                steps=self.steps,
                jobs={str(k): dict(v) for k, v in self.jobs.items()},
                errors=list(self.errors),
                artifacts_complete=self.done.is_set()
                and all(j["state"] in ("ready", "failed") for j in self.jobs.values()),
                empty=self.done.is_set() and not self.jobs,
                success=self.done.is_set()
                and bool(self.jobs)
                and not self.errors
                and all(
                    j["state"] == "ready" and not j.get("cleanup_error") for j in self.jobs.values()
                ),
            )

    def _write_status(self):
        # Serialize writers so their atomic temporary files cannot collide.
        with self.lock:
            status = dict(
                self.metadata,
                recording=self.recording,
                jobs={str(k): dict(v) for k, v in self.jobs.items()},
                errors=list(self.errors),
            )
            _json(self.directory / "status.json", status)
            if self.managed_manifest:
                world = int(os.environ.get("WORLD_SIZE", 1))
                statuses = {}
                for rank in range(world):
                    path = self.root / f"rank-{rank:05d}-{self.capture_time}" / "status.json"
                    statuses[str(rank)] = (
                        json.loads(path.read_text()) if path.exists() else {"state": "missing"}
                    )
                # Only rank zero owns the common manifest; all ranks retain status.json.
                if self.rank == 0:
                    _json(
                        self.root / f"manifest-{self.capture_time}.json",
                        dict(
                            session_id=self.session_id,
                            capture_time=self.capture_time,
                            ranks=statuses,
                        ),
                    )

    def _write_status_safely(self):
        try:
            self._write_status()
        except Exception as exc:
            logger.exception("cannot write profile status")
            with self.lock:
                self.errors.append(str(exc))

    def stop(self):
        self._check_owner()
        if self.recording:
            try:
                self.profiler.stop()
            except Exception as exc:
                self.errors.append(str(exc))
                logger.exception("profiler stop failed")
            finally:
                # Export has snapshotted everything needed by the artifact worker.
                # Drop native events (including shapes/stacks) on the owner thread.
                self.profiler = None
                self.recording = False
                self.metadata["stopped_at_ns"] = time.time_ns()
                self.done.set()
                self._write_status_safely()
        return self.status()

    def wait(self, timeout=None):
        if self.recording:
            raise RuntimeError("stop profiling before waiting for artifacts")
        self.thread.join(timeout)
        if self.thread.is_alive():
            raise TimeoutError("profile artifact processing is still running")
        return self.status()


class ProfileController:
    """Owner-thread control retaining only the latest capture's status."""

    def __init__(self, device):
        self.device = device
        self.current = None

    @property
    def recording(self):
        return self.current is not None and self.current.recording

    def start(self, config, *, scheduled=False, **identity):
        if isinstance(config, dict):
            config = ProfileConfig(**config)
        if self.recording:
            raise RuntimeError("a profiling session is already active")
        if not config.enabled:
            return {"recording": False, "disabled": True}
        if self.current is not None and not self.current.status()["artifacts_complete"]:
            raise RuntimeError("previous profile artifacts are still processing")
        if self.current is not None:
            self.current.wait()  # Finish the last status write before releasing the session.
        self.current = ProfileSession(config, self.device, scheduled=scheduled, **identity)
        return self.status()

    def stop(self):
        return self.current.stop() if self.current else self.status()

    def status(self):
        return (
            self.current.status()
            if self.current
            else {
                "state": "not_started",
                "recording": False,
                "jobs": {},
                "artifacts_complete": True,
                "success": False,
            }
        )

    def wait(self, timeout=None):
        if self.current is not None:
            return self.current.wait(timeout)
        return self.status()

    def step(self):
        if self.recording:
            self.current.step()

    def region(self, name):
        return torch.profiler.record_function(name) if self.recording else nullcontext()

    def close(self):
        self.stop()
        return self.wait()
