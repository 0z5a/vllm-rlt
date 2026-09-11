"""Per-instance, bounded diagnostic instrumentation, excluded from timing runs."""

import json
from contextlib import contextmanager
from pathlib import Path

import torch

from vllm_lt.request import Stage


def trace_summary(path: Path) -> dict:
    events = json.loads(path.read_text()).get("traceEvents", [])
    kernels = [e for e in events if e.get("ph") == "X" and e.get("cat") == "kernel"]
    copies = [e for e in events if e.get("ph") == "X" and "memcpy" in e.get("cat", "").lower()]
    return {
        "kernel_count": len(kernels),
        "memcpy_count": len(copies),
        "kernel_total_us": sum(e.get("dur", 0) for e in kernels),
        "memcpy_total_us": sum(e.get("dur", 0) for e in copies),
    }


class Capture:
    """Capture natural execution from first prelude through a bounded output window."""

    def __init__(self, engine, output_dir: Path, *, limit: int, run: dict):
        self.engine = engine
        self.output_dir = output_dir
        self.limit = limit
        self.run = run
        self.active = False
        self.started = False
        self.done = False
        self.step_id = 0
        self.decode_outputs = 0
        self.start_step = None
        self.profiler = None
        self.saved = []
        self.metadata = None

    def _wrap(self, obj, name, label):
        # Preserve whether the original method was an instance override or class method.
        local = obj.__dict__.get(name)
        had_local = name in obj.__dict__
        original = getattr(obj, name)
        self.saved.append((obj, name, had_local, local))

        def call(*args, **kwargs):
            if name == "execute" and args[0].stage == Stage.PRELUDE and not self.started:
                self._start()
            if self.active:
                with torch.profiler.record_function(label):
                    if name == "execute":
                        with torch.profiler.record_function(
                            f"vllm_lt::stage::{args[0].stage.value}"
                        ):
                            return original(*args, **kwargs)
                    return original(*args, **kwargs)
            return original(*args, **kwargs)

        setattr(obj, name, call)

    def __enter__(self):
        wrappers = (
            (self.engine.scheduler, "schedule", "vllm_lt::schedule"),
            (self.engine.model_runner, "execute", "vllm_lt::runner"),
            (self.engine, "_update", "vllm_lt::update"),
            (self.engine.model, "prelude", "vllm_lt::prelude"),
            (self.engine.model, "recurrent", "vllm_lt::recurrent"),
            (self.engine.model, "coda", "vllm_lt::coda"),
            (self.engine.model_runner, "_sample", "vllm_lt::sampling"),
            (self.engine.cache_manager, "write", "vllm_lt::kv_write"),
            (self.engine.cache_manager, "attend", "vllm_lt::attention"),
            (self.engine.cache_manager, "finalize_token", "vllm_lt::kv_finalize"),
        )
        try:
            for obj, name, label in wrappers:
                self._wrap(obj, name, label)
        except Exception:
            self.__exit__(None, None, None)
            raise
        return self

    def _start(self):
        self.output_dir.mkdir(parents=True, exist_ok=False)
        self.profiler = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=False,
            with_stack=True,
        )
        self.profiler.__enter__()
        self.started = self.active = True
        self.start_step = self.step_id

    def after_step(self, outputs):
        if self.active:
            self.decode_outputs += sum(len(row.token_ids) > 1 for row in outputs)
            if self.decode_outputs >= self.limit:
                self._stop(complete=True)
        self.step_id += 1

    def _stop(self, *, complete):
        if not self.active:
            return
        self.active = False
        self.profiler.__exit__(None, None, None)
        trace = self.output_dir / "trace.json"
        self.profiler.export_chrome_trace(str(trace))
        gpu = trace_summary(trace)
        scopes = [
            {
                "name": item.key,
                "cpu_total_us": item.cpu_time_total,
                "cpu_self_us": item.self_cpu_time_total,
                "count": item.count,
            }
            for item in self.profiler.key_averages()
            if item.key.startswith("vllm_lt::")
        ]
        self.metadata = {
            "schema_version": 1,
            "artifact_type": "profile_metadata",
            "capture_id": self.run["run_id"],
            "cell_id": self.run["cell_id"],
            "workload_id": self.run["workload_id"],
            "mode": self.run["mode"],
            "start_step": self.start_step,
            "end_step": self.step_id,
            "subsequent_outputs": self.decode_outputs,
            "complete_window": complete,
            "gpu_events": gpu,
            "scope_summary": scopes,
            "torch_version": torch.__version__,
            "options": {"record_shapes": False, "with_stack": True},
            "controls_sha256": self.run["controls_sha256"],
            "workload_sha256": self.run["workload_sha256"],
            "coverage": "Decode-focused; naturally interleaved prefill is included and labeled.",
        }
        (self.output_dir / "metadata.json").write_text(
            json.dumps(self.metadata, indent=2, allow_nan=False) + "\n"
        )
        self.done = complete
        self.profiler = None

    def __exit__(self, *exc):
        try:
            self._stop(complete=False)
        finally:
            for obj, name, had_local, original in reversed(self.saved):
                if had_local:
                    setattr(obj, name, original)
                else:
                    delattr(obj, name)
            self.saved.clear()
            self.engine = None


@contextmanager
def finite_checks(model, enabled):
    """Feasibility-only device reads; never installed during measured executions."""
    counts = {"recurrent": 0, "coda": 0}
    if not enabled:
        yield counts
        return
    saved = []

    def install(name):
        had_local = name in model.__dict__
        local = model.__dict__.get(name)
        original = getattr(model, name)
        saved.append((name, had_local, local))

        def checked(*args, **kwargs):
            result = original(*args, **kwargs)
            values = result if isinstance(result, tuple) else (result,)
            if not all(bool(value.isfinite().all().item()) for value in values):
                raise ValueError(f"nonfinite {name} result during feasibility")
            counts[name] += 1
            return result

        setattr(model, name, checked)

    try:
        install("recurrent")
        install("coda")
        yield counts
    finally:
        for name, had_local, local in reversed(saved):
            if had_local:
                setattr(model, name, local)
            else:
                delattr(model, name)
