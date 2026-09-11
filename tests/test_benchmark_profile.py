"""Profiler lifecycle tests use synthetic traces and never start a device profiler."""

import json
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from vllm_lt import CacheConfig, SamplingParams
from vllm_lt.benchmarks.profile import Capture, finite_checks, trace_summary
from vllm_lt.engine.llm_engine import LLMEngine
from vllm_lt.models import OuroConfig, OuroForCausalLM


class FakeProfiler:
    def __init__(self, *, export_error=False):
        self.export_error = export_error
        self.enters = self.exits = self.exports = 0

    def __enter__(self):
        self.enters += 1
        return self

    def __exit__(self, *exc):
        self.exits += 1

    def export_chrome_trace(self, path):
        self.exports += 1
        if self.export_error:
            raise RuntimeError("injected export failure")
        with open(path, "w") as stream:
            json.dump(
                {
                    "traceEvents": [
                        {"ph": "X", "cat": "kernel", "name": "synthetic kernel", "dur": 5},
                        {"ph": "X", "cat": "gpu_memcpy", "name": "synthetic copy", "dur": 2},
                    ]
                },
                stream,
            )

    def key_averages(self):
        return [
            SimpleNamespace(key=name, cpu_time_total=10, self_cpu_time_total=3, count=2)
            for name in ("vllm_lt::runner", "aten::matmul")
        ]


@pytest.fixture
def fake_profiler(monkeypatch):
    profiler, labels = FakeProfiler(), []

    def no_cuda(*args, **kwargs):
        pytest.fail("CPU profiler test attempted a CUDA query/initialization")

    for name in ("is_available", "device_count", "current_device", "_lazy_init"):
        monkeypatch.setattr(torch.cuda, name, no_cuda)
    monkeypatch.setattr(torch.profiler, "profile", lambda **kwargs: profiler)

    @contextmanager
    def record(label):
        labels.append(label)
        yield

    monkeypatch.setattr(torch.profiler, "record_function", record)
    return profiler, labels


def engine():
    model = OuroForCausalLM(OuroConfig.tiny())
    result = LLMEngine(model, cache_config=CacheConfig(64, 2))
    for index in range(4):
        result.add_request(str(index), [index + 1], SamplingParams(max_tokens=4, ignore_eos=True))
    return result


def run_spec():
    return {
        "run_id": "profile-W1-refill-1",
        "cell_id": "W1-refill",
        "workload_id": "W1",
        "mode": "refill",
        "controls_sha256": "c" * 64,
        "workload_sha256": "d" * 64,
    }


def bindings(instance):
    targets = [
        (instance.scheduler, "schedule"),
        (instance.model_runner, "execute"),
        (instance, "_update"),
        (instance.model, "prelude"),
        (instance.model, "recurrent"),
        (instance.model, "coda"),
        (instance.model_runner, "_sample"),
        (instance.cache_manager, "write"),
        (instance.cache_manager, "finalize_token"),
    ]
    if hasattr(instance.cache_manager, "attend"):
        targets.append((instance.cache_manager, "attend"))
    return [(obj, name, name in vars(obj), vars(obj).get(name)) for obj, name in targets]


def assert_restored(saved):
    for obj, name, existed, value in saved:
        assert (name in vars(obj)) == existed, f"instance override leaked: {name}"
        if existed:
            assert vars(obj)[name] is value


def test_trace_summary_counts_only_complete_device_events(tmp_path):
    path = tmp_path / "trace.json"
    path.write_text(
        json.dumps(
            {
                "traceEvents": [
                    {"ph": "X", "cat": "kernel", "dur": 4},
                    {"ph": "X", "cat": "kernel", "dur": 6},
                    {"ph": "X", "cat": "gpu_memcpy", "dur": 3},
                    {"ph": "X", "cat": "cpu_op", "name": "cudaMemcpyAsync", "dur": 100},
                    {"ph": "i", "cat": "kernel", "dur": 1000},
                    {"ph": "X", "cat": "gpu_memset", "dur": 30},
                ]
            }
        )
    )
    assert trace_summary(path) == {
        "kernel_count": 2,
        "memcpy_count": 1,
        "kernel_total_us": 10,
        "memcpy_total_us": 3,
    }


def test_capture_counts_subsequent_outputs_once_and_preserves_batch_crossing(
    tmp_path, fake_profiler, monkeypatch
):
    profiler, labels = fake_profiler
    instance = engine()
    original_coda = instance.model.coda

    def local_coda(hidden):
        return original_coda(hidden)

    monkeypatch.setattr(instance.model, "coda", local_coda)
    saved = bindings(instance)
    capture = Capture(instance, tmp_path / "capture", limit=3, run=run_spec())
    with capture:
        for step in range(100):
            if not instance.has_unfinished_requests():
                break
            outputs = instance.step()
            capture.after_step(outputs)
        else:
            pytest.fail("tiny workload did not complete")
    assert capture.done
    assert capture.metadata["start_step"] == 2  # prefill and first coda precede capture
    assert capture.metadata["end_step"] == 7  # prelude, four traversals, then coda
    assert capture.metadata["subsequent_outputs"] == 4  # whole coda batch crosses limit three
    assert labels.count("vllm_lt::runner") == 6
    assert profiler.enters == profiler.exits == profiler.exports == 1
    assert [row["name"] for row in capture.metadata["scope_summary"]] == ["vllm_lt::runner"]
    assert instance.cache_manager.num_used_blocks == 0
    assert capture.engine is None and not capture.saved
    assert_restored(saved)


def test_capture_restores_wrappers_when_trace_export_fails(tmp_path, fake_profiler):
    fake_profiler[0].export_error = True
    instance = engine()
    saved = bindings(instance)
    capture = Capture(instance, tmp_path / "capture", limit=1, run=run_spec())
    with pytest.raises(RuntimeError, match="export failure"), capture:
        for _ in range(100):
            capture.after_step(instance.step())
    assert_restored(saved)
    assert fake_profiler[0].exits == 1
    assert not capture.done
    for request_id in list(instance.scheduler.requests):
        instance.abort_request(request_id)
    assert instance.cache_manager.num_used_blocks == 0


def test_capture_restores_partial_installation(tmp_path, fake_profiler):
    instance = engine()
    cache = instance.cache_manager
    instance.cache_manager = SimpleNamespace(write=cache.write, finalize_token=cache.finalize_token)
    saved = bindings(instance)
    with (
        pytest.raises(AttributeError),
        Capture(instance, tmp_path / "capture", limit=1, run=run_spec()),
    ):
        pytest.fail("missing interface should fail during instrumentation installation")
    assert_restored(saved)


def test_finite_hooks_check_each_result_and_restore_after_failure():
    class Model:
        def recurrent(self):
            return torch.ones(1), torch.tensor([float("nan")])

        def coda(self):
            return torch.ones(2)

    model = Model()
    with (
        pytest.raises(ValueError, match="nonfinite recurrent"),
        finite_checks(model, True) as counts,
    ):
        model.coda()
        model.recurrent()
    assert counts == {"recurrent": 0, "coda": 1}
    assert "recurrent" not in vars(model) and "coda" not in vars(model)


def test_disabled_finite_checks_leave_results_and_methods_untouched():
    class Unreadable:
        def isfinite(self):
            pytest.fail("disabled finite checks must not read any tensor")

    value = Unreadable()
    model = SimpleNamespace(recurrent=lambda: (value, value), coda=lambda: value)
    original = dict(vars(model))
    with finite_checks(model, False) as counts:
        assert model.recurrent() == (value, value)
        assert model.coda() is value
    assert vars(model) == original
    assert counts == {"recurrent": 0, "coda": 0}
