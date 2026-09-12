"""CPU numerical/projection checks; stand-ins never qualify real graph replay."""

import math
import time
from copy import deepcopy
from pathlib import Path

import pytest
import torch

from vllm_lt.models import OuroConfig, OuroForCausalLM
from vllm_lt.validation import m3_capture as capture
from vllm_lt.validation import m3_persistent
from vllm_lt.validation.diagnostics import DiagnosticDump, SpoolBudget
from vllm_lt.validation.report import _expected_boundaries
from vllm_lt.validation.schema import read_json

ROOT = Path(__file__).resolve().parents[1]


def inputs():
    return (
        read_json(ROOT / "benchmarks/fixtures/ouro-q1.json"),
        read_json(ROOT / "benchmarks/fixtures/ouro-q1-contract.json"),
    )


@pytest.fixture(autouse=True)
def no_cuda(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("capture numerical CPU test attempted CUDA discovery/device work")

    for name in (
        "is_available",
        "device_count",
        "current_device",
        "_lazy_init",
        "init",
        "synchronize",
        "memory_allocated",
        "memory_reserved",
    ):
        monkeypatch.setattr(torch.cuda, name, forbidden)


def test_projected_plan_preserves_inputs_original_policy_and_full_kv():
    suite, contract = inputs()
    plan = capture.build_model_plan(suite, contract, OuroConfig().to_dict())
    capture.validate_model_plan(plan)
    previous = m3_persistent.build_model_plan(suite, contract, OuroConfig().to_dict())
    assert plan["contract"]["comparison_policy"] == contract["comparison_policy"]
    assert plan["suite"]["fixtures"] == previous["suite"]["fixtures"]
    assert len(plan["execution_order"]) == 15
    assert len(plan["comparison_order"]) == 31
    assert sum(c["phase"] == "feasibility" for c in plan["execution_order"]) == 2
    assert sum(c["family"] == "feasibility" for c in plan["comparison_order"]) == 1
    assert all(c["boundary_projection"] == capture.PROJECTION for c in plan["execution_order"])
    assert all(not c["require_exact"] for c in plan["comparison_order"])
    stats = plan["resource_estimates"]
    assert stats["comparison_records_upper_bound"] == 3879
    assert stats["retained_index_records_upper_bound"] == 1980
    assert stats["retained_tensor_bytes_upper_bound"] == 2536572960
    assert stats["max_retained_case_bytes"] == 788267520
    torch_cases = [c for c in plan["execution_order"] if c["backend"] == "torch"]
    assert len(torch_cases) == 2
    assert all(
        c["storage_strategy"] == "backend_fallback" and c["fixture_ids"] == ["Q1-L16-F0"]
        for c in torch_cases
    )
    assert all(c["history_mode"] == "teacher_forced" for c in torch_cases)


@pytest.mark.parametrize(
    "field",
    [
        "projection",
        "missing_case",
        "added_stream",
        "tolerance",
        "kv_bound",
        "graph_limit",
        "extra_field",
    ],
)
def test_rehashed_unfrozen_projection_matrix_or_threshold_is_rejected(field):
    plan = capture.build_model_plan(*inputs(), OuroConfig().to_dict())
    if field == "projection":
        plan["execution_order"][0]["boundary_projection"] = "last_layer_only"
    elif field == "missing_case":
        plan["execution_order"].pop()
    elif field == "added_stream":
        plan["comparison_order"].append(deepcopy(plan["comparison_order"][0]))
    elif field == "tolerance":
        plan["contract"]["comparison_policy"]["logits"]["float32"]["atol"] = 0.01
    elif field == "kv_bound":
        plan["resource_estimates"]["retained_tensor_bytes_upper_bound"] -= 4
    elif field == "graph_limit":
        plan["graph_limits"]["setup_timeout_s"] += 1
    else:
        plan["boundary_evidence"]["optional"] = True
    plan["numerical_plan_sha256"] = capture._digest(
        {k: v for k, v in plan.items() if k != "numerical_plan_sha256"}
    )
    with pytest.raises(ValueError, match="frozen projected"):
        capture.validate_model_plan(plan)


def test_shared_boundary_projection_leaves_old_coverage_unchanged_and_preserves_all_kv():
    plan = capture.build_model_plan(*inputs(), OuroConfig().to_dict())
    fixture = plan["suite"]["fixtures"][0]
    case = plan["execution_order"][1]
    traces = [
        {
            "output_index": i,
            "position": len(fixture["prompt_token_ids"]) - 1 + i,
            "exit_depth": 4,
            "history_sha256": "a" * 64,
        }
        for i in range(9)
    ]
    projected = _expected_boundaries(case, fixture, traces, plan["model_config"])
    legacy_case = {k: v for k, v in case.items() if k != "boundary_projection"}
    legacy = _expected_boundaries(legacy_case, fixture, traces, plan["model_config"])
    assert len(projected) == 93
    assert len(legacy) == 5277
    assert set(meta["operation"] for meta, _ in projected.values()) == set(capture.OPERATIONS)
    assert {k: v for k, v in legacy.items() if v[0]["operation"] in capture.OPERATIONS} == projected
    kv = [
        (meta, shape) for meta, shape in projected.values() if meta["operation"] == "populated_kv"
    ]
    for component in ("keys", "values"):
        assert [
            p for meta, _ in kv if meta["component"] == component for p in meta["positions"]
        ] == list(range(24))
    assert all(shape == [4, 24, 4, 16, 128] for _, shape in kv)
    with pytest.raises(ValueError, match="unknown numerical boundary projection"):
        _expected_boundaries(
            {**case, "boundary_projection": "unknown"}, fixture, traces, plan["model_config"]
        )


def test_actual_independent_oracle_emits_only_projected_records(tmp_path):
    from vllm_lt.validation.runner import execute_case

    config = OuroConfig.tiny(vocab_size=49152, max_position_embeddings=512)
    model = OuroForCausalLM(config)
    plan = capture.build_model_plan(*inputs(), config.to_dict())
    plan["plan_sha256"] = "a" * 64
    case = next(c for c in plan["execution_order"] if c["implementation"] == "oracle")
    budget = SpoolBudget()
    dumps = DiagnosticDump(tmp_path / "dumps", ["Q1-L64-F2", "Q1-L256-F0"], budget=budget)
    result = execute_case(model, plan, case, tmp_path, budget, dumps, time.monotonic() + 60)
    assert result["status"] == "complete" and result["observed_boundaries"] == 93
    index = read_json(tmp_path / "spools" / case["case_id"] / "Q1-L16-F0.index.json")
    operations = {r["metadata"]["operation"] for r in index["records"]}
    assert operations == set(capture.OPERATIONS)
    expected = _expected_boundaries(
        case, plan["suite"]["fixtures"][0], result["traces"]["Q1-L16-F0"], config.to_dict()
    )
    assert {r["key"] for r in index["records"]} == set(expected)
    assert (
        sum(r["size_bytes"] for r in index["records"])
        == capture.projected_stats(plan["suite"]["fixtures"][0], config.to_dict())["bytes"]
    )
    dumps.close()


@pytest.fixture(scope="module")
def completed_run(tmp_path_factory):
    """Real tensor/transaction/native/oracle paths, CPU graph replay stand-in only."""
    import sys
    from contextlib import contextmanager
    from types import SimpleNamespace

    from vllm_lt.core import kv_cache_manager as kv_module
    from vllm_lt.kernels.paged_attention import torch_paged_attention
    from vllm_lt.worker import recurrent_graph as graph_module

    class Stream:
        def synchronize(self):
            pass

        def wait_stream(self, other):
            pass

    class Pool:
        def __init__(self, identifier):
            self.id = (0, identifier)

    class Graph:
        def __init__(self, runtime):
            self.runtime, self.body = runtime, None
            self.pool_id = None

        def capture_begin(self, *, pool, capture_error_mode):
            assert pool[0] == 0 and pool[1] > 0 and capture_error_mode == "global"
            self.pool_id = pool
            self.runtime.capturing = self

        def capture_end(self):
            self.runtime.capturing = None

        def replay(self):
            assert self.body is not None
            self.body()

        def pool(self):
            return self.pool_id

        def raw_cuda_graph_exec(self):
            return id(self)

        def reset(self):
            self.body = None

    class Runtime:
        def __init__(self):
            self.current, self.capturing = Stream(), None
            self.pool_count = 0

        def current_stream(self):
            return self.current

        def new_stream(self):
            return Stream()

        def release_stream(self, stream):
            pass

        @contextmanager
        def stream_context(self, stream):
            previous, self.current = self.current, stream
            try:
                yield
            finally:
                self.current = previous

        def new_graph(self):
            return Graph(self)

        def new_pool(self):
            self.pool_count += 1
            return Pool(self.pool_count)

        def memory(self):
            return dict.fromkeys(
                (
                    "allocated_bytes",
                    "reserved_bytes",
                    "peak_allocated_bytes",
                    "peak_reserved_bytes",
                ),
                0,
            )

        def reset_peaks(self):
            pass

    def scatter(keys, values, blocks, offsets, key, value, active):
        keys[blocks[active], offsets[active]] = key[active]
        values[blocks[active], offsets[active]] = value[active]

    with pytest.MonkeyPatch.context() as patch:

        def forbidden(*args, **kwargs):
            pytest.fail("CPU graph stand-in attempted CUDA")

        for name in (
            "is_available",
            "device_count",
            "current_device",
            "_lazy_init",
            "init",
            "synchronize",
            "memory_allocated",
            "memory_reserved",
            "MemPool",
            "get_allocator_backend",
        ):
            patch.setattr(torch.cuda, name, forbidden)
        patch.setattr(graph_module, "_make_runtime", lambda device: Runtime())
        initialize = kv_module.KVCacheManager.__init__

        def cpu_cache(self, *args, **kwargs):
            backend = kwargs.get("backend", "torch")
            kwargs["backend"] = "torch"
            initialize(self, *args, **kwargs)
            self.backend = backend

        patch.setattr(kv_module.KVCacheManager, "__init__", cpu_cache)
        patch.setattr(torch.nn.Module, "register_forward_hook", forbidden)
        patch.setattr(torch.nn.Module, "register_forward_pre_hook", forbidden)
        patch.setattr(kv_module, "triton_paged_attention", torch_paged_attention)
        patch.setitem(
            sys.modules, "vllm_lt.kernels.triton_kv_write", SimpleNamespace(masked_kv_write=scatter)
        )
        original_body = graph_module.RecurrentGraphExecutor._tensor_body

        def body(self, bucket):
            if self.runtime.capturing is not None:
                self.runtime.capturing.body = lambda: original_body(self, bucket)
            return original_body(self, bucket)

        patch.setattr(graph_module.RecurrentGraphExecutor, "_tensor_body", body)
        config = OuroConfig.tiny(vocab_size=49152, max_position_embeddings=512)
        torch.manual_seed(41)
        model = OuroForCausalLM(config)
        with torch.no_grad():
            model.lm_head.weight.zero_()
            model.model.early_exit_gate.weight.zero_()
            model.model.early_exit_gate.bias.fill_(math.log(0.4 / 0.6))
        parent = {
            "plan_sha256": "b" * 64,
            "numerical": capture.build_model_plan(*inputs(), config.to_dict()),
        }
        output = tmp_path_factory.mktemp("m3-capture-cpu")
        for implementation in ("A", "B"):
            result = capture.run_model_rows(
                model, parent, implementation, output, time.perf_counter_ns() + 180 * 10**9
            )
            assert result["complete"] and result["passed"], result["errors"]
        yield output, parent


def test_complete_projected_eager_replay_and_backend_fallback_cpu_pipeline(completed_run):
    output, parent = completed_run
    result = capture.audit_model_rows(output, parent, expected_device="cpu")
    assert result["complete"] and result["passed"], result["errors"]
    assert result["counts"]["verified_cases"] == 15
    assert result["counts"]["verified_comparisons"] == 31
    assert result["counts"]["verified_qualification_comparisons"] == 30
    assert result["counts"]["verified_replay_dispatches"] > 0
    assert result["counts"]["verified_backend_fallbacks"] == 64
    # CPU stand-ins cannot satisfy the real CUDA evidence contract.
    real = capture.audit_model_rows(output, parent)
    assert not real["complete"] and not real["passed"]
    assert "tensor device differs" in real["errors"][-1]["message"]


@pytest.mark.parametrize(
    "change",
    [
        "missing",
        "fake_replay",
        "wrong_bucket",
        "captured_pointer",
        "pool_alias",
        "generation",
        "counter",
        "request_alias",
        "cleanup",
        "setup_count",
    ],
)
def test_rehashed_or_missing_replay_pointer_lifetime_evidence_cannot_qualify(completed_run, change):
    output, parent = completed_run
    case = next(
        c for c in parent["numerical"]["execution_order"] if c["storage_strategy"] == "graph_replay"
    )
    value = read_json(output / "numerical/cases" / case["case_id"] / "result.json")
    events = value["graph_observations"]
    if change == "missing":
        events.pop()
    elif change == "fake_replay":
        events[0]["after"]["last_dispatch"]["kind"] = "eager"
    elif change == "wrong_bucket":
        events[0]["after"]["last_dispatch"]["bucket_id"] = 8
    elif change == "captured_pointer":
        value["graph_lifecycle"]["setup"]["buckets"]["4"]["captured_inputs"]["hidden"][
            "data_ptr"
        ] += 4
    elif change == "pool_alias":
        value["graph_lifecycle"]["setup"]["buckets"]["8"]["pool_id"] = [0, 999]
    elif change == "generation":
        events[0]["after"]["buckets"]["4"]["generation"] += 1
    elif change == "counter":
        events[0]["after"]["counters"]["replays"] = 0
    elif change == "request_alias":
        events[0]["request_hidden"][0]["storage_ptr"] = events[0]["after"]["buckets"]["4"][
            "tensors"
        ]["hidden_out"]["storage_ptr"]
    elif change == "cleanup":
        value["graph_lifecycle"]["after_close"]["buckets"] = value["graph_lifecycle"]["setup"][
            "buckets"
        ]
    elif change == "setup_count":
        value["graph_lifecycle"]["setup"]["setup"]["captures"] += 1
    with pytest.raises(ValueError):
        capture._audit_graph_case(
            case, value, parent["numerical"]["model_config"], expected_device="cpu"
        )


def test_missing_populated_kv_record_cannot_pass_even_with_rehashed_bytes(completed_run, tmp_path):
    import hashlib
    import json

    from vllm_lt.validation.report import _audit_comparison

    output, parent = completed_run
    view = capture.model_view(parent)
    comparison = next(c for c in view["comparison_order"] if c["family"] == "feasibility")
    comparison_id = comparison["comparison_id"]
    source = output / "numerical/comparisons" / f"{comparison_id}.jsonl"
    records = [json.loads(line) for line in source.read_text().splitlines()]
    removed = next(r for r in records if r["metadata"]["operation"] == "populated_kv")
    records.remove(removed)
    folder = tmp_path / "comparisons"
    folder.mkdir()
    payload = "".join(json.dumps(r, separators=(",", ":")) + "\n" for r in records).encode()
    (folder / source.name).write_bytes(payload)
    summary = read_json(source.with_suffix(".summary.json"))
    summary.update(
        sha256=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
        count=len(records),
        compared_count=len(records),
    )
    summary["by_operation"]["populated_kv"]["count"] -= 1
    capture.write_json(folder / f"{comparison_id}.summary.json", summary)
    cases = {
        case["case_id"]: read_json(output / "numerical/cases" / case["case_id"] / "result.json")
        for case in view["execution_order"]
    }
    fixtures = {f["fixture_id"]: f for f in view["suite"]["fixtures"]}
    with pytest.raises(ValueError, match="missing|coverage|boundar|records"):
        _audit_comparison(tmp_path, view, comparison, cases, fixtures)


def test_post_setup_failure_keeps_primary_and_records_failed_close(monkeypatch):
    from vllm_lt.validation import runner

    class StubRunner:
        def _enable_recurrent_graph(self, **kwargs):
            from types import SimpleNamespace

            self._decode_executor = SimpleNamespace()

        def _graph_snapshot(self):
            return {"enabled": True}

        def _close_recurrent_graph(self):
            raise RuntimeError("close failed")

    class StubEngine:
        def __init__(self, *args, **kwargs):
            self.model_runner = StubRunner()

    monkeypatch.setattr(runner, "ValidationEngine", StubEngine)
    original = runner.observe_native
    lifecycle = {}
    with pytest.raises(KeyboardInterrupt, match="primary interrupt") as caught:
        with capture._execution({"implementation_id": "B"}, [], lifecycle, capture.GRAPH_LIMITS):
            runner.ValidationEngine()
            raise KeyboardInterrupt("primary interrupt")
    assert "close failed" in caught.value.__notes__[0]
    assert lifecycle["close_error"] == {"type": "RuntimeError", "message": "close failed"}
    assert runner.ValidationEngine is StubEngine and runner.observe_native is original


def _settled_prefix(completed_run, destination, *, fail=False):
    """Copy a bounded settled A prefix; edits replace hardlinks before mutation."""
    import hashlib
    import json
    import os
    import shutil

    from vllm_lt.validation.diagnostics import compare
    from vllm_lt.validation.evidence import accumulate, empty_summary

    output, parent = completed_run
    view = capture.model_view(parent)
    selected = view["contract"]["diagnostics"]["preselected_dump_fixture_ids"]
    last = next(
        c
        for c in view["execution_order"]
        if c["implementation"] == "native"
        and c["phase"] == "validation"
        and selected[0] in c["fixture_ids"]
    )
    rows = view["execution_order"][: view["execution_order"].index(last) + 1]
    ids = [r["case_id"] for r in rows]
    comparisons = [r for r in view["comparison_order"] if r["candidate_case_id"] in ids]
    wanted = {r["comparison_id"] for r in comparisons}
    shutil.copytree(output / "numerical", destination / "numerical", copy_function=os.link)
    folder = destination / "numerical"
    for directory in ("cases", "spools"):
        for path in (folder / directory).iterdir():
            if path.name not in ids:
                shutil.rmtree(path)
    for path in (folder / "comparisons").iterdir():
        name = path.name.removesuffix(".jsonl").removesuffix(".summary.json")
        if name not in wanted:
            path.unlink()
    shutil.rmtree(folder / "dumps")
    ledger = read_json(folder / "ledger.json")
    ledger.update(
        completed_cases=ids,
        started_cases=ids,
        active_case=None,
        case_lifetimes={k: v for k, v in ledger["case_lifetimes"].items() if k in ids},
        workers={
            "A": {
                "implementation_id": "A",
                "complete": False,
                "passed": False,
                "completed_cases": ids,
                "errors": [],
            }
        },
        diagnostic_dumps={
            "selected_fixture_ids": selected,
            "written_bytes": 0,
            "fixture_written_bytes": {},
        },
    )
    groups = {}
    for row in rows:
        size = sum(p.stat().st_size for p in (folder / "spools" / row["case_id"]).glob("*.bin"))
        if size:
            groups[row["spool_group"]] = size
    ledger["spool_bytes_by_group"], ledger["tensor_written_bytes"] = groups, sum(groups.values())
    if fail:
        comparison = next(
            r
            for r in comparisons
            if r["candidate_case_id"] == last["case_id"] and r["fixture_id"] == selected[0]
        )
        path = folder / "comparisons" / (comparison["comparison_id"] + ".jsonl")
        records = [json.loads(line) for line in path.read_text().splitlines()]
        record = next(r for r in records if r["metadata"]["operation"] == "logits")
        # Real comparison routine supplies a finite allclose failure with unchanged
        # top1=0 and top-two margin=0, matching the independently audited trace.
        record["stats"] = compare(
            torch.ones(49152),
            torch.zeros(49152),
            operation="logits",
            policy=view["contract"]["comparison_policy"],
        )
        payload = "".join(
            json.dumps(r, sort_keys=True, separators=(",", ":")) + "\n" for r in records
        ).encode()
        path.unlink()
        path.write_bytes(payload)
        summary_path = path.with_suffix(".summary.json")
        previous = read_json(summary_path)
        summary = empty_summary(comparison)
        for item in records:
            accumulate(summary, item)
        for key in ("behavior", "behavior_failures", "first_behavior_failure"):
            summary[key] = previous[key]
        summary.update(
            complete=True, sha256=hashlib.sha256(payload).hexdigest(), size_bytes=len(payload)
        )
        capture.write_json(summary_path, {**previous, **summary})
        ledger["workers"]["A"]["errors"] = [
            {
                "case_id": last["case_id"],
                "comparison_id": comparison["comparison_id"],
                "message": "required numerical or actual-decision gate failed",
            }
        ]
    capture.write_json(folder / "ledger.json", ledger)
    return destination, parent


def test_settled_missing_suffix_is_valid_incomplete_not_passed(completed_run, tmp_path):
    output, parent = _settled_prefix(completed_run, tmp_path / "prefix")
    result = capture.audit_completed_prefix(output, parent, expected_device="cpu")
    assert result["valid_prefix"] and result["evidence_status"] == "incomplete", result["errors"]
    assert result["errors"] == [] and not result["complete"] and not result["passed"]
    assert result["missing_case_ids"] and not result["known_required_failure"]
    assert result["raw_evidence"]["retained_references"]


def test_prefix_failure_requires_valid_streams_anchors_and_graph_proof(completed_run, tmp_path):
    output, parent = _settled_prefix(completed_run, tmp_path / "failed-prefix", fail=True)
    result = capture.audit_completed_prefix(output, parent, expected_device="cpu")
    assert result["valid_prefix"] and result["known_required_failure"], result["errors"]
    assert result["required_failures"] == 1 and result["behavior_failures"] == 0
    assert not result["complete"] and not result["passed"]


@pytest.mark.parametrize("change", ["anchor", "summary", "pointer", "lifetime", "worker_claim"])
def test_corrupt_prefix_never_establishes_trusted_failure(completed_run, tmp_path, change):
    output, parent = _settled_prefix(completed_run, tmp_path / change, fail=True)
    folder = output / "numerical"
    if change == "anchor":
        path = next((folder / "spools").glob("*/*.bin"))
        contents = path.read_bytes()
        path.unlink()
        path.write_bytes(bytes([contents[0] ^ 1]) + contents[1:])
    elif change == "summary":
        path = next((folder / "comparisons").glob("*.summary.json"))
        value = read_json(path)
        value["required_failures"] += 1
        capture.write_json(path, value)
    elif change == "pointer":
        ledger = read_json(folder / "ledger.json")
        path = folder / "cases" / ledger["completed_cases"][0] / "result.json"
        value = read_json(path)
        value["graph_observations"].pop()
        capture.write_json(path, value)
    else:
        path = folder / "ledger.json"
        ledger = read_json(path)
        if change == "lifetime":
            ledger["case_lifetimes"][ledger["completed_cases"][0]]["finished_ns"] = 2**62
        else:
            ledger["workers"]["A"]["errors"] = []
        capture.write_json(path, ledger)
    result = capture.audit_completed_prefix(output, parent, expected_device="cpu")
    assert result["evidence_status"] == "invalid" and result["errors"]
    assert not result["valid_prefix"] and not result["known_required_failure"]


def test_pending_case_is_unqualified_without_invented_corruption(completed_run, tmp_path):
    output, parent = _settled_prefix(completed_run, tmp_path / "active")
    path = output / "numerical/ledger.json"
    ledger = read_json(path)
    case = parent["numerical"]["execution_order"][len(ledger["completed_cases"])]
    ledger["started_cases"] = [*ledger["completed_cases"], case["case_id"]]
    ledger["active_case"] = {"case_id": case["case_id"], "started_ns": 100, "deadline_ns": 200}
    capture.write_json(path, ledger)
    result = capture.audit_completed_prefix(output, parent, expected_device="cpu")
    assert result["evidence_status"] == "incomplete" and result["errors"] == []
    assert not result["valid_prefix"] and not result["known_required_failure"]
