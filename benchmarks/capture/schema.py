"""CPU-only freeze for the single 101-execution eager/replay experiment."""

import importlib
import os
import subprocess
import sys
from pathlib import Path

from benchmarks.capture.validation import GRAPH_LIMITS, build_model_plan, validate_model_plan
from vllm_lt.benchmarks import ab_schema as ab
from vllm_lt.benchmarks.ab_schema import (
    CELLS,
    INPUTS,
    RUNTIME_VARIABLES,
    WORKERS,
    affinity_snapshot,
    equal,
    execution_view,
    read_json_string,
    require,
    validate_affinity,
    validate_model_config,
)
from vllm_lt.benchmarks.schema import (
    _constants,
    _digest,
    _file_record,
    _integer,
    _keys,
    _model_files,
    _text,
    _validate_contract,
    _validate_suite,
    _workload_stats,
    read_json,
)
from vllm_lt.models.config import OuroConfig
from vllm_lt.validation.schema import _file_records, _validate_dependencies

IMPORT_MODULES = (
    *ab.IMPORT_MODULES,
    "benchmarks.capture.schema",
    "benchmarks.capture.runner",
    "benchmarks.capture.report",
    "benchmarks.capture.runtime",
    "benchmarks.capture.validation",
    "benchmarks.capture.lifecycle",
    "benchmarks.capture.kernels",
    "vllm_lt.worker.recurrent_graph",
)
OPTIONS = {"A": {"use_graphs": False}, "B": {"use_graphs": True}}
CONTROLS = {
    **ab.CONTROL_CONSTANTS,
    "residency_policy": "one-fresh-worker-one-model-fresh-case-engine",
    "environment_policy": "same-prepared-Q1-environment-all-workers",
    "source_policy": "same-clean-commit-and-all-source-bytes",
    "graph_setup_policy": "fresh-executor-before-admission-outside-measured-arrival-window",
    "timing_observer": "lightweight-host-events-and-executor-counters",
    "diagnostic_observer": "correctness-and-profile-only",
}
LIMITS = {
    "total_timeout_s": 7200,
    "case_timeout_s": 600,
    "workers": 8,
    "model_loads": 8,
    "executions": 101,
    "correctness_executions": 23,
    "model_cases": 15,
    "model_qualification_cases": 13,
    "model_excluded_feasibility_cases": 2,
    "comparison_streams": 31,
    "qualification_comparison_streams": 30,
    "lifecycle_evaluations": 2,
    "kernel_evaluations": 6,
    "feasibility_executions": 14,
    "warmup_executions": 28,
    "measured_executions": 28,
    "diagnostic_warmup_executions": 4,
    "profile_executions": 4,
    "profile_decode_outputs": 16,
    "profile_trace_bytes_max": 2 * 1024**3,
    "profile_total_bytes_max": 4 * 1024**3,
    "artifact_bytes_max": 16 * 1024**3,
    "pointer_evidence_bytes_max": 200 * 1024**2,
    "lifecycle_evidence_bytes_max": 48 * 1024**2,
    "kernel_evidence_bytes_max": 256 * 1024**2,
    "sanitizer_executions": 0,
}
ACCEPTANCE = {
    "target_cell": "W1-refill",
    "target_ratio_min": 1.1,
    "control_cells": [cell for cell in CELLS if cell != "W1-refill"],
    "control_ratio_min": 0.95,
    "target_ttft_ratio_max": 1.05,
    "peak_increase_bytes_max": 512 * 1024**2,
    "pairs": [["A1", "B1"], ["A2", "B2"]],
    "pair_gates": "each-pair-not-only-aggregate",
    "variation": "min-B-strictly-greater-than-max-A",
    "verdict_precedence": "invalid-evidence-then-required-failure-then-missing-or-inconclusive",
    "numerical": "all-original-Q1-FP32-required-logit-token-exit-and-full-populated-KV-gates",
    "intermediates": "finite-required-deltas-diagnostic-no-new-tolerance",
    "lifecycle": "both-buckets-exact-held-input-outputs-guards-transactions-and-fallbacks",
    "profile": "actual-correlated-replay-launch-proof-for-W1-and-W4-refill",
    "setup": "reported-separately-original-graph-limits-no-M2-relative-setup-cap",
    "sanitizer": "unavailable-no-executions-no-coverage-claim",
}
STOP_CONDITIONS = [
    "Any execution, nonfinite, ownership, source, input, dependency, device, affinity or cleanup "
    "failure stops subsequent executions; retain the failed prefix and do not retry.",
    "All correctness and excluded feasibility gates must pass before measured A1.",
    "Independent parent watchdogs enforce case 600-second and total 7200-second limits, "
    "including setup, export and cleanup. Setup must complete within 60 seconds to qualify; "
    "its phase checks are cooperative, while blocked device calls remain subject to the "
    "case/global watchdogs. No hard 60-second interruption is claimed.",
    "Any frozen tensor, pointer, lifecycle, kernel, trace or total artifact budget overrun.",
    "Performance misses and overlapping ranges cannot promote the candidate or expand the matrix.",
]


def validate_contract(contract, *, resolved=False):
    _keys(
        contract,
        (
            "schema_version",
            "artifact_type",
            "contract_id",
            "hypothesis",
            "isolated_variable",
            "inputs",
            "implementation_options",
            "controls",
            "acceptance",
            "limits",
            "graph_limits",
            "stop_conditions",
        ),
        name="capture contract",
    )
    equal(
        [contract["schema_version"], contract["artifact_type"]],
        [1, "m3_capture_contract"],
        "version",
    )
    for name in ("contract_id", "hypothesis", "isolated_variable"):
        _text(contract[name], name)
    for name, expected in (
        ("inputs", INPUTS),
        ("implementation_options", OPTIONS),
        ("acceptance", ACCEPTANCE),
        ("limits", LIMITS),
        ("graph_limits", GRAPH_LIMITS),
    ):
        _constants(contract[name], expected, "capture " + name)
        equal(contract[name], expected, "strict capture " + name)
    equal(contract["stop_conditions"], STOP_CONDITIONS, "bounded stop conditions")
    controls = contract["controls"]
    _keys(controls, (*CONTROLS, "gpu_ids", "affinity"), name="capture controls")
    _constants({key: controls[key] for key in CONTROLS}, CONTROLS, "fixed controls")
    if resolved or controls["gpu_ids"] is not None:
        require(
            isinstance(controls["gpu_ids"], list) and len(controls["gpu_ids"]) == 1,
            "one explicitly selected physical GPU required",
        )
        _integer(controls["gpu_ids"][0], "physical GPU ID")
    if resolved or controls["affinity"] is not None:
        validate_affinity(controls["affinity"])


def source_probe():
    """Byte hashes, import origins and package versions; no model or CUDA access."""
    result = ab.source_probe()
    root = Path(result["source"]["root"]).resolve()
    for name in IMPORT_MODULES:
        path = Path(importlib.import_module(name).__file__).resolve()
        require(path.is_relative_to(root), f"import outside selected source: {name}")
        result["imports"][name] = str(path.relative_to(root))
    return result


def probe_checkout(root):
    root = Path(root).resolve()
    code = (
        "import json; from benchmarks.capture.schema import source_probe; "
        "print(json.dumps(source_probe(), allow_nan=False))"
    )
    result = read_json_string(
        subprocess.check_output(
            [sys.executable, "-c", code],
            cwd=root,
            env=dict(os.environ, PYTHONPATH=str(root)),
            text=True,
        )
    )
    equal(result["source"]["root"], str(root), "source probe checkout")
    return result


def _source_controls(implementations):
    _keys(implementations, ("A", "B"), name="implementations")
    for item in implementations.values():
        _keys(item, ("root", "source", "imports"), name="implementation")
        source = item["source"]
        _keys(source, ("root", "commit", "status", "files"), name="source")
        require(Path(item["root"]).is_absolute(), "absolute implementation root required")
        equal(item["root"], source["root"], "source root")
        require(source["status"] == "", "execution source must be clean")
        require(
            isinstance(source["commit"], str)
            and len(source["commit"]) == 40
            and all(c in "0123456789abcdef" for c in source["commit"]),
            "exact source SHA",
        )
        _file_records(source["files"], "source files")
        paths = [row["path"] for row in source["files"]]
        equal(paths, sorted(paths), "sorted source manifest")
        for path in paths:
            require(
                not Path(path).is_absolute() and ".." not in Path(path).parts,
                "source paths must remain inside checkout",
            )
        _keys(item["imports"], IMPORT_MODULES, name="source imports")
        for name, path in item["imports"].items():
            equal(path, name.replace(".", "/") + ".py", "module import origin")
            require(path in paths, "imported module absent from frozen source")
    a, b = (implementations[side]["source"] for side in ("A", "B"))
    equal(a["commit"], b["commit"], "A/B identical commit")
    equal(a["files"], b["files"], "A/B all source bytes identical")
    common_paths = {row["path"] for row in ab._harness(a)["files"]}
    files = [
        row
        for row in a["files"]
        if row["path"] in common_paths
        or row["path"].startswith("benchmarks/capture/")
        or row["path"] == "benchmarks/__init__.py"
    ]
    return {"files": files, "sha256": _digest(files)}


def execution_rows(suite, contract, stats, numerical, lifecycle, kernels):
    workloads = {row["workload_id"]: row for row in suite["workloads"]}
    rows, workers = [], []
    controls_hash = _digest(contract)
    for worker_id in WORKERS:
        side = "A" if "A" in worker_id else "B"
        worker = {
            "worker_id": worker_id,
            "implementation_id": side,
            "use_graphs": side == "B",
            "execution_ids": [],
        }

        def append(row):
            row.update(worker_id=worker_id, implementation_id=side, use_graphs=side == "B")
            rows.append(row)
            worker["execution_ids"].append(row["execution_id"])

        def benchmark(cell, phase, repetition=1):
            workload_id, mode = cell.split("-", 1)
            run_id = f"{worker_id}-{phase}-{cell}"
            append(
                {
                    "execution_id": run_id,
                    "kind": "benchmark",
                    "run_id": run_id,
                    "cell_id": cell,
                    "workload_id": workload_id,
                    "mode": mode,
                    "phase": phase,
                    "repetition": repetition,
                    "pair_id": f"M3-capture-{cell}-{repetition}" if phase == "measured" else None,
                    "instrumentation": {"feasibility": "validation", "profile": "profile"}.get(
                        phase, "timing"
                    ),
                    "case_lifetime_timeout_s": LIMITS["case_timeout_s"],
                    "max_steps": stats[workload_id]["max_steps"],
                    "max_events": stats[workload_id]["max_events"],
                    "workload_sha256": _digest(workloads[workload_id]),
                    "controls_sha256": controls_hash,
                }
            )

        if worker_id.startswith("N-"):
            cases = [r for r in numerical["execution_order"] if r["implementation_id"] == side]
            require(cases[0]["family"] == "feasibility", "first model must be excluded feasibility")
            append({"execution_id": cases[0]["case_id"], "kind": "model", "phase": "feasibility"})
            for kind, plan in (("kernel", kernels), ("lifecycle", lifecycle)):
                for row in plan["execution_order"]:
                    if row["implementation_id"] == side:
                        append(
                            {
                                "execution_id": row["evaluation_id"],
                                "kind": kind,
                                "phase": "validation",
                            }
                        )
            for row in cases[1:]:
                append({"execution_id": row["case_id"], "kind": "model", "phase": "validation"})
            for cell in CELLS:
                benchmark(cell, "feasibility")
        elif worker_id.startswith("P-"):
            for phase in ("warmup", "profile"):
                for cell in ("W1-refill", "W4-refill"):
                    benchmark(cell, phase)
        else:
            for phase in ("warmup", "measured"):
                for cell in CELLS:
                    benchmark(cell, phase, int(worker_id[-1]))
        workers.append(worker)
    require(
        len(rows) == len({r["execution_id"] for r in rows}) == 101,
        "exact101 unique executions required",
    )
    return workers, rows


def resource_estimates(numerical, rows):
    r = numerical["resource_estimates"]
    graph_models = sum(
        c["implementation_id"] == "B" and c["storage_strategy"] == "graph_replay"
        for c in numerical["execution_order"]
    )
    graph_cases = graph_models + sum(
        row["implementation_id"] == "B" and row["kind"] in ("benchmark", "lifecycle")
        for row in rows
    )
    equal(graph_cases, 44, "B graph-owning cases")
    pointer = r["pointer_evidence_bytes_upper_bound"]
    require(pointer <= LIMITS["pointer_evidence_bytes_max"], "projected pointer evidence cap")
    total = (
        r["retained_tensor_bytes_upper_bound"]
        + r["retained_index_records_upper_bound"] * 1024
        + r["comparison_records_upper_bound"] * 2048
        + LIMITS["pointer_evidence_bytes_max"]
        + numerical["contract"]["limits"]["persisted_dump_bytes"]
        + LIMITS["lifecycle_evidence_bytes_max"]
        + LIMITS["kernel_evidence_bytes_max"]
        + LIMITS["profile_total_bytes_max"]
        + 1024**3
    )
    require(total <= LIMITS["artifact_bytes_max"], "estimated artifacts exceed total cap")
    return {
        "numerical": r,
        "artifact_bytes_upper_bound": total,
        "auxiliary_evidence_allowance_bytes": 1024**3,
        "graph_owning_B_executions": graph_cases,
        "graph_captures": graph_cases * 2,
        "capture_recordings": graph_cases * 2,
        "warmup_device_traversals": graph_cases * 2 * 3,
        "verification_device_traversals": graph_cases * 2,
        "total_device_scratch_traversals": graph_cases * 2 * 4,
        "capture_semantics": "recording-does-not-execute-captured-device-kernels",
    }


def _held_plans():
    from benchmarks.capture.kernels import build_kernel_plan
    from benchmarks.capture.lifecycle import build_lifecycle_plan

    return build_lifecycle_plan(), build_kernel_plan()


def make_plan(*, baseline_root, candidate_root, contract_path, model_path, gpu_ids, affinity):
    contract_path, model_path = Path(contract_path).resolve(), Path(model_path).resolve()
    contract = read_json(contract_path)
    validate_contract(contract)
    contract["controls"].update(gpu_ids=gpu_ids, affinity=affinity)
    validate_contract(contract, resolved=True)
    roots = {"A": Path(baseline_root).resolve(), "B": Path(candidate_root).resolve()}
    probes = {side: probe_checkout(root) for side, root in roots.items()}
    equal(probes["A"]["dependencies"], probes["B"]["dependencies"], "A/B dependencies")
    implementations = {
        side: {"root": str(roots[side]), "source": probe["source"], "imports": probe["imports"]}
        for side, probe in probes.items()
    }
    harness = _source_controls(implementations)
    inputs = {
        key: {"file": _file_record(roots["A"] / path), "contents": read_json(roots["A"] / path)}
        for key, path in INPUTS.items()
    }
    for key, path in INPUTS.items():
        equal(
            _file_record(roots["B"] / path, relative_to=roots["B"]),
            _file_record(roots["A"] / path, relative_to=roots["A"]),
            "A/B input " + key,
        )
    config = OuroConfig.from_dict(read_json(model_path / "config.json"))
    numerical = build_model_plan(
        inputs["numerical_suite"]["contents"],
        inputs["numerical_contract"]["contents"],
        config.to_dict(),
    )
    lifecycle, kernels = _held_plans()
    suite, benchmark = (
        inputs["benchmark_suite"]["contents"],
        inputs["benchmark_contract"]["contents"],
    )
    stats = _workload_stats(suite, config, benchmark)
    workers, rows = execution_rows(suite, contract, stats, numerical, lifecycle, kernels)
    plan = {
        "schema_version": 1,
        "artifact_type": "m3_capture_plan",
        "contract": contract,
        "contract_file": _file_record(contract_path),
        "implementations": implementations,
        "harness": harness,
        "production_differences": [],
        "dependencies": probes["A"]["dependencies"],
        "inputs": inputs,
        "model_path": str(model_path),
        "model_config": config.to_dict(),
        "model_files": _model_files(model_path),
        "suite": suite,
        "benchmark_contract": benchmark,
        "workload_stats": stats,
        "numerical": numerical,
        "lifecycle": lifecycle,
        "kernels": kernels,
        "graph_limits": dict(GRAPH_LIMITS),
        "workers": workers,
        "execution_order": rows,
        "resource_estimates": resource_estimates(numerical, rows),
        "interpreter": sys.executable,
        "runtime_environment": {name: os.environ.get(name) for name in RUNTIME_VARIABLES},
    }
    plan["plan_sha256"] = _digest(plan)
    validate_plan(plan)
    return plan


def validate_plan(plan):
    _keys(
        plan,
        (
            "schema_version",
            "artifact_type",
            "contract",
            "contract_file",
            "implementations",
            "harness",
            "production_differences",
            "dependencies",
            "inputs",
            "model_path",
            "model_config",
            "model_files",
            "suite",
            "benchmark_contract",
            "workload_stats",
            "numerical",
            "lifecycle",
            "kernels",
            "graph_limits",
            "workers",
            "execution_order",
            "resource_estimates",
            "interpreter",
            "runtime_environment",
            "plan_sha256",
        ),
        name="capture plan",
    )
    equal([plan["schema_version"], plan["artifact_type"]], [1, "m3_capture_plan"], "plan version")
    equal(
        plan["plan_sha256"],
        _digest({k: v for k, v in plan.items() if k != "plan_sha256"}),
        "plan content hash",
    )
    validate_contract(plan["contract"], resolved=True)
    equal(plan["harness"], _source_controls(plan["implementations"]), "common harness")
    equal(plan["production_differences"], [], "no source differences")
    _validate_dependencies(plan["dependencies"])
    official = plan["dependencies"]["official"]
    equal(official["dependencies"]["transformers"], "4.55.0", "prepared Q1 transformers")
    equal(official["optional_kernels_present"], False, "prepared Q1 optional kernels absence")
    _file_records(plan["model_files"], "model files")
    _file_records([plan["contract_file"]], "capture contract file")
    require(Path(plan["contract_file"]["path"]).is_absolute(), "absolute contract path required")
    for row in plan["model_files"]:
        require(
            Path(row["path"]).name == row["path"], "checkpoint file must remain inside model root"
        )
    _keys(plan["inputs"], INPUTS, name="input references")
    for name, item in plan["inputs"].items():
        _keys(item, ("file", "contents"), name="input")
        _file_records([item["file"]], "input file")
        equal(
            item["file"]["path"],
            str(Path(plan["implementations"]["A"]["root"]) / INPUTS[name]),
            "input belongs to frozen A source",
        )
        source_record = next(
            (
                r
                for r in plan["implementations"]["A"]["source"]["files"]
                if r["path"] == INPUTS[name]
            ),
            None,
        )
        require(source_record is not None, "input missing from source manifest")
        equal({**item["file"], "path": INPUTS[name]}, source_record, "input frozen source bytes")
    _validate_suite(plan["suite"])
    _validate_contract(plan["benchmark_contract"])
    validate_model_config(plan["model_config"])
    equal(plan["suite"], plan["inputs"]["benchmark_suite"]["contents"], "benchmark suite inputs")
    equal(
        plan["benchmark_contract"],
        plan["inputs"]["benchmark_contract"]["contents"],
        "benchmark policy",
    )
    validate_model_plan(plan["numerical"])
    equal(
        plan["numerical"]["source_inputs"],
        {
            "suite": plan["inputs"]["numerical_suite"]["contents"],
            "contract": plan["inputs"]["numerical_contract"]["contents"],
        },
        "original numerical inputs",
    )
    equal(plan["numerical"]["model_config"], plan["model_config"], "numerical model configuration")
    tokenizer = next((r for r in plan["model_files"] if r["path"] == "tokenizer.json"), None)
    require(tokenizer is not None, "checkpoint tokenizer required")
    for suite in (plan["suite"], plan["inputs"]["numerical_suite"]["contents"]):
        expected = suite["provenance"]["tokenizer"]
        equal(
            [tokenizer[k] for k in ("sha256", "size_bytes")],
            [expected[k] for k in ("sha256", "size_bytes")],
            "tokenizer provenance",
        )
    lifecycle, kernels = _held_plans()
    equal([plan["lifecycle"], plan["kernels"]], [lifecycle, kernels], "exact held-input plans")
    for kind, nested in (("lifecycle", lifecycle), ("kernel", kernels)):
        require(
            nested["resource_estimates"]["artifact_bytes_upper_bound"]
            <= LIMITS[kind + "_evidence_bytes_max"],
            kind + " nested artifact estimate exceeds combined cap",
        )
    equal(plan["graph_limits"], GRAPH_LIMITS, "graph setup/memory limits")
    equal(plan["numerical"]["graph_limits"], GRAPH_LIMITS, "numerical graph limits")
    stats = _workload_stats(
        plan["suite"], OuroConfig.from_dict(plan["model_config"]), plan["benchmark_contract"]
    )
    equal(plan["workload_stats"], stats, "derived workload limits")
    workers, rows = execution_rows(
        plan["suite"], plan["contract"], stats, plan["numerical"], lifecycle, kernels
    )
    equal([plan["workers"], plan["execution_order"]], [workers, rows], "exact101 execution order")
    equal(
        plan["resource_estimates"],
        resource_estimates(plan["numerical"], rows),
        "resource estimates",
    )
    _keys(plan["runtime_environment"], RUNTIME_VARIABLES, name="runtime environment")
    for value in plan["runtime_environment"].values():
        require(
            value is None or isinstance(value, str), "runtime variables must be strings or null"
        )
    for field in ("interpreter", "model_path"):
        require(
            isinstance(plan[field], str) and Path(plan[field]).is_absolute(), "absolute " + field
        )


def verify_plan(plan, *, implementation_id=None):
    validate_plan(plan)
    require(implementation_id in (None, "A", "B"), "unknown implementation")
    for side in (implementation_id,) if implementation_id else ("A", "B"):
        expected = plan["implementations"][side]
        actual = source_probe() if implementation_id else probe_checkout(expected["root"])
        equal(actual["source"], expected["source"], "frozen source")
        equal(actual["imports"], expected["imports"], "frozen imports")
        equal(actual["dependencies"], plan["dependencies"], "frozen dependencies")
    equal(_model_files(Path(plan["model_path"])), plan["model_files"], "frozen checkpoint files")
    equal(
        OuroConfig.from_dict(read_json(Path(plan["model_path"]) / "config.json")).to_dict(),
        plan["model_config"],
        "embedded checkpoint configuration",
    )
    equal(
        _file_record(Path(plan["contract_file"]["path"])), plan["contract_file"], "frozen contract"
    )
    original = read_json(Path(plan["contract_file"]["path"]))
    original["controls"].update(
        gpu_ids=plan["contract"]["controls"]["gpu_ids"],
        affinity=plan["contract"]["controls"]["affinity"],
    )
    equal(original, plan["contract"], "resolved contract contents")
    for name, item in plan["inputs"].items():
        equal(_file_record(Path(item["file"]["path"])), item["file"], "frozen input " + name)
        equal(read_json(Path(item["file"]["path"])), item["contents"], "embedded input " + name)
    equal(sys.executable, plan["interpreter"], "prepared interpreter")
    equal(
        {name: os.environ.get(name) for name in RUNTIME_VARIABLES},
        plan["runtime_environment"],
        "runtime environment",
    )
    equal(affinity_snapshot(), plan["contract"]["controls"]["affinity"], "CPU/NUMA affinity")


__all__ = [
    "make_plan",
    "validate_plan",
    "verify_plan",
    "source_probe",
    "probe_checkout",
    "execution_view",
]
