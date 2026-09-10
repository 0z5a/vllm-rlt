"""Freeze budgets and byte identities without CUDA discovery or tensor loading."""

from collections import Counter
from copy import deepcopy
from pathlib import Path

import pytest
import torch

from vllm_lt.benchmarks import capture_schema as schema
from vllm_lt.benchmarks.schema import _digest, _file_record, read_json, write_json
from vllm_lt.models import OuroConfig

ROOT = Path(__file__).resolve().parents[1]
AFFINITY = {
    "cpu_ids": [56, 57],
    "numa_status": ["Cpus_allowed_list:\t56-57", "Mems_allowed_list:\t0-1"],
    "numactl_show": {
        "policy": "bind",
        "preferred_node": "1",
        "physcpubind": "56 57",
        "cpubind": "1",
        "nodebind": "1",
        "membind": "1",
    },
}


def rehash(plan):
    plan["plan_sha256"] = _digest({k: v for k, v in plan.items() if k != "plan_sha256"})
    return plan


def held_plans():
    """Only shared ordering in these tests; exact real nested plans tested separately."""
    lifecycle = {
        "resource_estimates": {"artifact_bytes_upper_bound": 48 * 1024**2},
        "execution_order": [
            {
                "evaluation_id": f"LIFE-{side}-triton",
                "implementation_id": side,
                "backend": "triton",
                "use_graphs": side == "B",
            }
            for side in ("A", "B")
        ],
    }
    kernels = {
        "resource_estimates": {"artifact_bytes_upper_bound": 256 * 1024**2},
        "execution_order": [
            {
                "evaluation_id": f"K-{side}-triton-{layout}",
                "implementation_id": side,
                "backend": "triton",
                "layout_id": layout,
            }
            for side in ("A", "B")
            for layout in ("K4-zero", "K4-one-limit", "K4-two-crossing")
        ],
    }
    return lifecycle, kernels


@pytest.fixture(autouse=True)
def no_cuda_or_weights(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("CPU plan attempted accelerator discovery or weight deserialization")

    for name in (
        "is_available",
        "device_count",
        "current_device",
        "init",
        "_lazy_init",
        "synchronize",
    ):
        monkeypatch.setattr(torch.cuda, name, forbidden)
    monkeypatch.setattr(torch, "load", forbidden)
    import safetensors.torch

    monkeypatch.setattr(safetensors.torch, "load_file", forbidden)


@pytest.fixture
def capture_plan(tmp_path, monkeypatch):
    monkeypatch.setattr(schema, "_held_plans", held_plans)
    model = tmp_path / "model"
    model.mkdir()
    write_json(model / "config.json", OuroConfig().to_dict())
    write_json(model / "tokenizer.json", {"fake": "byte-only-tokenizer"})
    write_json(model / "tokenizer_config.json", {})
    (model / "model.safetensors").write_bytes(b"hash only; not model weights")
    tokenizer = _file_record(model / "tokenizer.json")
    deps = schema.ab.dependency_manifest()
    deps["official"]["dependencies"].update(transformers="4.55.0", kernels=None)
    deps["official"]["optional_kernels_present"] = False
    roots = {side: tmp_path / side for side in ("A", "B")}
    for root in roots.values():
        for name in schema.INPUTS.values():
            contents = read_json(ROOT / name)
            if "provenance" in contents:
                for field in ("sha256", "size_bytes"):
                    contents["provenance"]["tokenizer"][field] = tokenizer[field]
            (root / name).parent.mkdir(parents=True, exist_ok=True)
            write_json(root / name, contents)
        for name in schema.IMPORT_MODULES:
            path = root / (name.replace(".", "/") + ".py")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("same frozen source\n")

    def probe(root):
        root = Path(root)
        return {
            "source": {
                "root": str(root),
                "commit": "a" * 40,
                "status": "",
                "files": [
                    _file_record(path, relative_to=root)
                    for path in sorted(root.rglob("*"))
                    if path.is_file()
                ],
            },
            "imports": {name: name.replace(".", "/") + ".py" for name in schema.IMPORT_MODULES},
            "dependencies": deepcopy(deps),
        }

    monkeypatch.setattr(schema, "probe_checkout", probe)
    monkeypatch.setattr(schema, "affinity_snapshot", lambda: deepcopy(AFFINITY))
    contract = tmp_path / "contract.json"
    write_json(contract, read_json(ROOT / "benchmarks/fixtures/ouro-m3-capture-contract.json"))
    return schema.make_plan(
        baseline_root=roots["A"],
        candidate_root=roots["B"],
        contract_path=contract,
        model_path=model,
        gpu_ids=[0],
        affinity=deepcopy(AFFINITY),
    )


def test_cpu_plan_exact_order_exclusions_pairs_and_graph_work(capture_plan):
    schema.verify_plan(capture_plan)
    rows, workers = capture_plan["execution_order"], capture_plan["workers"]
    assert len(rows) == len({r["execution_id"] for r in rows}) == 101
    assert [w["worker_id"] for w in workers] == list(schema.WORKERS)
    assert [len(w["execution_ids"]) for w in workers] == [21, 16, 14, 14, 14, 14, 4, 4]
    assert Counter(r["kind"] for r in rows) == {
        "model": 15,
        "lifecycle": 2,
        "kernel": 6,
        "benchmark": 78,
    }
    for side in ("A", "B"):
        worker_rows = [r for r in rows if r["worker_id"] == "N-" + side]
        assert [r["kind"] for r in worker_rows[:5]] == [
            "model",
            "kernel",
            "kernel",
            "kernel",
            "lifecycle",
        ]
        assert worker_rows[0]["phase"] == "feasibility"
        assert all(
            r["kind"] == "benchmark" and r["phase"] == "feasibility" for r in worker_rows[-7:]
        )
    benchmark = [r for r in rows if r["kind"] == "benchmark"]
    assert Counter(r["phase"] for r in benchmark) == {
        "feasibility": 14,
        "warmup": 32,
        "measured": 28,
        "profile": 4,
    }
    measured = [r for r in benchmark if r["phase"] == "measured"]
    assert [r["worker_id"] for r in measured] == [
        w for w in ("A1", "B1", "B2", "A2") for _ in range(7)
    ]
    assert len({r["controls_sha256"] for r in measured}) == 1
    assert len(set(r["pair_id"] for r in measured)) == 14
    assert set(Counter(r["pair_id"] for r in measured).values()) == {2}
    assert all(r["pair_id"] is None for r in benchmark if r["phase"] != "measured")
    assert all(r["use_graphs"] == (r["implementation_id"] == "B") for r in rows)
    estimates = capture_plan["resource_estimates"]
    assert estimates["graph_owning_B_executions"] == 44
    assert estimates["graph_captures"] == estimates["capture_recordings"] == 88
    assert estimates["warmup_device_traversals"] == 264
    assert estimates["verification_device_traversals"] == 88
    assert estimates["total_device_scratch_traversals"] == 352
    assert estimates["numerical"]["retained_tensor_bytes_upper_bound"] == 2536572960
    assert estimates["numerical"]["comparison_records_upper_bound"] == 3879
    assert estimates["numerical"]["pointer_evidence_bytes_upper_bound"] < 200 * 1024**2
    assert estimates["artifact_bytes_upper_bound"] < 16 * 1024**3
    assert capture_plan["production_differences"] == []
    assert schema.execution_view(capture_plan)["plan_sha256"] == capture_plan["plan_sha256"]


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: p["contract"]["controls"].update(gpu_ids=None),
        lambda p: p["contract"]["controls"].update(gpu_ids=[True]),
        lambda p: p["contract"]["implementation_options"]["B"].update(use_graphs=False),
        lambda p: p["contract"]["implementation_options"]["B"].update(use_graphs=1),
        lambda p: p["contract"]["limits"].update(executions=102),
        lambda p: p["contract"]["graph_limits"].update(setup_timeout_s=61),
        lambda p: p["contract"]["acceptance"].update(target_ratio_min=1.01),
        lambda p: p["contract"]["acceptance"].update(target_ttft_ratio_max=1.1),
        lambda p: p["contract"]["acceptance"].update(peak_increase_bytes_max=2**30),
        lambda p: p["contract"]["acceptance"].update(setup_increase_ns_max=100000000),
        lambda p: p["contract"]["controls"]["affinity"]["numactl_show"].update(policy="default"),
        lambda p: p["execution_order"].reverse(),
        lambda p: p["execution_order"][0].update(use_graphs=True),
        lambda p: p["execution_order"][-1].update(instrumentation="timing"),
        lambda p: p["workers"][0]["execution_ids"].pop(),
        lambda p: p["implementations"]["B"]["source"].update(commit="b" * 40),
        lambda p: p["implementations"]["B"]["source"].update(status=" M x.py"),
        lambda p: p["implementations"]["B"]["source"]["files"][0].update(sha256="b" * 64),
        lambda p: p["implementations"]["A"]["imports"].pop(schema.IMPORT_MODULES[-1]),
        lambda p: p["implementations"]["B"]["imports"].update(
            {schema.IMPORT_MODULES[-1]: "/elsewhere.py"}
        ),
        lambda p: p["lifecycle"]["execution_order"].pop(),
        lambda p: p["kernels"]["execution_order"][0].update(layout_id="unplanned"),
        lambda p: p["resource_estimates"].update(total_device_scratch_traversals=440),
        lambda p: p["graph_limits"].update(graph_retained_reserved_bytes=2**30),
        lambda p: p["production_differences"].append("vllm_lt/models/ouro.py"),
        lambda p: p["benchmark_contract"]["engine"]["sampling"].update(ignore_eos=False),
        lambda p: p["dependencies"]["official"].update(optional_kernels_present=True),
        lambda p: p.update(unrecognized=True),
    ],
)
def test_rehashed_protocol_tampering_rejected(capture_plan, mutate):
    changed = deepcopy(capture_plan)
    mutate(changed)
    with pytest.raises((ValueError, TypeError)):
        schema.validate_plan(rehash(changed))


@pytest.mark.parametrize(
    "change", ["weights", "input", "source", "contract", "affinity", "environment", "config"]
)
def test_verify_rejects_actual_frozen_control_drift(capture_plan, monkeypatch, change):
    if change == "weights":
        (Path(capture_plan["model_path"]) / "model.safetensors").write_bytes(b"changed")
    elif change == "input":
        Path(capture_plan["inputs"]["numerical_suite"]["file"]["path"]).write_text("{}")
    elif change == "source":
        path = Path(capture_plan["implementations"]["B"]["root"]) / "vllm_lt/models/ouro.py"
        path.write_text("changed")
    elif change == "contract":
        Path(capture_plan["contract_file"]["path"]).write_text("{}")
    elif change == "config":
        (Path(capture_plan["model_path"]) / "config.json").write_text("{}")
    elif change == "affinity":
        changed = deepcopy(AFFINITY)
        changed["numactl_show"]["membind"] = "0"
        monkeypatch.setattr(schema, "affinity_snapshot", lambda: changed)
    else:
        monkeypatch.setenv("OMP_NUM_THREADS", "changed")
    with pytest.raises(ValueError):
        schema.verify_plan(capture_plan)


def test_embedded_input_is_bound_to_actual_parsed_frozen_bytes(capture_plan):
    from vllm_lt.validation.schema import FIXTURE_HASH_FIELDS

    changed = deepcopy(capture_plan)
    suite = changed["inputs"]["numerical_suite"]["contents"]
    suite["fixtures"][0]["continuation_input_ids"][0] += 1
    suite["fixtures_sha256"] = _digest({k: suite[k] for k in FIXTURE_HASH_FIELDS})
    changed["numerical"] = schema.build_model_plan(
        suite, changed["inputs"]["numerical_contract"]["contents"], changed["model_config"]
    )
    # Original file hashes remain unchanged: this is self-consistent invented content.
    schema.validate_plan(rehash(changed))
    with pytest.raises(ValueError, match="embedded input"):
        schema.verify_plan(changed)


def test_unresolved_template_and_nonfinite_json_cannot_run():
    contract = read_json(ROOT / "benchmarks/fixtures/ouro-m3-capture-contract.json")
    schema.validate_contract(contract)
    with pytest.raises(ValueError):
        schema.validate_contract(contract, resolved=True)
    for text in ('{"x":1,"x":2}', '{"x":NaN}', '{"x":1e999}'):
        with pytest.raises(ValueError):
            schema.read_json_string(text)


def test_actual_projected_lifecycle_kernel_builders_fit_combined_budget(capture_plan, monkeypatch):
    from vllm_lt.validation.m3_capture_kernels import build_kernel_plan
    from vllm_lt.validation.m3_capture_lifecycle import build_lifecycle_plan

    def actual_held():
        return build_lifecycle_plan(), build_kernel_plan()

    monkeypatch.setattr(schema, "_held_plans", actual_held)
    plan = deepcopy(capture_plan)
    plan["lifecycle"], plan["kernels"] = actual_held()
    schema.validate_plan(rehash(plan))
    schema.verify_plan(plan)
    assert len(plan["lifecycle"]["execution_order"]) == 2
    assert len(plan["kernels"]["execution_order"]) == 6
    assert plan["lifecycle"]["resource_estimates"]["artifact_bytes_upper_bound"] <= 48 * 1024**2
    assert plan["kernels"]["resource_estimates"]["artifact_bytes_upper_bound"] <= 256 * 1024**2
    assert next(s for s in plan["lifecycle"]["steps"] if s["step_id"] == 3)["positions"] == [511]
    assert plan["lifecycle"]["expected_executor"]["bucket_visits"] == {"4": 4, "8": 3}
