"""Freeze the actual M2 matrix using byte-only fake checkpoints and no accelerator."""

from collections import Counter
from copy import deepcopy
from pathlib import Path

import pytest
import torch

from vllm_lt.benchmarks import ab_schema as schema
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


@pytest.fixture
def ab_plan(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("CPU plan must not initialize/discover CUDA or deserialize weight tensors")

    for name in ("is_available", "device_count", "current_device", "_lazy_init"):
        monkeypatch.setattr(torch.cuda, name, forbidden)
    monkeypatch.setattr(torch, "load", forbidden)
    import safetensors.torch

    monkeypatch.setattr(safetensors.torch, "load_file", forbidden)
    model = tmp_path / "model"
    model.mkdir()
    write_json(model / "config.json", OuroConfig().to_dict())
    (model / "tokenizer.json").write_text('{"fake":"byte-only-tokenizer"}\n')
    (model / "tokenizer_config.json").write_text("{}\n")
    (model / "model.safetensors").write_bytes(b"not weight tensors; hashing only")
    tokenizer = _file_record(model / "tokenizer.json")
    deps = schema.dependency_manifest()
    deps["official"]["dependencies"]["transformers"] = "4.55.0"
    deps["official"]["dependencies"]["kernels"] = None
    deps["official"]["optional_kernels_present"] = False
    roots = {key: tmp_path / key for key in ("A", "B")}
    for key, root in roots.items():
        for relative in schema.INPUTS.values():
            contents = read_json(ROOT / relative)
            if "provenance" in contents:
                for field in ("size_bytes", "sha256"):
                    contents["provenance"]["tokenizer"][field] = tokenizer[field]
            (root / relative).parent.mkdir(parents=True, exist_ok=True)
            write_json(root / relative, contents)
        for relative in [*schema.DIFF_PATHS, "vllm_lt/benchmarks/ab.py"]:
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(key if relative in schema.DIFF_PATHS else "same common harness")

    def probe(root):
        root = Path(root)
        return {
            "source": {
                "root": str(root),
                "commit": ("a" if root.name == "A" else "b") * 40,
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
    write_json(contract, read_json(ROOT / "benchmarks/fixtures/ouro-m2-contract.json"))
    return schema.make_ab_plan(
        baseline_root=roots["A"],
        candidate_root=roots["B"],
        contract_path=contract,
        model_path=model,
        gpu_ids=[7],
        affinity=deepcopy(AFFINITY),
    )


def test_cpu_probe_resolves_exact_order_counts_and_shared_controls(ab_plan):
    schema.verify_ab_plan(ab_plan)
    rows = ab_plan["execution_order"]
    assert len(rows) == len({r["execution_id"] for r in rows}) == 105
    assert [r["worker_id"] for r in ab_plan["workers"]] == list(schema.WORKERS)
    assert Counter(r.get("phase", "numerical") for r in rows) == {
        "numerical": 27,
        "feasibility": 14,
        "warmup": 32,
        "measured": 28,
        "profile": 4,
    }
    timed = [r for r in rows if r.get("phase") == "measured"]
    assert [r["worker_id"] for r in timed] == [
        x for x in ("A1", "B1", "B2", "A2") for _ in range(7)
    ]
    assert set(Counter(r["pair_id"] for r in timed).values()) == {2}
    assert len({r["controls_sha256"] for r in timed}) == 1
    assert all(r["case_lifetime_timeout_s"] == 600 for r in timed)
    assert all(r.get("pair_id") is None for r in rows if r.get("phase") != "measured")
    assert ab_plan["numerical"]["resource_estimates"]["cases"] == 27


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: p["contract"]["controls"].update(gpu_ids=None),
        lambda p: p["contract"]["controls"].update(gpu_ids=[True]),
        lambda p: p["contract"]["limits"].update(executions=106),
        lambda p: p["contract"]["acceptance"].update(target_ratio_min=1.01),
        lambda p: p["execution_order"].reverse(),
        lambda p: p["execution_order"][0].update(instrumentation="timing"),
        lambda p: p["implementations"]["B"]["source"].update(status=" M arbitrary.py"),
        lambda p: p["implementations"]["A"]["imports"].pop(schema.IMPORT_MODULES[0]),
        lambda p: p["implementations"]["B"]["imports"].update(
            {schema.IMPORT_MODULES[0]: "/elsewhere.py"}
        ),
        lambda p: p["contract"]["controls"]["affinity"]["numactl_show"].update(policy="default"),
        lambda p: p["dependencies"]["official"].update(optional_kernels_present=True),
        lambda p: p.update(unrecognized=True),
    ],
)
def test_rehashed_invalid_contracts_and_rows_are_rejected(ab_plan, mutate):
    plan = deepcopy(ab_plan)
    mutate(plan)
    rehash(plan)
    with pytest.raises((ValueError, TypeError)):
        schema.validate_ab_plan(plan)


@pytest.mark.parametrize("change", ["weight", "input", "source", "affinity", "environment"])
def test_runtime_verification_detects_frozen_control_drift(ab_plan, monkeypatch, change):
    if change == "weight":
        (Path(ab_plan["model_path"]) / "model.safetensors").write_bytes(b"changed")
    elif change == "input":
        Path(ab_plan["inputs"]["benchmark_suite"]["file"]["path"]).write_text("{}\n")
    elif change == "source":
        (Path(ab_plan["implementations"]["A"]["root"]) / schema.DIFF_PATHS[0]).write_text("changed")
    elif change == "affinity":
        altered = deepcopy(AFFINITY)
        altered["numactl_show"]["membind"] = "0"
        monkeypatch.setattr(schema, "affinity_snapshot", lambda: altered)
    else:
        monkeypatch.setenv("OMP_NUM_THREADS", "changed")
    with pytest.raises(ValueError):
        schema.verify_ab_plan(ab_plan)


def test_json_rejects_duplicate_keys_nan_and_overflow():
    for value in ('{"x":1,"x":2}', '{"x":NaN}', '{"x":1e999}'):
        with pytest.raises(ValueError):
            schema.read_json_string(value)


def test_active_numa_policy_is_distinct_from_cpuset_allowance(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(
        schema.subprocess,
        "run",
        lambda *a, **kw: SimpleNamespace(
            stdout="policy: bind\npreferred node: 1\nphyscpubind: 56 57\n"
            "cpubind: 1\nnodebind: 1\nmembind: 1\n",
            stderr="",
        ),
    )
    # Model the Linux benchmark host even when the CPU suite runs on macOS.
    monkeypatch.setattr(schema.os, "sched_getaffinity", lambda _: {56, 57}, raising=False)
    snapshot = schema.affinity_snapshot()
    assert snapshot["numactl_show"]["membind"] == "1"
    assert snapshot["numactl_show"]["policy"] == "bind"
    schema.validate_affinity(snapshot)


def test_rehashed_embedded_input_cannot_differ_from_original_file(ab_plan):
    from vllm_lt.validation.m2 import build_numerical_plan
    from vllm_lt.validation.schema import FIXTURE_HASH_FIELDS

    changed = deepcopy(ab_plan)
    suite = changed["inputs"]["numerical_suite"]["contents"]
    suite["fixtures"][0]["continuation_input_ids"][0] += 1
    suite["fixtures_sha256"] = _digest({key: suite[key] for key in FIXTURE_HASH_FIELDS})
    changed["numerical"] = build_numerical_plan(
        suite, changed["inputs"]["numerical_contract"]["contents"], changed["model_config"]
    )
    schema.validate_ab_plan(rehash(changed))
    with pytest.raises(ValueError, match="embedded input"):
        schema.verify_ab_plan(changed)
