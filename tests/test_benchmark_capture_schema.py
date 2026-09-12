"""Freeze budgets and byte identities without CUDA discovery or tensor loading."""

from collections import Counter
from copy import deepcopy
from pathlib import Path

import pytest
import torch

from benchmarks.capture import schema
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


@pytest.fixture(autouse=True)
def no_cuda_or_weights(forbid_cuda, monkeypatch):
    import safetensors.torch

    monkeypatch.setattr(torch, "load", forbid_cuda)
    monkeypatch.setattr(safetensors.torch, "load_file", forbid_cuda)


@pytest.fixture
def capture_plan(tmp_path, monkeypatch):
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
    write_json(
        contract, read_json(ROOT / "benchmarks/capture/fixtures/ouro-m3-capture-contract.json")
    )
    return schema.make_plan(
        baseline_root=roots["A"],
        candidate_root=roots["B"],
        contract_path=contract,
        model_path=model,
        gpu_ids=[0],
        affinity=deepcopy(AFFINITY),
    )


def test_plan_binds_capture_sources_order_and_budgets(capture_plan):
    schema.verify_plan(capture_plan)
    paths = {row["path"] for row in capture_plan["harness"]["files"]}
    assert {
        name.replace(".", "/") + ".py"
        for name in schema.IMPORT_MODULES
        if name.startswith("benchmarks.capture.")
    } <= paths
    rows = capture_plan["execution_order"]
    assert len(rows) == len({r["execution_id"] for r in rows}) == 93
    assert Counter(r["kind"] for r in rows) == {
        "model": 15,
        "benchmark": 78,
    }
    measured = [r for r in rows if r["phase"] == "measured"]
    assert [r["worker_id"] for r in measured] == [
        w for w in ("A1", "B1", "B2", "A2") for _ in range(7)
    ]
    assert len(set(r["pair_id"] for r in measured)) == 14
    assert set(Counter(r["pair_id"] for r in measured).values()) == {2}
    assert all(r["use_graphs"] == (r["implementation_id"] == "B") for r in rows)
    assert all(
        r["pair_id"] is None for r in rows if r["kind"] == "benchmark" and r["phase"] != "measured"
    )
    assert capture_plan["resource_estimates"]["artifact_bytes_upper_bound"] < 16 * 1024**3
    assert capture_plan["resource_estimates"]["total_device_scratch_traversals"] == 500
    assert capture_plan["resource_estimates"]["capture_recordings"] == 125
    assert capture_plan["graph_limits"]["common_payload_bytes"] == 1024**2


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: p.update(schema_version=1),
        lambda p: p["contract"]["implementation_options"]["B"].update(use_graphs=1),
        lambda p: p["contract"]["acceptance"].update(target_ttft_ratio_max=1.1),
        lambda p: p["execution_order"].reverse(),
        lambda p: p["implementations"]["B"]["source"].update(commit="b" * 40),
        lambda p: p["implementations"]["B"]["imports"].update(
            {schema.IMPORT_MODULES[-1]: "/elsewhere.py"}
        ),
        lambda p: p["numerical"]["execution_order"].pop(),
        lambda p: p["graph_limits"].update(graph_retained_reserved_bytes=2**30),
    ],
)
def test_rehashed_protocol_tampering_rejected(capture_plan, mutate):
    changed = deepcopy(capture_plan)
    mutate(changed)
    with pytest.raises((ValueError, TypeError)):
        schema.validate_plan(rehash(changed))


@pytest.mark.parametrize("change", ["source", "weights", "affinity"])
def test_verify_rejects_frozen_control_drift(capture_plan, monkeypatch, change):
    if change == "source":
        root = Path(capture_plan["implementations"]["B"]["root"])
        (root / "benchmarks/capture/runtime.py").write_text("changed")
    elif change == "weights":
        (Path(capture_plan["model_path"]) / "model.safetensors").write_bytes(b"changed")
    else:
        changed = deepcopy(AFFINITY)
        changed["numactl_show"]["membind"] = "0"
        monkeypatch.setattr(schema, "affinity_snapshot", lambda: changed)
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
