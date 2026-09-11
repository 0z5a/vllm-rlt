"""The Q2 CPU probe freezes eight actual-generation executions without tensors."""

import hashlib
import subprocess
from collections import Counter
from copy import deepcopy
from pathlib import Path

import pytest
import torch

from vllm_lt.benchmarks import q2_external_schema as schema
from vllm_lt.benchmarks.schema import _digest, _file_record, read_json, write_json
from vllm_lt.models.config import OuroConfig

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
    plan["controls_sha256"] = schema._controls_hash(plan)
    plan["plan_sha256"] = _digest({k: v for k, v in plan.items() if k != "plan_sha256"})
    return plan


@pytest.fixture
def external_plan(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("CPU probe must not discover CUDA or deserialize checkpoint tensors")

    for name in ("is_available", "device_count", "current_device", "_lazy_init", "init"):
        monkeypatch.setattr(torch.cuda, name, forbidden)
    monkeypatch.setattr(torch, "load", forbidden)
    import safetensors.torch

    monkeypatch.setattr(safetensors.torch, "load_file", forbidden)
    model = tmp_path / "model"
    model.mkdir()
    write_json(model / "config.json", OuroConfig().to_dict())
    (model / "tokenizer.json").write_bytes(b"byte-only tokenizer")
    (model / "tokenizer_config.json").write_text("{}")
    (model / "model.safetensors").write_bytes(b"only hash these bytes")
    root = tmp_path / "source"
    fixture = read_json(ROOT / "benchmarks/fixtures/ouro-q2-external-contract.json")
    for record in fixture["production_files"]:
        target = root / record["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        data = (ROOT / record["path"]).read_bytes()
        if hashlib.sha256(data).hexdigest() != record["sha256"]:
            # The contract is a historical measurement, not the current branch.
            # Keep testing its real immutable bytes after prerequisite fixes.
            data = subprocess.check_output(
                ["git", "show", f"{fixture['base_sha']}:{record['path']}"], cwd=ROOT
            )
        assert hashlib.sha256(data).hexdigest() == record["sha256"]
        target.write_bytes(data)
    for name in schema.IMPORT_MODULES:
        relative = name.replace(".", "/") + ".py"
        if not (root / relative).exists():
            (root / relative).parent.mkdir(parents=True, exist_ok=True)
            (root / relative).write_text("# fake common harness file\n")
    suite = read_json(ROOT / schema.INPUTS["suite"])
    token = _file_record(model / "tokenizer.json")
    for field in ("size_bytes", "sha256"):
        suite["provenance"]["tokenizer"][field] = token[field]
    suite_file = root / schema.INPUTS["suite"]
    suite_file.parent.mkdir(parents=True, exist_ok=True)
    write_json(suite_file, suite)
    deps = schema.dependency_manifest()
    deps["official"]["dependencies"]["transformers"] = "4.55.0"
    deps["official"]["dependencies"]["kernels"] = None
    deps["official"]["optional_kernels_present"] = False

    def probe():
        files = [_file_record(p, relative_to=root) for p in sorted(root.rglob("*")) if p.is_file()]
        shim = next(r for r in files if r["path"] == "vllm_lt/validation/official_cached.py")
        cached = {
            **deps["official"],
            "use_cache": True,
            "dtype": "torch.float32",
            "logits_to_keep": 1,
            "use_weighted_exit": False,
            "cache_slots": 96,
            "cache_class": "CompatibleUniversalTransformerCache",
            "cache_shim_sha256": shim["sha256"],
            "cache_update": "unchanged published append at depth * layers + layer",
        }
        return {
            "source": {"root": str(root), "commit": "a" * 40, "status": "", "files": files},
            "imports": {k: k.replace(".", "/") + ".py" for k in schema.IMPORT_MODULES},
            "dependencies": deepcopy(deps),
            "cached_official": deepcopy(cached),
        }

    monkeypatch.setattr(schema, "source_probe", probe)
    monkeypatch.setattr(schema, "affinity_snapshot", lambda: deepcopy(AFFINITY))
    contract = tmp_path / "contract.json"
    write_json(contract, fixture)
    return schema.make_plan(
        contract,
        model,
        gpu_ids=[0],
        gpu_uuid="GPU-006eb78c-23cb-f37d-eb7c-0ccb578b8f11",
        affinity=deepcopy(AFFINITY),
    )


def test_frozen_eight_runs_and_cpu_verification(external_plan):
    schema.verify_plan(external_plan)
    rows = external_plan["execution_order"]
    assert [r["run_id"] for r in rows] == [
        "N-feas",
        "O-feas",
        "N-warm",
        "O-warm",
        "N1",
        "O1",
        "O2",
        "N2",
    ]
    assert Counter(r["phase"] for r in rows) == {"feasibility": 2, "warmup": 2, "measured": 4}
    assert [r["pair_id"] for r in rows[4:]] == ["pair-1", "pair-1", "pair-2", "pair-2"]
    assert external_plan["cached_official"]["use_cache"] is True
    assert external_plan["dependencies"]["official"]["use_cache"] is False
    resources = external_plan["resource_estimates"]
    assert resources["native_pool_bytes"] == 6 * 1024**3
    assert resources["native_required_pages"] == 48
    assert resources["official_final_cache_bytes"] == 300417024
    assert resources["artifact_bytes_upper_bound"] == 32 * 1024**2
    assert len(external_plan["contract"]["production_files"]) == 22


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: p["execution_order"].reverse(),
        lambda p: p["execution_order"][0].update(max_steps=381),
        lambda p: p["execution_order"][4].update(implementation_id="official"),
        lambda p: p["contract"]["limits"].update(executions=9),
        lambda p: p["contract"]["controls"].update(gpu_ids=[True]),
        lambda p: p["contract"]["controls"].update(gpu_ids=None),
        lambda p: p["contract"]["controls"].update(gpu_uuid=None),
        lambda p: p["contract"]["controls"].update(gpu_uuid="GPU-0"),
        lambda p: p["contract"]["engine"]["sampling"].update(ignore_eos=False),
        lambda p: p["contract"]["engine"]["sampling"].update(min_loops=2),
        lambda p: p["contract"]["engine"].update(persistent_decode=True),
        lambda p: p["contract"]["official"].update(cache_slots=24),
        lambda p: p["cached_official"].update(use_cache=False),
        lambda p: p["cached_official"].update(cache_shim_sha256="b" * 64),
        lambda p: p["source"].update(status=" M changed.py"),
        lambda p: p["source"]["files"][0].update(sha256="b" * 64),
        lambda p: p["imports"].update({schema.IMPORT_MODULES[0]: "../outside.py"}),
        lambda p: p["model_config"].update(extra=True),
        lambda p: p["model_config"].update(eos_token_id=1),
        lambda p: p["workload"]["requests"][0]["prompt_token_ids"].__setitem__(0, 12),
        lambda p: p["resource_estimates"].update(official_final_cache_bytes=0),
        lambda p: p.update(unrecognized=True),
        lambda p: p["dependencies"]["official"].update(optional_kernels_present=True),
    ],
)
def test_rehashed_policy_or_identity_changes_are_rejected(external_plan, mutate):
    plan = deepcopy(external_plan)
    mutate(plan)
    rehash(plan)
    with pytest.raises((ValueError, TypeError)):
        schema.validate_plan(plan)


@pytest.mark.parametrize("kind", ["model", "input", "source", "affinity", "environment"])
def test_execution_rehashes_files_and_environment(external_plan, monkeypatch, kind):
    if kind == "model":
        (Path(external_plan["model_path"]) / "model.safetensors").write_bytes(b"new")
    elif kind == "input":
        Path(external_plan["inputs"]["contract"]["file"]["path"]).write_text("{}")
    elif kind == "source":
        (Path(external_plan["source"]["root"]) / "vllm_lt/validation/official.py").write_text("new")
    elif kind == "affinity":
        altered = deepcopy(AFFINITY)
        altered["numactl_show"]["membind"] = "0"
        monkeypatch.setattr(schema, "affinity_snapshot", lambda: altered)
    else:
        monkeypatch.setenv("OMP_NUM_THREADS", "99")
    with pytest.raises(ValueError):
        schema.verify_plan(external_plan)


def test_embedded_file_mismatch_is_not_cured_by_rehash(external_plan):
    plan = deepcopy(external_plan)
    # Hypothesis text is freeform but must still match the source input bytes.
    plan["inputs"]["contract"]["contents"]["hypothesis"] = "Changed embedded hypothesis"
    plan["contract"]["hypothesis"] = "Changed embedded hypothesis"
    rehash(plan)
    schema.validate_plan(plan)
    with pytest.raises(ValueError, match="embedded input"):
        schema.verify_plan(plan)


def test_reader_rejects_duplicate_and_nonfinite_json(tmp_path):
    for data in ('{"a":1,"a":2}', '{"a":NaN}', '{"a":1e999}'):
        file = tmp_path / "invalid.json"
        file.write_text(data)
        with pytest.raises(ValueError):
            read_json(file)


@pytest.mark.parametrize(
    "value", [None, 2, "GPU-0", "GPU-GPU-cbf66259-f4ab-0ede-1811-82037dde5924"]
)
def test_uuid_format_rejects_malformed_identity(value):
    with pytest.raises(ValueError, match="UUID"):
        schema.canonical_gpu_uuid(value)
