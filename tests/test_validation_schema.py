"""The resolved Q1 workload and byte contract requires neither CUDA nor real tensors."""

import hashlib
from collections import Counter
from copy import deepcopy
from pathlib import Path

import pytest
import torch

from vllm_lt.models.config import OuroConfig
from vllm_lt.validation import common, schema

FIXTURES = Path(__file__).resolve().parents[1] / "benchmarks" / "fixtures"


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("CPU schema probe must not discover CUDA or load checkpoint tensors")

    for name in ("is_available", "device_count", "current_device", "_lazy_init"):
        monkeypatch.setattr(torch.cuda, name, forbidden)
    monkeypatch.setattr(torch, "load", forbidden)
    import safetensors.torch

    monkeypatch.setattr(safetensors.torch, "load_file", forbidden)
    source = tmp_path / "source.py"
    source.write_text("# frozen validation source\n")
    monkeypatch.setattr(
        schema,
        "source_manifest",
        lambda: {
            "root": str(tmp_path),
            "commit": "a" * 40,
            "status": "",
            "files": [common.make_file_record(source, relative_to=tmp_path)],
        },
    )
    model = tmp_path / "model"
    model.mkdir()
    common.write_json(model / "config.json", OuroConfig().to_dict())
    (model / "tokenizer.json").write_bytes(b'{"fixture": "metadata-only tokenizer"}\n')
    (model / "tokenizer_config.json").write_bytes(b"{}\n")
    (model / "model.safetensors").write_bytes(b"not tensors; the probe hashes bytes only")
    official = tmp_path / "official"
    official.mkdir()
    expected = {}
    for name in schema.OFFICIAL_HASHES:
        content = f"# reviewed fixture source {name}\n".encode()
        (official / name).write_bytes(content)
        expected[name] = hashlib.sha256(content).hexdigest()
    monkeypatch.setattr(schema, "OFFICIAL_HASHES", expected)
    dependency_record = {
        "python": "frozen CPU test Python",
        "torch": "test-version",
        "torch_cuda_build": None,
        "distributions": [{"name": "torch", "version": "test-version"}],
        "official": {
            "model_id": schema.OURO_MODEL_ID,
            "revision": schema.OURO_REVISION,
            "source_sha256": expected,
            "dependencies": {
                "transformers": "4.55.0",
                "torch": "test-version",
                "huggingface-hub": "test",
                "tokenizers": "test",
                "safetensors": "test",
                "kernels": None,
            },
            "required_transformers": "4.55.0",
            "optional_kernels_present": False,
            "attention_implementation": "eager",
            "total_ut_steps": 4,
            "exit_at_step": 3,
            "use_cache": False,
            "weight_storage": "shared immutable parameter mapping",
        },
    }
    monkeypatch.setattr(schema, "dependency_manifest", lambda: deepcopy(dependency_record))
    suite = schema.load_suite(FIXTURES / "ouro-q1.json")
    tokenizer = common.make_file_record(model / "tokenizer.json")
    for key in ("sha256", "size_bytes"):
        suite["provenance"]["tokenizer"][key] = tokenizer[key]
    suite_path, contract_path = tmp_path / "suite.json", tmp_path / "contract.json"
    common.write_json(suite_path, suite)
    common.write_json(contract_path, schema.load_contract(FIXTURES / "ouro-q1-contract.json"))
    return suite_path, contract_path, model, official, source


def plan_for(prepared):
    return schema.make_plan(*prepared[:4], gpu_ids=[0])


def test_exact_case_table_references_and_resource_bounds_without_device(prepared):
    plan = plan_for(prepared)
    schema.validate_plan_integrity(plan)
    schema.verify_plan(plan)
    rows = plan["execution_order"]
    assert len(rows) == len({row["case_id"] for row in rows}) == 168
    assert Counter(row["family"] for row in rows) == {
        "feasibility": 10,
        "main": 112,
        "live_gate": 28,
        "official": 8,
        "original": 10,
    }
    assert all(row["phase"] == "feasibility" for row in rows[:10])
    assert not any(row["phase"] == "feasibility" for row in rows[10:])
    indices = {row["case_id"]: index for index, row in enumerate(rows)}
    assert len(plan["comparison_order"]) == 210
    for comparison in plan["comparison_order"]:
        assert indices[comparison["reference_case_id"]] < indices[comparison["candidate_case_id"]]
        assert comparison["expected_prediction_points"] == (
            20 if comparison["family"] == "original" else 9
        )
    original = [row for row in rows if row["family"] == "original"]
    assert all(row["block_size"] == 8 and row["num_blocks"] == 32 for row in original)
    assert len([row for row in original if row["implementation"] == "legacy_packed"]) == 4
    resources = plan["resource_estimates"]
    assert resources["parameter_count"] == 1434652673
    assert resources["main_native_prediction_comparisons"] == 1152
    assert resources["comparison_boundary_records_upper_bound"] == 886504
    assert resources["reference_spool_payload_bytes_upper_bound"] == 7236407304
    assert resources["largest_native_group_pages"] == 132
    assert resources["native_pool_bytes"]["float32"] == 960 * 1024**2
    assert resources["largest_oracle_cache_bytes"]["float32"] == 396 * 1024**2
    assert resources["kv_snapshot_pair_bytes_fp32"] == 6 * 1024**2
    assert resources["largest_reference_group_payload_bytes"] < 2 * 1024**3
    assert resources["reference_spool_payload_bytes_upper_bound"] < 8 * 1024**3
    assert plan["contract"]["limits"]["total_disk_bytes"] == 12 * 1024**3
    assert resources["planned_artifact_bytes_with_auxiliary_allowance"] == 10601674760
    fixed = resources["fixture_stats"]["float32"]["Q1-L256-F0"]
    assert fixed["capacity"] == 264 and fixed["prediction_points"] == 9
    assert fixed["selected_boundary_records"] == 5265
    assert fixed["kv_comparison_records"] == 132
    assert fixed["selected_boundary_bytes"] == 44531856
    assert (
        resources["fixture_stats"]["bfloat16"]["Q1-L256-F0"]["selected_boundary_bytes"] == 22265928
    )


def test_history_vectors_are_heterogeneous_and_feasibility_disjoint():
    suite = schema.load_suite(FIXTURES / "ouro-q1.json")
    fixtures = {row["fixture_id"]: row for row in suite["fixtures"]}
    for group in suite["groups"][2:]:
        histories = [fixtures[key]["forced_exit_depths"] for key in group["fixture_ids"]]
        assert {depths[1] for depths in histories} == {2, 3, 4}
        assert all(depths[0] == 4 and sum(depths) == 28 for depths in histories)
        assert all((2, 4) in list(zip(depths[1:], depths[2:])) for depths in histories)
    histories = [
        tuple(row["prompt_token_ids"]) for row in suite["fixtures"] + suite["feasibility_fixtures"]
    ]
    assert len(set(histories)) == 18
    for fixture in suite["fixtures"]:
        assert len(fixture["continuation_input_ids"]) == 8
        assert len(fixture["forced_exit_depths"]) == 9
    assert suite["original_reproduction"]["prompt_token_ids"] == schema.ORIGINAL_PROMPTS


@pytest.mark.parametrize("gpu_ids", [None, [], [0, 1], [False], [-1], [1.5]])
def test_gpu_assignment_must_be_explicit_and_single_without_discovery(prepared, gpu_ids):
    with pytest.raises(ValueError, match="GPU|gpu_ids|integer"):
        schema.make_plan(*prepared[:4], gpu_ids=gpu_ids)


def test_gpu_override_conflicts_are_rejected(prepared):
    contract = schema.load_contract(prepared[1])
    contract["controls"]["gpu_ids"] = [3]
    common.write_json(prepared[1], contract)
    with pytest.raises(ValueError, match="conflict"):
        schema.make_plan(*prepared[:4], gpu_ids=[0])


@pytest.mark.parametrize("value", [True, -1, 1.5, 49152])
def test_invalid_tokens_rejected_before_planning(prepared, value):
    suite = common.read_json(prepared[0])
    suite["fixtures"][0]["prompt_token_ids"][0] = value
    common.write_json(prepared[0], suite)
    with pytest.raises(ValueError, match="integer|vocabulary"):
        schema.load_suite(prepared[0])


@pytest.mark.parametrize(
    "change",
    [
        "unknown",
        "depth_zero",
        "eight_predictions",
        "nine_inputs",
        "duplicate_id",
        "group",
        "homogeneous",
        "original_float_token",
        "fixture_hash",
    ],
)
def test_malformed_histories_and_group_assignments_rejected(prepared, change):
    suite = common.read_json(prepared[0])
    if change == "unknown":
        suite["fixtures"][0]["expected_top1"] = 0
    elif change == "depth_zero":
        suite["fixtures"][0]["forced_exit_depths"][0] = 2
    elif change == "eight_predictions":
        suite["fixtures"][0]["forced_exit_depths"].pop()
    elif change == "nine_inputs":
        suite["fixtures"][0]["continuation_input_ids"].append(1)
    elif change == "duplicate_id":
        suite["fixtures"][1]["fixture_id"] = suite["fixtures"][0]["fixture_id"]
    elif change == "group":
        suite["groups"][0]["fixture_ids"][0] = "Q1-L16-F1"
    elif change == "homogeneous":
        suite["fixtures"][6]["forced_exit_depths"] = suite["fixtures"][2]["forced_exit_depths"]
    elif change == "original_float_token":
        suite["original_reproduction"]["prompt_token_ids"][0][0] = 504.0
    else:
        suite["fixtures_sha256"] = "0" * 64
    common.write_json(prepared[0], suite)
    with pytest.raises(ValueError):
        schema.load_suite(prepared[0])


@pytest.mark.parametrize(
    "change",
    ["logit_tolerance", "diagnostic_gate", "budget", "sampling_bool", "unknown", "original_top1"],
)
def test_unapproved_policy_or_budget_changes_rejected(prepared, change):
    contract = common.read_json(prepared[1])
    if change == "logit_tolerance":
        contract["comparison_policy"]["logits"]["bfloat16"]["atol"] = 0.5
    elif change == "diagnostic_gate":
        contract["comparison_policy"]["diagnostic_only"].remove("populated_kv")
    elif change == "budget":
        contract["limits"]["implementation_executions"] = 169
    elif change == "sampling_bool":
        contract["engine"]["sampling"]["seed"] = False
    elif change == "unknown":
        contract["diagnostics"]["rerun_on_failure"] = True
    else:
        contract["comparison_policy"]["original_top1_rule"] = "every-loop"
    common.write_json(prepared[1], contract)
    with pytest.raises(ValueError):
        plan_for(prepared)


@pytest.mark.parametrize(
    "content", ['{"x": NaN}', '{"x": Infinity}', '{"x": 1e999}', '{"x": 1, "x": 2}']
)
def test_nonfinite_and_duplicate_json_fields_are_rejected(tmp_path, content):
    path = tmp_path / "bad.json"
    path.write_text(content)
    with pytest.raises(ValueError, match="nonfinite|duplicate"):
        common.read_json(path)


@pytest.mark.parametrize("changed", ["source", "weights", "official", "tokenizer", "contract"])
def test_verification_rehashes_every_execution_dependency(prepared, changed):
    plan = plan_for(prepared)
    if changed == "source":
        prepared[4].write_text("# changed source\n")
    elif changed == "weights":
        (prepared[2] / "model.safetensors").write_bytes(b"changed model")
    elif changed == "official":
        (prepared[3] / "modeling_ouro.py").write_text("# changed official equations\n")
    elif changed == "tokenizer":
        (prepared[2] / "tokenizer.json").write_text("{}\n")
    else:
        contract = common.read_json(prepared[1])
        contract["hypothesis"] = "An unrecorded changed hypothesis."
        common.write_json(prepared[1], contract)
    with pytest.raises(ValueError, match="changed|mismatch|does not match"):
        schema.verify_plan(plan)


def test_offline_validation_rejects_rehashed_resolved_case_corruption(prepared, monkeypatch):
    plan = plan_for(prepared)
    monkeypatch.setattr(
        schema, "make_plan", lambda *a, **k: pytest.fail("offline cannot rehash files")
    )
    schema.validate_plan_integrity(deepcopy(plan))
    plan["execution_order"][0]["max_tokens"] = 8
    plan["plan_sha256"] = common.digest(
        {key: value for key, value in plan.items() if key != "plan_sha256"}
    )
    with pytest.raises(ValueError, match="execution_order"):
        schema.validate_plan_integrity(plan)


def test_original_record_provenance_checked_even_after_fixture_rehash(prepared):
    suite = common.read_json(prepared[0])
    suite["original_reproduction"]["sources"][0]["sha256"] = "0" * 64
    suite["fixtures_sha256"] = common.digest(
        {key: suite[key] for key in schema.FIXTURE_HASH_FIELDS}
    )
    common.write_json(prepared[0], suite)
    with pytest.raises(ValueError, match="original source content provenance"):
        plan_for(prepared)


def test_dependency_drift_invalidates_frozen_plan(prepared, monkeypatch):
    plan = plan_for(prepared)
    changed = deepcopy(plan["dependencies"])
    changed["distributions"][0]["version"] = "unplanned-version"
    monkeypatch.setattr(schema, "dependency_manifest", lambda: changed)
    with pytest.raises(ValueError, match="dependencies"):
        schema.verify_plan(plan)


def test_installed_dependency_metadata_probe_never_discovers_cuda(monkeypatch):
    for name in ("is_available", "device_count", "current_device", "_lazy_init"):
        monkeypatch.setattr(torch.cuda, name, lambda *a, **k: pytest.fail("no CUDA discovery"))
    dependencies = schema.dependency_manifest()
    schema.validate_dependencies(dependencies)
    assert dependencies["torch"] == str(torch.__version__)
    assert dependencies["distributions"]
