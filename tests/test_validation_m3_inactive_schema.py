"""CPU-only frozen source/metadata/65-row plan checks reuse byte-only model fixtures."""

from copy import deepcopy
from pathlib import Path

import pytest
import test_benchmark_ab_schema as fixtures

from vllm_lt.benchmarks import ab_schema
from vllm_lt.benchmarks.schema import read_json, write_json
from vllm_lt.validation import m3_inactive_schema as schema

ab_plan = fixtures.ab_plan


@pytest.fixture
def m3_plan(ab_plan, monkeypatch, tmp_path):
    monkeypatch.setattr(schema, "probe_checkout", ab_schema.probe_checkout)
    monkeypatch.setattr(schema, "affinity_snapshot", lambda: deepcopy(fixtures.AFFINITY))
    path = tmp_path / "m3-contract.json"
    write_json(
        path, read_json(fixtures.ROOT / "benchmarks/fixtures/ouro-m3-inactive-contract.json")
    )
    return schema.make_plan(
        baseline_root=ab_plan["implementations"]["A"]["root"],
        candidate_root=ab_plan["implementations"]["B"]["root"],
        contract_path=path,
        model_path=ab_plan["model_path"],
        gpu_ids=[7],
        affinity=deepcopy(fixtures.AFFINITY),
    )


def rehash(plan):
    plan["plan_sha256"] = schema._digest({k: v for k, v in plan.items() if k != "plan_sha256"})


def test_cpu_plan_freezes_65_model_kernel_rows_without_profiler_or_checker(m3_plan):
    schema.verify_plan(m3_plan)
    rows = m3_plan["execution_order"]
    assert len(rows) == 65 and len({row["execution_id"] for row in rows}) == 65
    assert len(m3_plan["workers"]) == 2
    assert [len(row["execution_ids"]) for row in m3_plan["workers"]] == [35, 30]
    assert sum(row["kind"] == "model" for row in rows) == 13
    assert sum(row["kind"] == "kernel" for row in rows) == 52
    for worker in m3_plan["workers"]:
        selected = [row for row in rows if row["worker_id"] == worker["worker_id"]]
        assert selected[0]["phase"] == "feasibility"
        assert all(row["kind"] == "kernel" for row in selected[1:27])
        assert all(row["kind"] == "model" for row in selected[27:])
    assert m3_plan["contract"]["limits"]["profile_total_bytes_max"] == 0
    assert m3_plan["kernels"]["checker"]["execution_count"] == 0
    assert schema.loading_view(m3_plan)["workload_stats"]["model"]["pool_bytes"] == 1006632960


@pytest.mark.parametrize(
    "change",
    [
        "null_gpu",
        "bool_gpu",
        "extra_case",
        "checker",
        "kernel_layout",
        "precision",
        "affinity",
        "source",
        "unknown",
    ],
)
def test_invalid_rehashed_plan_is_rejected(m3_plan, change):
    plan = deepcopy(m3_plan)
    if change == "null_gpu":
        plan["contract"]["controls"]["gpu_ids"] = None
    elif change == "bool_gpu":
        plan["contract"]["controls"]["gpu_ids"] = [True]
    elif change == "extra_case":
        plan["execution_order"].append(deepcopy(plan["execution_order"][-1]))
    elif change == "checker":
        plan["kernels"]["checker"]["execution_count"] = 13
    elif change == "kernel_layout":
        plan["kernels"]["layouts"][-1]["table_width"] = 64
    elif change == "precision":
        plan["contract"]["controls"]["dtype"] = "bfloat16"
    elif change == "affinity":
        plan["contract"]["controls"]["affinity"]["numactl_show"]["policy"] = "default"
    elif change == "source":
        plan["implementations"]["B"]["source"]["status"] = " M extra.py"
    else:
        plan["unrecognized"] = True
    rehash(plan)
    with pytest.raises((ValueError, TypeError)):
        schema.validate_plan(plan)


def test_embedded_input_drift_rejected_against_original_file(m3_plan):
    plan = deepcopy(m3_plan)
    # Descriptive provenance may change without altering frozen fixture tensor IDs,
    # but the executed embedded contract must still match its source file bytes.
    plan["inputs"]["suite"]["contents"]["provenance"]["prompt_construction"] += " changed"
    plan["numerical"] = schema.build_model_plan(
        plan["inputs"]["suite"]["contents"],
        plan["inputs"]["contract"]["contents"],
        plan["model_config"],
    )
    rehash(plan)
    schema.validate_plan(plan)
    with pytest.raises(ValueError, match="embedded input"):
        schema.verify_plan(plan)


def test_checkpoint_byte_drift_rejected(m3_plan):
    (Path(m3_plan["model_path"]) / "model.safetensors").write_bytes(b"changed")
    with pytest.raises(ValueError, match="checkpoint file"):
        schema.verify_plan(m3_plan)
