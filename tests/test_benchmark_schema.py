"""Plan contracts are checked with tiny files, without weights or device access."""

from collections import Counter
from copy import deepcopy
from pathlib import Path

import pytest
import torch

from vllm_lt.benchmarks import schema
from vllm_lt.models.config import OuroConfig

FIXTURES = Path(__file__).resolve().parents[1] / "benchmarks" / "fixtures"


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    def no_device(*args, **kwargs):
        raise AssertionError("CPU planning must not query or initialize CUDA")

    for name in ("is_available", "current_device", "device_count", "_lazy_init"):
        monkeypatch.setattr(torch.cuda, name, no_device)
    source_file = tmp_path / "source.py"
    source_file.write_text("# frozen source\n")
    monkeypatch.setattr(
        schema,
        "_source_manifest",
        lambda: {
            "root": str(tmp_path),
            "commit": "a" * 40,
            "status": "",
            "files": [schema._file_record(source_file, relative_to=tmp_path)],
        },
    )
    model_path = tmp_path / "model"
    model_path.mkdir()
    schema.write_json(model_path / "config.json", OuroConfig().to_dict())
    (model_path / "tokenizer.json").write_text('{"test": "tokenizer content only"}\n')
    (model_path / "tokenizer_config.json").write_text("{}\n")
    # Probe hashes bytes; it must never attempt to load this file as model tensors.
    (model_path / "model.safetensors").write_bytes(b"not real model weights")
    suite = schema.load_suite(FIXTURES / "ouro-m1.json")
    tokenizer = schema._file_record(model_path / "tokenizer.json")
    for name in ("size_bytes", "sha256"):
        suite["provenance"]["tokenizer"][name] = tokenizer[name]
    suite_path, contract_path = tmp_path / "suite.json", tmp_path / "contract.json"
    schema.write_json(suite_path, suite)
    schema.write_json(contract_path, schema.read_json(FIXTURES / "ouro-m1-contract.json"))
    return suite_path, contract_path, model_path, source_file


def test_exact_plan_budget_pair_order_and_capacity_without_cuda(prepared):
    plan = schema.make_plan(*prepared[:3])
    schema.verify_plan(plan)
    schema.validate_plan_integrity(plan)
    order = plan["execution_order"]
    assert Counter(run["phase"] for run in order) == {
        "feasibility": 7,
        "warmup": 7,
        "measured": 14,
        "profile": 2,
    }
    assert len({run["cell_id"] for run in order}) == 7
    assert len({run["run_id"] for run in order}) == 30
    for workload in ("W4", "W5"):
        paired = [
            run for run in order if run["workload_id"] == workload and run["phase"] == "measured"
        ]
        assert [(run["mode"], run["repetition"]) for run in paired] == [
            ("refill", 1),
            ("no_refill", 1),
            ("no_refill", 2),
            ("refill", 2),
        ]
        assert [run["pair_id"] for run in paired] == [
            f"{workload}-pair-1",
            f"{workload}-pair-1",
            f"{workload}-pair-2",
            f"{workload}-pair-2",
        ]
        assert len({run["controls_sha256"] for run in paired}) == 1
        assert len({run["workload_sha256"] for run in paired}) == 1
    profiles = [run for run in order if run["phase"] == "profile"]
    assert [(run["workload_id"], run["mode"]) for run in profiles] == [
        ("W1", "refill"),
        ("W4", "refill"),
    ]
    stats = plan["workload_stats"]
    assert stats["W1"]["pool_bytes"] == 6 * 1024**3
    assert stats["W1"]["reserved_pages"] == 48
    assert stats["W2"]["reserved_pages"] == 384
    assert stats["W3"]["reserved_pages"] == 136
    assert stats["W4"]["reserved_pages"] == 288
    assert stats["W1"]["max_steps"] == 507
    assert stats["W4"]["max_steps"] == 3032
    assert stats["W1"]["decode_token_loops"] == 252
    assert stats["W4"]["decode_token_loops"] < stats["W5"]["decode_token_loops"]


@pytest.mark.parametrize("value", [True, -1, 1.5, 49152])
def test_invalid_token_ids_rejected(prepared, value):
    suite_path = prepared[0]
    suite = schema.read_json(suite_path)
    suite["workloads"][0]["requests"][0]["prompt_token_ids"][0] = value
    schema.write_json(suite_path, suite)
    with pytest.raises(ValueError, match="integer|vocabulary"):
        schema.load_suite(suite_path)


@pytest.mark.parametrize(
    "corruption,match",
    [
        ("unknown", "unknown fields"),
        ("duplicate", "duplicate request IDs"),
        ("arrival", "arrival_offset_ns"),
        ("first_depth", "depth four"),
        ("trace_length", "exactly 32"),
        ("fractional_depth", "integers 2 through 4"),
        ("hash", "workloads_sha256"),
        ("workload_type", "ordered workloads"),
    ],
)
def test_invalid_fixtures_rejected(prepared, corruption, match):
    suite_path = prepared[0]
    suite = schema.read_json(suite_path)
    first_request = suite["workloads"][0]["requests"][0]
    trace = suite["workloads"][3]["replay"]["request-0"]
    if corruption == "unknown":
        first_request["stream"] = True
    elif corruption == "duplicate":
        suite["workloads"][1]["requests"][1]["request_id"] = "request-0"
    elif corruption == "arrival":
        first_request["arrival_offset_ns"] = False
    elif corruption == "first_depth":
        trace["exit_depths"][0] = 2
    elif corruption == "trace_length":
        trace["output_token_ids"].pop()
    elif corruption == "fractional_depth":
        trace["exit_depths"][1] = 2.0
    elif corruption == "hash":
        suite["workloads_sha256"] = "0" * 64
    elif corruption == "workload_type":
        suite["workloads"][0] = None
    schema.write_json(suite_path, suite)
    with pytest.raises(ValueError, match=match):
        schema.load_suite(suite_path)


@pytest.mark.parametrize(
    "corruption", ["unknown", "bool_integer", "bf16", "extra_sampling", "budget"]
)
def test_incompatible_contract_rejected(prepared, corruption):
    contract_path = prepared[1]
    contract = schema.read_json(contract_path)
    if corruption == "unknown":
        contract["arithmetic"]["extra_flag"] = True
    elif corruption == "bool_integer":
        contract["engine"]["sampling"]["seed"] = False
    elif corruption == "bf16":
        contract["engine"]["dtype"] = "bfloat16"
    elif corruption == "extra_sampling":
        contract["engine"]["sampling"]["max_tokens"] = 64
    elif corruption == "budget":
        contract["limits"]["total_timeout_s"] = 7201
    schema.write_json(contract_path, contract)
    with pytest.raises(ValueError):
        schema.make_plan(*prepared[:3])


@pytest.mark.parametrize(
    "content", ['{"x": NaN}', '{"x": Infinity}', '{"x": 1e999}', '{"x": 1, "x": 2}']
)
def test_nonfinite_and_duplicate_json_rejected(tmp_path, content):
    path = tmp_path / "bad.json"
    path.write_text(content)
    with pytest.raises(ValueError, match="nonfinite|duplicate"):
        schema.read_json(path)


@pytest.mark.parametrize("changed", ["source", "weight", "contract", "plan"])
def test_plan_verification_detects_changes(prepared, changed):
    plan = schema.make_plan(*prepared[:3])
    if changed == "source":
        prepared[3].write_text("# changed source\n")
    elif changed == "weight":
        (prepared[2] / "model.safetensors").write_bytes(b"changed weights")
    elif changed == "contract":
        contract = schema.read_json(prepared[1])
        contract["hypothesis"] = "A changed hypothesis."
        schema.write_json(prepared[1], contract)
    else:
        plan["execution_order"][0]["mode"] = "no_refill"
    with pytest.raises(ValueError, match="changed|hash mismatch"):
        schema.verify_plan(plan)


def test_offline_integrity_requires_no_source_or_weights(prepared, monkeypatch):
    plan = schema.make_plan(*prepared[:3])

    def forbidden(*args, **kwargs):
        pytest.fail("offline integrity must not read model/source files")

    monkeypatch.setattr(schema, "make_plan", forbidden)
    schema.validate_plan_integrity(deepcopy(plan))


def test_config_capacity_and_tokenizer_mismatch_rejected(prepared):
    config_path = prepared[2] / "config.json"
    config = schema.read_json(config_path)
    config["max_position_embeddings"] = 500
    schema.write_json(config_path, config)
    with pytest.raises(ValueError, match="context or KV capacity"):
        schema.make_plan(*prepared[:3])
    schema.write_json(config_path, OuroConfig().to_dict())
    (prepared[2] / "tokenizer.json").write_text("{}\n")
    with pytest.raises(ValueError, match="tokenizer does not match"):
        schema.make_plan(*prepared[:3])
