"""Offline Q2 reconstruction rejects fabricated clocks, cache histories and ACKs."""

import hashlib
import math
import shutil

import pytest
import test_q2_external_schema as schema_tests

from vllm_lt.benchmarks import q2_external_report as report
from vllm_lt.benchmarks.observe import RunCollector
from vllm_lt.benchmarks.q2_external_driver import cpu_control_result
from vllm_lt.benchmarks.q2_external_qualification import build_qualification
from vllm_lt.benchmarks.schema import read_json, write_json
from vllm_lt.request import RequestOutput

external_plan = schema_tests.external_plan


def digest(path):
    data = path.read_bytes()
    return {"size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def update_result(root, run_id, mutate):
    path = root / "runs" / run_id
    value = read_json(path / "result.json")
    mutate(value)
    write_json(path / "result.json", value)
    for filename in ("completed.json", "acknowledged.json"):
        record = read_json(path / filename)
        record["result"] = digest(path / "result.json")
        if filename == "acknowledged.json":
            record["completion"] = digest(path / "completed.json")
        write_json(path / filename, record)


@pytest.fixture
def external_run(tmp_path, external_plan):
    plan = external_plan
    root = tmp_path / "run"
    root.mkdir()
    write_json(root / "plan.json", plan)
    controls = plan["contract"]["controls"]
    start = 10**9
    deadline = start + 3600 * 10**9
    pool = plan["resource_estimates"]["native_pool_bytes"]
    environment = {
        "cuda_visible_devices": "0",
        "gpu_uuid": controls["gpu_uuid"],
        "logical_device": "cuda:0",
        "cpu_affinity": controls["affinity"]["cpu_ids"],
        "numa_status": controls["affinity"]["numa_status"],
        "active_affinity": controls["affinity"],
        "cudnn_allow_tf32": False,
        "arithmetic": plan["contract"]["arithmetic"],
        "actual_torch_threads": {"intraop": 1, "interop": 1},
        "runtime_environment": plan["runtime_environment"],
        "account": "test",
        "scheduler": [{"gpu_id": 0, "type": "RUN", "user": "test"}],
    }
    shapes = report._parameter_shapes(plan["model_config"])
    parameters = [
        {
            "name": name,
            "shape": shape,
            "numel": math.prod(shape),
            "dtype": "torch." + plan["contract"]["engine"]["dtype"],
            "device": "cuda:0",
            "native_data_ptr": 1000 + j,
            "official_data_ptr": 1000 + j,
            "native_requires_grad": False,
            "official_requires_grad": False,
        }
        for j, (name, shape) in enumerate(shapes.items())
    ]
    worker = {
        "schema_version": 1,
        "artifact_type": "q2_external_worker",
        "plan_sha256": plan["plan_sha256"],
        "started_ns": start + 20,
        "deadline_ns": deadline,
        "status": "complete",
        "completed_runs": [],
        "failures": [],
        "source_probe": {
            k: plan[k] for k in ("source", "imports", "dependencies", "cached_official")
        },
        "environment": environment,
        "preparation": {
            "started_ns": start + 30,
            "ended_ns": start + 40,
            "shared_weights": {
                "all_storage_equal": True,
                "records": parameters,
                "parameter_count": sum(p["numel"] for p in parameters),
            },
            "native_pool": {
                "num_blocks": 1024,
                "block_size": 16,
                "shape": [1024, 24, 16, 16, 128],
                "bytes": pool,
                "key_data_ptr": 100000,
                "value_data_ptr": 200000,
            },
        },
        "teardown_after_workspace_release": {"allocated_bytes": 0, "reserved_bytes": 0},
    }
    cursor = start + 50
    ids = list(range(1, 65))
    for row in plan["execution_order"]:
        path = root / "runs" / row["run_id"]
        path.mkdir(parents=True)
        began, arrival = cursor, cursor + 10
        native = row["implementation_id"] == "native"
        collector = RunCollector(["W1"], arrival_ns=arrival, max_events=128)
        step = 1 if native else 0
        for j in range(64):
            token_time = arrival + (j + 1) * (10000 if native else 20000)
            output = RequestOutput(
                "W1",
                plan["workload"]["requests"][0]["prompt_token_ids"],
                ids[: j + 1],
                [4] * (j + 1),
                j == 63,
                "length" if j == 63 else None,
            )
            collector.observe_outputs([output], step_id=step, returned_ns=token_time)
            step += 6 if native else 1
        sync = token_time + 5
        values = collector.finish(synchronized_ns=sync)
        marker = {
            "schema_version": 1,
            "run": row,
            "plan_sha256": plan["plan_sha256"],
            "started_ns": began,
            "deadline_ns": min(deadline, began + 600 * 10**9),
        }
        stages = ["prefill", "coda"] + [
            "prelude",
            "recurrent",
            "recurrent",
            "recurrent",
            "recurrent",
            "coda",
        ] * 63
        counts = (
            {
                "steps": 380,
                "stage_counts": {"prefill": 1, "coda": 64, "prelude": 63, "recurrent": 252},
                "dispatches": [
                    {"step_id": j, "stage": s, "num_tokens": 128 if j == 0 else 1}
                    for j, s in enumerate(stages)
                ],
            }
            if native
            else {
                "steps": 64,
                "stage_counts": {"prefill": 1, "cached_decode": 63},
                "official_logits_checked": 64 if row["phase"] == "feasibility" else 0,
            }
        )
        official = None
        if not native:
            summary = {
                "slot_count": 96,
                "max_cache_size": 96,
                "lengths": [191] * 96,
                "key_shapes": [[1, 16, 191, 128]] * 96,
                "value_shapes": [[1, 16, 191, 128]] * 96,
                "dtype": "torch." + plan["contract"]["engine"]["dtype"],
                "device": "cuda:0",
                "distinct_storage": True,
                "all_finite": True if row["phase"] == "feasibility" else None,
            }
            actual = {
                "status": "awaiting_completion",
                "prompt_length": 128,
                "max_outputs": 64,
                "expected_position": 191,
                "output_count": 64,
                "forward_calls": 64,
                "cache_slots": 96,
                "failure": None,
                "final_summary": summary,
                "calls": [
                    {
                        "output_index": j,
                        "input_count": 128 if j == 0 else 1,
                        "position": 0 if j == 0 else 127 + j,
                        "cache_length": 128 + j,
                    }
                    for j in range(64)
                ],
            }
            official = {
                "snapshot": actual,
                "final_summary": summary,
                "calls": [
                    {
                        "output_index": j,
                        "input_count": 128 if j == 0 else 1,
                        "last_input_position": 127 + j,
                        "token_id": ids[j],
                    }
                    for j in range(64)
                ],
            }
        empty = {"native_requests": 0, "native_used_blocks": 0, "official_cache_slots": 0}
        result = {
            **marker,
            "artifact_type": "q2_external_result",
            "status": "complete",
            "arrival_ns": arrival,
            "synchronized_ns": sync,
            "ended_ns": sync + 10,
            "requests": values["requests"],
            "events": values["events"],
            "metrics": values["metrics"],
            "setup_ns": 5,
            "counts": counts,
            "before_request": empty,
            "cleanup": empty,
            "official_cache": official,
            "feasibility": {
                "finite_checks": {"recurrent": 256, "coda": 64} if native else {},
                "passed": True,
            }
            if row["phase"] == "feasibility"
            else None,
            "memory": {
                "before": {"allocated_bytes": pool + 10, "reserved_bytes": pool + 20},
                "after_requests": {"allocated_bytes": pool + 10, "reserved_bytes": pool + 20},
                "peak_allocated_bytes": pool + 30,
                "peak_reserved_bytes": pool + 40,
                "pool_bytes": pool,
            },
            "failures": [],
        }
        if plan["schema_version"] == 2:
            before = {
                "affinity": controls["affinity"]["cpu_ids"],
                "busy_ticks": {"cpu56": 0, "cpu57": 0},
                "process_ticks": 0,
                "runtime_ns": 0,
                "delay_ns": 0,
                "clock_ticks_per_second": 100,
                "time_ns": began + 5,
            }
            after = {**before, "runtime_ns": sync - began, "time_ns": sync + 2}
            result["cpu_control"] = cpu_control_result(before, after)
        write_json(path / "started.json", marker)
        write_json(path / "result.json", result)
        completion = {
            **marker,
            "status": "complete",
            "result": digest(path / "result.json"),
            "completed_ns": sync + 20,
        }
        write_json(path / "completed.json", completion)
        write_json(
            path / "acknowledged.json",
            {
                **completion,
                "completion": digest(path / "completed.json"),
                "acknowledged_ns": sync + 30,
            },
        )
        worker["completed_runs"].append(row["run_id"])
        if row["run_id"] == "O-feas":
            worker["equivalence"] = {
                "passed": True,
                "token_ids": ids,
                "exit_depths": [4] * 64,
                "checked_ns": sync + 15,
            }
        cursor = sync + 100
    worker["ended_ns"] = cursor
    write_json(root / "worker.json", worker)
    manifest = {
        "schema_version": 1,
        "artifact_type": "q2_external_manifest",
        "plan_sha256": plan["plan_sha256"],
        "started_ns": start,
        "deadline_ns": deadline,
        "ended_ns": cursor + 20,
        "completed_runs": list(worker["completed_runs"]),
        "status": "complete",
        "failures": [],
        "launch": {"started_ns": start + 10, "ended_ns": cursor + 10, "returncode": 0},
    }
    write_json(root / "manifest.json", manifest)
    return root


def test_complete_portable_metric_reconstruction(external_run):
    value = report.build_report(external_run)
    assert value["errors"] == []
    assert value["passed"] and value["complete"]
    assert value["counts"] == {
        "acknowledged_valid_runs": 8,
        "valid_measured_runs": 4,
        "validated_generation_runs": 8,
        "retained_failed_completed_runs": 0,
    }
    assert [p["native_over_official"] for p in value["pairs"]] == [2.0, 2.0]
    assert value["runs"][4]["metrics"]["generated_tokens_per_second"] == 100000.0
    assert value["runs"][4]["metrics"]["synchronized_tokens_per_second"] < 100000.0


@pytest.mark.parametrize(
    "mutate",
    [
        lambda r: r["events"][0].update(token_id=9),
        lambda r: r["events"][0].update(step_id=0),
        lambda r: r["metrics"].update(generated_tokens_per_second=999999),
        lambda r: r["requests"][0]["exit_depths"].__setitem__(0, 2),
        lambda r: r["counts"]["dispatches"].pop(),
        lambda r: r["before_request"].update(native_used_blocks=1),
        lambda r: r["cleanup"].update(native_requests=1),
        lambda r: r["memory"].update(pool_bytes=0),
        lambda r: r.update(arrival_ns=r["started_ns"] - 1),
    ],
)
def test_rehashed_case_tampering_is_not_accepted(external_run, mutate):
    update_result(external_run, "N1", mutate)
    value = report.build_report(external_run)
    assert not value["passed"] and value["evidence_status"] == "invalid"


@pytest.mark.parametrize(
    "mutate",
    [
        lambda r: r["official_cache"]["snapshot"].update(forward_calls=63),
        lambda r: r["official_cache"]["snapshot"]["final_summary"].update(slot_count=24),
        lambda r: r["official_cache"]["snapshot"]["final_summary"]["lengths"].__setitem__(40, 192),
        lambda r: r["official_cache"]["snapshot"]["calls"][1].update(position=129),
        lambda r: r["counts"].update(official_logits_checked=0),
    ],
)
def test_cached_full_depth_and_feasibility_proof_required(external_run, mutate):
    update_result(external_run, "O-feas", mutate)
    value = report.build_report(external_run)
    assert not value["passed"] and value["errors"]


@pytest.mark.parametrize("missing", ["started.json", "completed.json", "acknowledged.json"])
def test_missing_marker_cannot_pass(external_run, missing):
    (external_run / "runs" / "N1" / missing).unlink()
    value = report.build_report(external_run)
    assert not value["passed"]


def test_late_acknowledgment_invalidates_case(external_run):
    path = external_run / "runs" / "N1" / "acknowledged.json"
    value = read_json(path)
    value["acknowledged_ns"] = value["deadline_ns"] + 1
    write_json(path, value)
    assert report.build_report(external_run)["errors"]


@pytest.mark.parametrize("field", ["gpu_uuid", "arithmetic", "source", "alias"])
def test_control_and_weight_ownership_mismatch_invalidates(external_run, field):
    path = external_run / "worker.json"
    worker = read_json(path)
    if field == "gpu_uuid":
        worker["environment"][field] = "GPU-elsewhere"
    elif field == "arithmetic":
        worker["environment"][field]["allow_tf32"] = True
    elif field == "source":
        worker["source_probe"][field]["commit"] = "b" * 40
    else:
        worker["preparation"]["shared_weights"]["records"][0]["official_data_ptr"] += 1
    write_json(path, worker)
    assert report.build_report(external_run)["evidence_status"] == "invalid"


def test_torch_bare_uuid_is_same_device_but_different_uuid_is_rejected(external_run):
    path = external_run / "worker.json"
    worker = read_json(path)
    bare = worker["environment"]["gpu_uuid"].removeprefix("GPU-")
    worker["environment"]["gpu_uuid"] = bare
    write_json(path, worker)
    assert report.build_report(external_run)["passed"]
    worker["environment"]["gpu_uuid"] = bare[:-1] + ("0" if bare[-1] != "0" else "1")
    write_json(path, worker)
    assert report.build_report(external_run)["evidence_status"] == "invalid"


def test_stopped_worker_retains_valid_prefix_without_invented_cleanup(external_run):
    path = external_run / "worker.json"
    worker = read_json(path)
    worker["completed_runs"] = worker["completed_runs"][:6]
    shutil.rmtree(external_run / "runs" / "N2")
    worker["status"] = "running"
    worker.pop("ended_ns")
    worker.pop("teardown_after_workspace_release")
    write_json(path, worker)
    path = external_run / "manifest.json"
    manifest = read_json(path)
    manifest["completed_runs"] = worker["completed_runs"]
    manifest["status"] = "failed"
    manifest["failures"] = [{"type": "OSError", "message": "controller stopped"}]
    write_json(path, manifest)
    value = report.build_report(external_run)
    assert value["errors"] == []
    assert value["counts"]["acknowledged_valid_runs"] == 6
    assert value["decision"] == "failed" and value["evidence_status"] == "incomplete"
    assert "final memory cleanup" in value["missing"]


def test_slower_native_and_overlapping_ranges_are_valid_evidence(external_run):
    # A valid slower outcome is not rejected by a hidden speed minimum.
    records = []
    for number in (1, 2):
        for impl, value in (("N", 10.0), ("O", 20.0)):
            records.append(
                {
                    "run_id": f"{impl}{number}",
                    "metrics": {"generated_tokens_per_second": value},
                    "memory": {},
                    "setup_ns": 1,
                }
            )
    pairs, variation = report._pairs(records)
    assert all(p["complete"] for p in pairs)
    assert variation["status"] == "official_faster"
    records[3]["metrics"]["generated_tokens_per_second"] = 9.0
    assert report._pairs(records)[1]["status"] == "overlapping"


def test_actual_measured_output_mismatch_is_failed_not_speed_inconclusive(external_run):
    def change(result):
        result["requests"][0]["token_ids"][0] = 100
        result["events"][0]["token_id"] = 100

    update_result(external_run, "N1", change)
    value = report.build_report(external_run)
    assert value["errors"] == []
    assert value["decision"] == "failed"
    assert any(f["type"] == "EquivalenceFailure" for f in value["failures"])


def test_report_write_keeps_raw_generations_unchanged(external_run):
    raw = external_run / "runs" / "N1" / "result.json"
    original = raw.read_bytes()
    value = report.write_report(external_run)
    assert value["passed"]
    assert raw.read_bytes() == original
    assert (external_run / "q2-external-report.md").exists()


def test_report_regeneration_is_byte_identical(external_run):
    first = report.write_report(external_run)
    payload = (external_run / "q2-external-report.json").read_bytes()
    second = report.write_report(external_run)
    assert first == second
    assert (external_run / "q2-external-report.json").read_bytes() == payload
    assert "q2-external-report.json" not in second["artifacts"]["json_files"]


@pytest.mark.parametrize("external_plan", [2], indirect=True)
def test_bf16_report_validates_actual_dtype_and_one_measured_pair(external_run):
    value = report.build_report(external_run)
    assert value["passed"], value
    assert value["dtype"] == "bfloat16" and value["schema_version"] == 2
    assert value["counts"]["valid_measured_runs"] == 2
    assert len(value["pairs"]) == 1
    assert value["variation"]["status"] == "single_observation"
    update_result(
        external_run,
        "O1",
        lambda r: r["official_cache"]["snapshot"]["final_summary"].update(dtype="torch.float32"),
    )
    assert not report.build_report(external_run)["passed"]


@pytest.mark.parametrize("external_plan", [2], indirect=True)
def test_bf16_cpu_contention_cannot_pass(external_run):
    def contaminate(result):
        control = result["cpu_control"]
        control["after"]["busy_ticks"]["cpu56"] += 100
        result["cpu_control"] = cpu_control_result(control["before"], control["after"])

    update_result(external_run, "N1", contaminate)
    value = report.build_report(external_run)
    assert not value["passed"] and "CPU contention" in value["errors"][0]


@pytest.mark.parametrize("external_plan", [2], indirect=True)
@pytest.mark.parametrize("fault", [None, "contention", "affinity", "clock"])
def test_bf16_preflight_is_independently_audited(external_run, fault):
    def add_gate(result):
        before = {**result["cpu_control"]["before"], "time_ns": result["started_ns"] + 1}
        after = {**before, "time_ns": before["time_ns"] + 1}
        if fault == "contention":
            after["busy_ticks"] = {"cpu56": 100, "cpu57": 0}
        elif fault == "affinity":
            before["affinity"] = after["affinity"] = [99]
        elif fault == "clock":
            after["time_ns"] = result["arrival_ns"] + 1
        result["cpu_preflight"] = cpu_control_result(before, after)

    update_result(external_run, "N1", add_gate)
    value = report.build_report(external_run)
    assert value["passed"] is (fault is None), value
    if fault:
        assert "CPU preflight" in value["errors"][0]


@pytest.mark.parametrize("external_plan", [2], indirect=True)
def test_bf16_cross_backend_tokens_are_diagnostic_but_backend_drift_fails(external_run):
    def change(result):
        result["requests"][0]["token_ids"][0] = 100
        result["events"][0]["token_id"] = 100

    for name in ("N-feas", "N1"):
        update_result(external_run, name, change)
    path = external_run / "worker.json"
    worker = read_json(path)
    worker["equivalence"]["passed"] = False
    write_json(path, worker)
    value = report.build_report(external_run)
    assert value["passed"] and value["equivalence"] == {"complete": True, "passed": False}
    update_result(external_run, "N1", lambda r: r["requests"][0]["token_ids"].__setitem__(1, 101))
    assert not report.build_report(external_run)["passed"]


def test_hidden_extra_execution_beyond_ack_prefix_rejected(external_run):
    for filename in ("worker.json", "manifest.json"):
        path = external_run / filename
        record = read_json(path)
        record.update(completed_runs=["N-feas"], status="failed")
        write_json(path, record)
    value = report.build_report(external_run)
    assert value["errors"]
    assert value["evidence_status"] == "invalid"


def late_failure(root, *, missing_ack=False):
    path = root / "runs" / "N2"
    original = (path / "result.json").read_bytes()
    marker = read_json(path / "started.json")
    if missing_ack:
        (path / "acknowledged.json").unlink()
    occurred = marker["deadline_ns"] + 1
    failure = {
        "run": marker["run"],
        "plan_sha256": marker["plan_sha256"],
        "occurred_ns": occurred,
        "failure": {"type": "TimeoutError", "message": "case export exceeded deadline"},
    }
    write_json(path / "failure.json", failure)
    worker = read_json(root / "worker.json")
    worker.update(status="failed", ended_ns=occurred + 5, failures=[failure["failure"]])
    write_json(root / "worker.json", worker)
    manifest = read_json(root / "manifest.json")
    manifest.update(
        status="failed",
        ended_ns=occurred + 15,
        failures=[{"type": "RuntimeError", "message": "worker failed"}],
    )
    manifest["launch"].update(ended_ns=occurred + 10, returncode=1)
    write_json(root / "manifest.json", manifest)
    return original


@pytest.mark.parametrize("missing_ack", [False, True])
def test_postcompletion_failure_preserves_generation_but_excludes_pair(external_run, missing_ack):
    original = late_failure(external_run, missing_ack=missing_ack)
    value = report.build_report(external_run)
    assert value["errors"] == []
    assert value["decision"] == "failed" and value["evidence_status"] == "incomplete"
    assert value["counts"] == {
        "acknowledged_valid_runs": 7,
        "valid_measured_runs": 3,
        "validated_generation_runs": 8,
        "retained_failed_completed_runs": 1,
    }
    assert value["pairs"][0]["complete"] and not value["pairs"][1]["complete"]
    tail = value["retained_failed_runs"][0]
    assert tail["generation_valid"] and not tail["comparison_eligible"]
    assert (tail["acknowledged_ns"] is None) == missing_ack
    assert (external_run / "runs" / "N2" / "result.json").read_bytes() == original


@pytest.mark.parametrize("field", ["plan", "row", "time", "payload"])
def test_postcompletion_failure_requires_valid_raw_bindings(external_run, field):
    late_failure(external_run)
    path = external_run / "runs" / "N2" / "failure.json"
    failure = read_json(path)
    if field == "plan":
        failure["plan_sha256"] = "b" * 64
    elif field == "row":
        failure["run"]["run_id"] = "N1"
    elif field == "time":
        failure["occurred_ns"] = 1
    else:
        update_result(external_run, "N2", lambda result: result["events"][0].update(token_id=999))
    write_json(path, failure)
    value = report.build_report(external_run)
    assert value["errors"] and value["evidence_status"] == "invalid"
    assert value["counts"]["retained_failed_completed_runs"] == 0


def test_ack_completion_hash_is_independently_verified(external_run):
    path = external_run / "runs" / "N1" / "acknowledged.json"
    ack = read_json(path)
    ack["completion"]["sha256"] = "f" * 64
    write_json(path, ack)
    value = report.build_report(external_run)
    assert value["errors"] and value["evidence_status"] == "invalid"


@pytest.fixture
def qualification_inputs(tmp_path):
    report = tmp_path / "report.json"
    chronology = tmp_path / "chronology.json"
    marker = {"size_bytes": 1, "sha256": "a" * 64}
    base = {
        "schema_version": 1,
        "artifact_type": "q2_external_report",
        "complete": True,
        "evidence_status": "complete",
        "decision": "passed",
        "passed": True,
        "errors": [],
        "missing": [],
        "failures": [],
        "plan_sha256": "b" * 64,
        "equivalence": {"complete": True, "passed": True},
        "runs": [
            {"run_id": name, "token_ids": list(range(64))}
            for name in ("N-feas", "O-feas", "N-warm", "O-warm", "N1", "O1", "O2", "N2")
        ],
        "artifacts": {
            "json_files": {
                f"runs/N2/{name}.json": marker for name in ("started", "result", "completed")
            }
        },
    }
    write_json(report, base)
    binding = digest(report)
    review = {
        "schema_version": 1,
        "artifact_type": "q2_external_manual_timing_disposition",
        "manual_timing_decision": "inconclusive",
        "performance_qualified": False,
        "primary_report_decision": "passed",
        "primary_report_evidence_status": "complete",
        "reason": "CPU preparation overlaps N2 on the same binding.",
        "records": {
            "primary_report": binding,
            "external_N2_start": {**marker, "mtime_ns": 10},
            "quality_plan": {"mtime_ns": 11},
            "quality_probe_log": {"mtime_ns": 12},
            "external_N2_result": {**marker, "mtime_ns": 13},
            "external_N2_completion": marker,
        },
        "shared_affinity": {"cpu_ids": list(range(8)), "numactl_show": {"membind": "0"}},
    }
    write_json(chronology, review)
    return report, chronology, base, review


def test_known_contamination_supersedes_passed_report_without_rewriting_it(
    qualification_inputs, tmp_path
):
    report, chronology, _, _ = qualification_inputs
    before = report.read_bytes(), chronology.read_bytes()
    result = build_qualification(report, chronology, tmp_path / "final.json")
    assert result["generation_equivalence"] == "passed"
    assert result["timing_controls"] == "invalid"
    assert result["performance_qualification"] == result["decision"] == "inconclusive"
    assert result["passed"] is False and result["performance_qualified"] is False
    assert result["supersedes"] == hashlib.sha256(before[0]).hexdigest()
    assert (report.read_bytes(), chronology.read_bytes()) == before
    with pytest.raises(FileExistsError):
        build_qualification(report, chronology, report)


@pytest.mark.parametrize("mutation", ["report_hash", "case_hash", "clock", "status", "affinity"])
def test_unbound_or_noncontaminated_chronology_is_rejected(
    qualification_inputs, tmp_path, mutation
):
    report, chronology, _, review = qualification_inputs
    if mutation == "report_hash":
        review["records"]["primary_report"]["sha256"] = "c" * 64
    elif mutation == "case_hash":
        review["records"]["external_N2_result"]["sha256"] = "c" * 64
    elif mutation == "clock":
        review["records"]["quality_probe_log"]["mtime_ns"] = 14
    elif mutation == "status":
        review["performance_qualified"] = True
    else:
        review["shared_affinity"]["cpu_ids"] = [56]
    write_json(chronology, review)
    output = tmp_path / "final.json"
    with pytest.raises(ValueError):
        build_qualification(report, chronology, output)
    assert not output.exists()


def test_failed_subchecks_cannot_qualify_generation(qualification_inputs, tmp_path):
    report, chronology, base, review = qualification_inputs
    base["errors"] = ["invalid generation record"]
    write_json(report, base)
    review["records"]["primary_report"] = digest(report)
    write_json(chronology, review)
    result = build_qualification(report, chronology, tmp_path / "final.json")
    assert result["generation_equivalence"] == "inconclusive"
    assert result["passed"] is False
