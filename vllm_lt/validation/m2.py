"""The fixed M2 FP32 subset: independent oracle gates plus exact implementation A/B.

This is a 27-case numerical plan, not a shortened Q1 plan. The parent A/B
controller owns source/environment verification, model loading and GPU lifetime.
"""

import math
import time
from copy import deepcopy
from pathlib import Path

from vllm_lt.models.config import OuroConfig

from .schema import (
    _digest,
    _fixture_stats,
    _validate_contract,
    _validate_suite,
    read_json,
    write_json,
)

FIXTURE_IDS = ("Q1-L16-F0", "Q1-L256-F0", "Q1-L64-F2", "Q1-L128-F3")
PRESELECTED = ("Q1-L64-F2", "Q1-L256-F0")


def build_numerical_plan(suite, contract, model_config):
    """Resolve original Q1 inputs into the exact M2 subset without device work."""
    _validate_suite(suite)
    _validate_contract(contract)
    config = OuroConfig.from_dict(model_config)
    if config.total_ut_steps != 4:
        raise ValueError("M2 numerical validation requires four Ouro loops")
    fixtures = {row["fixture_id"]: deepcopy(row) for row in suite["fixtures"]}
    fixtures = {key: fixtures[key] for key in FIXTURE_IDS}
    for fixture in fixtures.values():
        if any(
            token >= config.vocab_size
            for token in fixture["prompt_token_ids"] + fixture["continuation_input_ids"]
        ):
            raise ValueError("fixture token exceeds model vocabulary")
        if len(fixture["prompt_token_ids"]) + 8 > config.max_position_embeddings:
            raise ValueError("fixture exceeds model position capacity")
    adapted = {
        "schema_version": 1,
        "artifact_type": "m2_numerical_contract",
        **{
            key: deepcopy(contract[key])
            for key in ("engine", "arithmetic", "limits", "diagnostics", "comparison_policy")
        },
    }
    adapted["limits"].update(implementation_executions=27, feasibility_fixtures_per_dtype=0)
    adapted["diagnostics"].update(
        preselected_dump_fixture_ids=list(PRESELECTED),
        preselected_dump_dtype="float32",
        preselected_dump_comparison_kind="implementation_exact",
        reference_retention="retain-oracle-and-A-per-execution-through-B-audit",
    )
    order = []
    policy_hash = _digest(adapted["comparison_policy"])

    def add(implementation_id, implementation, backend, schedule, ids, *, live=False):
        family = "live_gate" if live else "main"
        suffix = ids[0] if len(ids) == 1 else "mixed"
        case_id = (
            f"m2-{implementation_id}-{family}-{implementation}-"
            f"{backend or 'dense'}-{schedule}-{suffix}"
        )
        histories = [fixtures[key] for key in ids]
        capacities = {row["fixture_id"]: len(row["prompt_token_ids"]) + 8 for row in histories}
        cache = adapted["engine"]["cache"]
        pages = sum(4 * math.ceil(size / cache["block_size"]) for size in capacities.values())
        if implementation == "native" and pages > cache["num_blocks"]:
            raise ValueError("M2 packed group exceeds reserved KV capacity")
        row = {
            "case_id": case_id,
            "implementation_id": implementation_id,
            "family": family,
            "phase": "validation",
            "dtype": "float32",
            "implementation": implementation,
            "backend": backend,
            "schedule": schedule,
            "fixture_ids": list(ids),
            "group_id": suffix,
            "history_mode": "live_gate" if live else "teacher_forced",
            "exit_policy": {
                "mode": "live"
                if live
                else "forced"
                if any(row["history_policy"] == "forced" for row in histories)
                else "fixed",
                "threshold": 0.7 if live else 1.0,
                "min_loops": 2,
                "max_loops": 4,
            },
            "block_size": cache["block_size"],
            "num_blocks": cache["num_blocks"],
            "chunk_size": adapted["engine"]["scheduler"]["max_num_batched_tokens"],
            "max_tokens": 9,
            "expected_capacity": capacities,
            "max_steps": sum(len(row["prompt_token_ids"]) + 49 for row in histories),
            "fixture_sha256": _digest(histories),
            "policy_sha256": policy_hash,
            "retain_evidence": implementation_id == "A",
            "spool_group": case_id,
        }
        order.append(row)
        return row

    oracles = {}
    for fixture_id in FIXTURE_IDS:
        oracles[("main", fixture_id)] = add("A", "oracle", None, "serial", [fixture_id])
    oracles[("live_gate", FIXTURE_IDS[0])] = add(
        "A", "oracle", None, "serial", [FIXTURE_IDS[0]], live=True
    )
    for implementation_id in ("A", "B"):
        for backend in ("torch", "triton"):
            for fixture_id in FIXTURE_IDS:
                add(implementation_id, "native", backend, "serial", [fixture_id])
        for schedule in ("refill", "no_refill"):
            add(implementation_id, "native", "triton", schedule, FIXTURE_IDS)
        add(implementation_id, "native", "triton", "serial", [FIXTURE_IDS[0]], live=True)
    comparisons = []
    baselines = {}
    for case in order:
        if case["implementation"] != "native":
            continue
        identity = (case["family"], case["backend"], case["schedule"], tuple(case["fixture_ids"]))
        if case["implementation_id"] == "A":
            baselines[identity] = case
        for fixture_id in case["fixture_ids"]:
            refs = [(oracles[(case["family"], fixture_id)], False)]
            if case["implementation_id"] == "B":
                refs.append((baselines[identity], True))
            for reference, exact in refs:
                stats = _fixture_stats(
                    fixtures[fixture_id], "float32", config, live=case["family"] == "live_gate"
                )
                kind = "implementation_exact" if exact else "same_dtype_fidelity"
                comparisons.append(
                    {
                        "comparison_id": f"{case['case_id']}--{fixture_id}--{kind}",
                        "family": case["family"],
                        "dtype": "float32",
                        "reference_case_id": reference["case_id"],
                        "candidate_case_id": case["case_id"],
                        "fixture_id": fixture_id,
                        "comparison_kind": kind,
                        "required": True,
                        "require_exact": exact,
                        "expected_prediction_points": 9,
                        "expected_boundary_records": stats["selected_boundary_records"]
                        + stats["kv_comparison_records"],
                        "counts_are_upper_bounds": stats["counts_are_upper_bounds"],
                        "policy_sha256": policy_hash,
                    }
                )
    retained_bytes = retained_records = max_case_bytes = 0
    for case in order:
        if not case["retain_evidence"]:
            continue
        stats = [
            _fixture_stats(fixtures[key], "float32", config, live=case["family"] == "live_gate")
            for key in case["fixture_ids"]
        ]
        case_bytes = sum(row["reference_spool_payload_bytes"] for row in stats)
        retained_bytes += case_bytes
        retained_records += sum(
            row["selected_boundary_records"] + row["kv_comparison_records"] for row in stats
        )
        max_case_bytes = max(max_case_bytes, case_bytes)
    limits = adapted["limits"]
    if (
        max_case_bytes > limits["group_spool_bytes"]
        or retained_bytes + limits["persisted_dump_bytes"]
        > limits["cumulative_spool_written_bytes"]
    ):
        raise ValueError("M2 retained evidence exceeds the frozen byte limits")
    result = {
        "schema_version": 1,
        "artifact_type": "m2_numerical_plan",
        "source_inputs": {"suite": deepcopy(suite), "contract": deepcopy(contract)},
        "suite": {
            "schema_version": 1,
            "artifact_type": "m2_numerical_suite",
            "fixtures": list(fixtures.values()),
            "source_suite_sha256": _digest(suite),
        },
        "contract": adapted,
        "model_config": config.to_dict(),
        "execution_order": order,
        "comparison_order": comparisons,
        "resource_estimates": {
            "retained_tensor_bytes_upper_bound": retained_bytes,
            "retained_index_records_upper_bound": retained_records,
            "comparison_records_upper_bound": sum(
                row["expected_boundary_records"] for row in comparisons
            ),
            "max_retained_case_bytes": max_case_bytes,
            "cases": len(order),
            "comparisons": len(comparisons),
        },
    }
    result["numerical_plan_sha256"] = _digest(result)
    return result


def validate_numerical_plan(plan):
    """Rebuild the strict subset from its frozen inputs; never access device/files."""
    if not isinstance(plan, dict) or plan.get("artifact_type") != "m2_numerical_plan":
        raise ValueError("expected an M2 numerical plan")
    expected = build_numerical_plan(
        plan["source_inputs"]["suite"], plan["source_inputs"]["contract"], plan["model_config"]
    )
    if _digest(plan) != _digest(expected):
        raise ValueError("M2 numerical plan differs from its exact frozen subset")


def _view(parent_plan):
    numerical = parent_plan["numerical"]
    validate_numerical_plan(numerical)
    return {**numerical, "plan_sha256": parent_plan["plan_sha256"]}


def _save_ledger(folder, ledger):
    temporary = folder / "ledger.tmp.json"
    write_json(temporary, ledger)
    temporary.replace(folder / "ledger.json")


def _restore_budget(folder, view, ledger):
    from .diagnostics import SpoolBudget, TensorSpool

    limits = view["contract"]["limits"]
    budget = SpoolBudget(limits["cumulative_spool_written_bytes"], limits["group_spool_bytes"])
    expected_groups = {}
    for case in view["execution_order"]:
        if case["case_id"] not in ledger["completed_cases"] or not case["retain_evidence"]:
            continue
        size = 0
        for fixture_id in case["fixture_ids"]:
            spool = TensorSpool.open(folder / "spools", case["case_id"], fixture_id)
            size += spool.data_path.stat().st_size
            spool.close()
        expected_groups[case["spool_group"]] = size
    if ledger["diagnostic_dumps"] != {
        "selected_fixture_ids": list(PRESELECTED),
        "written_bytes": 0,
        "fixture_written_bytes": {},
    } or any((folder / "dumps").rglob("*.bin")):
        raise ValueError("a passing A worker cannot have consumed B/A-only dump quota")
    if ledger["spool_bytes_by_group"] != expected_groups:
        raise ValueError("cross-worker spool ledger differs from retained A files")
    for group, size in expected_groups.items():
        budget.claim(group, size)
    if budget.written_bytes != ledger["tensor_written_bytes"]:
        raise ValueError("cross-worker cumulative tensor ledger differs")
    return budget


def run_numerical_rows(model, parent_plan, implementation_id, output_dir, deadline_ns):
    """Run A once, then B once against its retained oracle/A evidence."""
    from .diagnostics import DiagnosticDump, SpoolBudget
    from .runner import execute_case

    if implementation_id not in ("A", "B"):
        raise ValueError("implementation_id must be A or B")
    view = _view(parent_plan)
    if (
        _digest(model.config.to_dict()) != _digest(view["model_config"])
        or str(next(model.parameters()).dtype) != "torch.float32"
    ):
        raise ValueError("loaded model differs from the frozen FP32 numerical controls")
    folder = Path(output_dir) / "numerical"
    limits = view["contract"]["limits"]
    selected = view["contract"]["diagnostics"]["preselected_dump_fixture_ids"]
    planned = [
        case for case in view["execution_order"] if case["implementation_id"] == implementation_id
    ]
    if implementation_id == "A":
        folder.mkdir(exist_ok=False)
        budget = SpoolBudget(limits["cumulative_spool_written_bytes"], limits["group_spool_bytes"])
        ledger = {
            "schema_version": 1,
            "artifact_type": "m2_numerical_ledger",
            "plan_sha256": parent_plan["plan_sha256"],
            "numerical_plan_sha256": view["numerical_plan_sha256"],
            "completed_cases": [],
            "started_cases": [],
            "active_case": None,
            "case_lifetimes": {},
            "workers": {},
            "tensor_written_bytes": 0,
            "spool_bytes_by_group": {},
            "diagnostic_dumps": {
                "selected_fixture_ids": selected,
                "written_bytes": 0,
                "fixture_written_bytes": {},
            },
        }
    else:
        ledger = read_json(folder / "ledger.json")
        if (
            ledger["plan_sha256"] != parent_plan["plan_sha256"]
            or ledger["numerical_plan_sha256"] != view["numerical_plan_sha256"]
            or set(ledger["workers"]) != {"A"}
            or not ledger["workers"]["A"]["passed"]
            or ledger["completed_cases"]
            != [
                case["case_id"]
                for case in view["execution_order"]
                if case["implementation_id"] == "A"
            ]
            or ledger["started_cases"] != ledger["completed_cases"]
            or ledger["active_case"] is not None
            or set(ledger["case_lifetimes"]) != set(ledger["completed_cases"])
        ):
            raise ValueError("B requires one complete, passing A numerical worker")
        budget = _restore_budget(folder, view, ledger)
    dumps = DiagnosticDump(
        folder / "dumps",
        selected,
        budget=budget,
        per_fixture_bytes=limits["dump_bytes_per_fixture"],
        total_bytes=limits["persisted_dump_bytes"],
    )
    result = {
        "implementation_id": implementation_id,
        "complete": False,
        "passed": False,
        "completed_cases": [],
        "errors": [],
    }
    ledger["workers"][implementation_id] = result

    def checkpoint():
        ledger.update(
            tensor_written_bytes=budget.written_bytes,
            spool_bytes_by_group=dict(budget.group_written_bytes),
            diagnostic_dumps={
                "selected_fixture_ids": list(dumps.selected_fixture_ids),
                "written_bytes": dumps.written_bytes,
                "fixture_written_bytes": dict(dumps.fixture_written_bytes),
            },
        )
        _save_ledger(folder, ledger)

    checkpoint()
    try:
        for case in planned:
            started_ns = time.perf_counter_ns()
            case_deadline_ns = min(
                deadline_ns, started_ns + limits["case_timeout_s"] * 1_000_000_000
            )
            remaining = (case_deadline_ns - started_ns) / 1e9
            if remaining <= 0:
                raise TimeoutError("parent numerical deadline exhausted")
            ledger["active_case"] = {
                "case_id": case["case_id"],
                "started_ns": started_ns,
                "deadline_ns": case_deadline_ns,
            }
            ledger["started_cases"].append(case["case_id"])
            checkpoint()
            remaining = (case_deadline_ns - time.perf_counter_ns()) / 1e9
            if remaining <= 0:
                raise TimeoutError("numerical case deadline exhausted before execution")
            value = execute_case(
                model, view, case, folder, budget, dumps, time.monotonic() + remaining
            )
            if value["status"] != "complete":
                raise RuntimeError("numerical case did not complete")
            for comparison_id in value["comparisons"]:
                summary = read_json(folder / "comparisons" / (comparison_id + ".summary.json"))
                if summary["required_failures"] or summary["behavior_failures"]:
                    result["errors"].append(
                        {
                            "case_id": case["case_id"],
                            "comparison_id": comparison_id,
                            "message": "required numerical or actual-decision gate failed",
                        }
                    )
            if case is planned[-1] or result["errors"]:
                # Keep the watchdog marker live through the final dump index export.
                dumps.close()
            finished_ns = time.perf_counter_ns()
            if (
                finished_ns > case_deadline_ns
                or not 0 <= value["elapsed_s"] <= limits["case_timeout_s"]
            ):
                raise TimeoutError("numerical case exceeded its frozen lifetime deadline")
            ledger["case_lifetimes"][case["case_id"]] = {
                "started_ns": started_ns,
                "finished_ns": finished_ns,
                "deadline_ns": case_deadline_ns,
            }
            ledger["completed_cases"].append(case["case_id"])
            result["completed_cases"].append(case["case_id"])
            ledger["active_case"] = None
            checkpoint()
            if result["errors"]:
                break
        result["complete"] = result["completed_cases"] == [case["case_id"] for case in planned]
        result["passed"] = result["complete"] and not result["errors"]
    except (Exception, KeyboardInterrupt) as exc:
        result["errors"].append({"type": type(exc).__name__, "message": str(exc)})
    finally:
        try:
            dumps.close()
        except (Exception, KeyboardInterrupt) as exc:
            result["complete"] = result["passed"] = False
            result["errors"].append({"type": type(exc).__name__, "message": str(exc)})
        checkpoint()
    return {**result, "ledger": deepcopy(ledger)}


def audit_numerical(output_dir, parent_plan):
    """Offline audit using the same exact boundary/trace and typed-payload audits."""
    from .report import _audit_case, _audit_comparison, _audit_raw_evidence

    view = _view(parent_plan)
    folder = Path(output_dir) / "numerical"
    result = {
        "schema_version": 1,
        "artifact_type": "m2_numerical_report",
        "plan_sha256": parent_plan["plan_sha256"],
        "complete": False,
        "passed": False,
        "completed_cases": [],
        "comparisons": [],
        "errors": [],
    }
    fixtures = {row["fixture_id"]: row for row in view["suite"]["fixtures"]}
    verified_cases = 0
    try:
        ledger = read_json(folder / "ledger.json")
        expected_ids = [case["case_id"] for case in view["execution_order"]]
        result["ledger"] = ledger
        if (
            ledger["plan_sha256"] == parent_plan["plan_sha256"]
            and ledger["numerical_plan_sha256"] == view["numerical_plan_sha256"]
            and ledger["completed_cases"] == expected_ids[: len(ledger["completed_cases"])]
        ):
            result["completed_cases"] = list(ledger["completed_cases"])
        if (
            ledger["plan_sha256"] != parent_plan["plan_sha256"]
            or ledger["numerical_plan_sha256"] != view["numerical_plan_sha256"]
            or ledger["completed_cases"] != expected_ids
            or ledger["started_cases"] != expected_ids
            or set(ledger["workers"]) != {"A", "B"}
            or not all(row["complete"] for row in ledger["workers"].values())
            or ledger["active_case"] is not None
            or set(ledger["case_lifetimes"]) != set(expected_ids)
        ):
            raise ValueError("numerical ledger lacks the complete ordered 27-case execution")
        previous_end = 0
        for case_id in expected_ids:
            lifetime = ledger["case_lifetimes"][case_id]
            start, end, deadline = (
                lifetime[key] for key in ("started_ns", "finished_ns", "deadline_ns")
            )
            if (
                any(type(value) is not int or value <= 0 for value in (start, end, deadline))
                or not previous_end <= start <= end <= deadline
                or deadline - start > view["contract"]["limits"]["case_timeout_s"] * 1_000_000_000
            ):
                raise ValueError("numerical case lifetime exceeded its frozen deadline")
            previous_end = end
        actual_ids = sorted(path.name for path in (folder / "cases").iterdir() if path.is_dir())
        if actual_ids != sorted(expected_ids):
            raise ValueError("unexpected or missing numerical case directories")
        cases = {}
        for case in view["execution_order"]:
            value = read_json(folder / "cases" / case["case_id"] / "result.json")
            _audit_case(value, view, case, fixtures)
            cases[case["case_id"]] = value
            verified_cases += 1
        observations, reverse = {}, {}
        for comparison in view["comparison_order"]:
            case_id = comparison["candidate_case_id"]
            seen = observations.setdefault(case_id, {})
            keys = reverse.setdefault(case_id, {})

            def observe(index, fixture_id, key):
                identity = (fixture_id, key)
                if (index in seen and seen[index] != identity) or (
                    identity in keys and keys[identity] != index
                ):
                    raise ValueError("inconsistent cross-comparison observation order")
                seen[index], keys[identity] = identity, index

            result["comparisons"].append(
                _audit_comparison(folder, view, comparison, cases, fixtures, observe)
            )
        for case_id, seen in observations.items():
            if sorted(seen) != list(range(1, cases[case_id]["observed_boundaries"] + 1)):
                raise ValueError("missing global candidate observation coverage")
        expected_comparisons = sorted(row["comparison_id"] for row in view["comparison_order"])
        actual_comparisons = sorted(path.stem for path in (folder / "comparisons").glob("*.jsonl"))
        if actual_comparisons != expected_comparisons:
            raise ValueError("unexpected or missing numerical comparison artifacts")
        result["raw_evidence"] = _audit_raw_evidence(
            folder, view, cases, fixtures, result["comparisons"], ledger
        )
        expected_groups = {}
        for spool in result["raw_evidence"]["retained_references"]:
            group = cases[spool["namespace"]]["case"]["spool_group"]
            expected_groups[group] = expected_groups.get(group, 0) + spool["size_bytes"]
        expected_groups.update(ledger["diagnostic_dumps"]["fixture_written_bytes"])
        if expected_groups != ledger["spool_bytes_by_group"]:
            raise ValueError("per-execution ledger differs from retained spool/dump sizes")
        if sum(ledger["spool_bytes_by_group"].values()) != ledger["tensor_written_bytes"]:
            raise ValueError("cumulative numerical ledger differs from group sums")
        caps = view["contract"]["limits"]
        if (
            ledger["tensor_written_bytes"] > caps["cumulative_spool_written_bytes"]
            or any(
                size > caps["group_spool_bytes"] for size in ledger["spool_bytes_by_group"].values()
            )
            or ledger["diagnostic_dumps"]["written_bytes"] > caps["persisted_dump_bytes"]
        ):
            raise ValueError("numerical spool/dump budget exceeded")
        result["complete"] = True
        result["passed"] = not any(
            row["required_failures"] or row["behavior_failures"] for row in result["comparisons"]
        )
        for implementation_id in ("A", "B"):
            worker = ledger["workers"][implementation_id]
            worker_cases = [
                case["case_id"]
                for case in view["execution_order"]
                if case["implementation_id"] == implementation_id
            ]
            failed = any(
                (row["required_failures"] or row["behavior_failures"])
                and any(
                    comparison["comparison_id"] == row["comparison_id"]
                    and comparison["candidate_case_id"] in worker_cases
                    for comparison in view["comparison_order"]
                )
                for row in result["comparisons"]
            )
            if (
                worker["implementation_id"] != implementation_id
                or worker["completed_cases"] != worker_cases
                or worker["passed"] is not (not failed)
                or bool(worker["errors"]) != failed
            ):
                raise ValueError("numerical worker summary differs from audited evidence")
        result["ledger"] = ledger
    except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
        result["complete"] = result["passed"] = False
        result["errors"].append({"type": type(exc).__name__, "message": str(exc)})
    result["counts"] = {
        "planned_cases": 27,
        "verified_cases": verified_cases,
        "planned_comparisons": 51,
        "verified_comparisons": len(result["comparisons"]),
    }
    return result
