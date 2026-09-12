"""Two bounded eager/replay held-input lifecycle checks; no pretrained inference."""

import time
from copy import deepcopy
from functools import lru_cache
from pathlib import Path

from vllm_lt.validation import m3_persistent_lifecycle as base

MiB = 1024**2
_METADATA = base._METADATA
_require = base._require
# Failed device completion cannot authorize releasing captured buffers. The
# owning worker stops; retain these task-owned resources until process teardown.
_unsettled = []


def _metadata(step):
    import torch

    padded = step["kind"] == "supported"
    rows = step["row_count"]
    width = 32 if padded else step["required_width"]
    result = {
        "position_ids": torch.zeros(rows, dtype=torch.int64),
        "write_blocks": torch.full((rows,), -1, dtype=torch.int64),
        "write_offsets": torch.full((rows,), -1, dtype=torch.int64),
        "block_tables": torch.full((rows, width), -1, dtype=torch.int32),
        "context_lengths": torch.zeros(rows, dtype=torch.int32),
    }
    if padded:
        result["active"] = torch.zeros(rows, dtype=torch.bool)
    for row, name, depth, pos in zip(
        step["live_rows"], step["request_ids"], step["depths"], step["positions"]
    ):
        table = step["allocations"][name]["block_tables"][depth]
        result["position_ids"][row] = pos
        result["write_blocks"][row] = table[pos // 16]
        result["write_offsets"][row] = pos % 16
        result["context_lengths"][row] = pos + 1
        selected = table[:width]
        result["block_tables"][row, : len(selected)] = torch.tensor(selected, dtype=torch.int32)
        if padded:
            result["active"][row] = True
    return result


def _tensor_keys(step):
    if step["kind"] == "host_only":
        return []
    if step["kind"] == "empty":
        return ["publication_hidden", "publication_gates"]
    keys = (
        ["input"]
        + ["metadata/" + name for name in _metadata(step)]
        + [
            f"layer{layer}/{name}"
            for layer in range(2)
            for name in ("query", "key", "value", "attention")
        ]
        + ["physical_hidden", "physical_gates", "publication_hidden", "publication_gates"]
    )
    if step["step_id"] == 2:
        keys += ["held_hidden", "held_generator"]
    return keys


def _check_snapshot(snapshot, initial, counters, fallbacks, bucket_counts, *, in_flight=None):
    _require(
        snapshot["enabled"] is True
        and snapshot["failure"] is None
        and snapshot["status"] == ("in_flight" if in_flight is not None else "ready")
        and snapshot["use_graphs"] == initial["use_graphs"]
        and snapshot["backend"] == "triton",
        "executor state differs",
    )
    _require(
        snapshot["counters"] == counters and snapshot["fallback_counts"] == fallbacks,
        "executor dispatch/commit/fallback counters differ",
    )
    for field in ("setup", "limits", "device_payload_bytes", "cpu_staging_bytes"):
        _require(snapshot[field] == initial[field], f"executor immutable {field} differs")
    _require(set(snapshot["buckets"]) == {"4", "8"}, "bucket snapshot coverage differs")
    for name, bucket in snapshot["buckets"].items():
        fixed = initial["buckets"][name]
        for field in (
            "row_count",
            "table_width",
            "max_live_rows",
            "setup_generation",
            "tensors",
            "staging_tensors",
            "graph_id",
            "graph_exec_id",
            "pool_id",
            "captured_inputs",
            "captured_outputs",
            "verification",
        ):
            _require(bucket[field] == fixed[field], f"stable bucket {field} changed")
        _require(
            bucket["counters"] == bucket_counts[name]
            and bucket["generation"] == fixed["setup_generation"] + bucket_counts[name]["prepared"]
            and bucket["in_use"] is (in_flight == name)
            and bucket["failed"] is False,
            "bucket generation/lease/counters differ",
        )


def _audit_one(root, plan, evaluation, *, expected_device):
    import math

    import torch

    from benchmarks.capture.validation import _audit_graph_setup
    from vllm_lt.validation.diagnostics import TensorSpool

    eid = evaluation["evaluation_id"]
    result = base._read(root / "evaluations" / eid / "result.json")
    _require(
        result["schema_version"] == 1
        and result["artifact_type"] == "m3_capture_lifecycle_result"
        and result["evaluation"] == evaluation
        and result["lifecycle_plan_sha256"] == plan["lifecycle_plan_sha256"]
        and result["device"] == expected_device
        and result["status"] == "complete"
        and result["passed"] is True
        and result["errors"] == [],
        "lifecycle result identity/status differs",
    )
    start, end, deadline = (result[k] for k in ("started_ns", "finished_ns", "deadline_ns"))
    _require(
        all(type(v) is int for v in (start, end, deadline))
        and 0 < start <= end < deadline
        and deadline - start <= 600 * 10**9,
        "lifecycle lifetime differs",
    )
    _require(
        base._read(root / "evaluations" / eid / "started.json")
        == {
            "evaluation": evaluation,
            "started_ns": start,
            "deadline_ns": deadline,
            "lifecycle_plan_sha256": plan["lifecycle_plan_sha256"],
        },
        "lifecycle start marker differs",
    )
    _require(
        result["cleanup"]
        == {
            "completion_confirmed": True,
            "used_blocks": 0,
            "active_requests": 0,
            "cache_references_released": True,
            "buffer_references_released": True,
        }
        and (
            result["graph_closed"].get("status") == "closed"
            or result["graph_closed"].get("enabled") is False
        ),
        "lifecycle cleanup incomplete",
    )
    initial = result["graph_initial"]
    replay = evaluation["use_graphs"]
    stable, owned = _audit_graph_setup(
        initial,
        {"hidden_size": 2048, "num_hidden_layers": 2, "num_key_value_heads": 16, "head_dim": 128},
        replay=replay,
        block_size=16,
        expected_device=expected_device,
    )
    _require(
        start <= initial["setup"]["started_ns"] <= initial["setup"]["finished_ns"] <= end,
        "graph setup lies outside lifecycle lifetime",
    )
    base._allocation_evidence(result["initial_allocations"], plan["initial_allocations"])
    base._guard_evidence(result["initial_guard_chunks"], plan["initial_guard_chunks"])
    _require(
        len(result["steps"]) == 11
        and [s["step_id"] for s in result["steps"]] == list(range(1, 12)),
        "lifecycle action coverage differs",
    )
    counters = dict.fromkeys(
        ("calls", "empty", "prepared", "eager", "replays", "committed", "completed"), 0
    )
    fallbacks = {"backend": 0, "live_count": 0, "table_width": 0}
    bucket_counts = {
        name: dict.fromkeys(("prepared", "eager", "replays", "committed", "completed"), 0)
        for name in stable
    }
    _check_snapshot(initial, initial, counters, fallbacks, bucket_counts)
    spool = TensorSpool.open(root / "outputs", eid, "evidence")
    expected_keys = [
        f"step{s['step_id']:02d}/{op}" for s in plan["steps"] for op in _tensor_keys(s)
    ]
    _require(
        set(spool.index) == set(expected_keys) and len(spool.index) == 173,
        "held tensor record coverage differs",
    )
    raw_hashes, probabilities, changed_pages = {}, {}, set()
    previous_objects = result["initial_allocations"]
    try:
        for step, record in zip(plan["steps"], result["steps"]):
            _require(
                record["kind"] == step["kind"] and record["status"] == "complete",
                "held action identity/status differs",
            )
            _check_snapshot(record["before"], initial, counters, fallbacks, bucket_counts)
            base._allocation_evidence(record["allocations"], step["allocations"])
            for name, row in record["allocations"].items():
                if name in previous_objects:
                    old = previous_objects[name]
                    _require(
                        (row["object_id"] == old["object_id"])
                        is (row["allocation_id"] == old["allocation_id"]),
                        "allocation object lifetime differs",
                    )
            previous_objects = record["allocations"]
            _require(
                record["prefixes_before"] == step["prefixes_before"]
                and record["prefixes_after"] == step["prefixes_after"],
                "prefix transaction evidence differs",
            )
            base._guard_evidence(record["guard_chunks"], step["guard_chunks"])
            count, rows = len(step["request_ids"]), step["row_count"]
            selected = str(rows) if step["kind"] == "supported" else None
            if step["kind"] != "host_only":
                counters["calls"] += 1
                if step["kind"] == "empty":
                    counters["empty"] += 1
                elif selected:
                    mode = "replays" if replay else "eager"
                    counters["prepared"] += 1
                    counters[mode] += 1
                    bucket_counts[selected]["prepared"] += 1
                    bucket_counts[selected][mode] += 1
                else:
                    fallbacks[step["fallback_reason"]] += 1
            if count:
                _check_snapshot(
                    record["at_publication"],
                    initial,
                    counters,
                    fallbacks,
                    bucket_counts,
                    in_flight=selected or "compact",
                )
                _require(record["input_hashes"] == step["input_hashes"], "held input hashes differ")
                _require(
                    record["prefixes_at_publication"]
                    == step["prefixes_before" if selected else "prefixes_after"],
                    "prefix publication occurs at wrong completion boundary",
                )
                if selected:
                    _require(
                        record["busy_lease_rejected"] is True
                        and record["expired_generation_rejected"] is True,
                        "lease exclusion/expiration evidence missing",
                    )
                    tensors = stable[selected]["tensors"]
                    expected_inputs = {
                        "hidden": tensors["hidden_in"],
                        **{name: tensors[name] for name in _METADATA},
                    }
                    _require(
                        record["actual_inputs"]
                        == {
                            name: {k: v for k, v in d.items() if k != "storage_bytes"}
                            for name, d in expected_inputs.items()
                        },
                        "model inputs did not borrow selected bucket",
                    )
                    dispatch = record["at_publication"]["last_dispatch"]
                    _require(
                        dispatch["actual_inputs"] == expected_inputs
                        and dispatch["actual_physical_outputs"]
                        == {"hidden": tensors["hidden_out"], "gates": tensors["gate_out"]},
                        "actual dispatch pointers differ from stable bucket",
                    )
                    counters["committed"] += 1
                    bucket_counts[selected]["committed"] += 1
                    bucket_counts[selected]["completed"] += 1
                counters["completed"] += 1
                publication = record["publication"]
                for name, shape in (("hidden", [count, 2048]), ("gates", [count])):
                    base._pointer(publication[name], shape, "torch.float32", device=expected_device)
                    _require(
                        publication[name]["storage_ptr"] not in owned,
                        "publication aliases graph storage",
                    )
                _require(
                    publication["hidden"]["storage_ptr"] != publication["gates"]["storage_ptr"],
                    "hidden and gate publications alias",
                )
                _require(
                    record["after"]["last_publication"] is not None
                    and {
                        name: {k: v for k, v in d.items() if k != "storage_bytes"}
                        for name, d in record["after"]["last_publication"].items()
                    }
                    == publication,
                    "returned publication pointer evidence differs",
                )
                values = record["gate_probabilities"]
                _require(
                    len(values) == count
                    and all(type(v) is float and math.isfinite(v) and 0 <= v <= 1 for v in values),
                    "actual gate readback is incomplete/nonfinite",
                )
                probabilities[step["step_id"]] = values
            elif step["kind"] == "host_only":
                _require(
                    all(
                        record[k] is True
                        for k in (
                            "stale_allocation_rejected",
                            "stale_generation_rejected",
                            "duplicate_commit_rejected",
                        )
                    ),
                    "stale owner/generation or duplicate commit was not rejected",
                )
            _check_snapshot(record["after"], initial, counters, fallbacks, bucket_counts)
            if step["kind"] != "host_only":
                dispatch = record["after"]["last_dispatch"]
                expected_kind = (
                    "empty"
                    if not count
                    else ("replay" if replay else "eager")
                    if selected
                    else "compact"
                )
                for key, expected in {
                    "dispatch_id": counters["calls"],
                    "kind": expected_kind,
                    "request_ids": step["request_ids"],
                    "depths": step["depths"],
                    "positions": step["positions"],
                    "live_rows": step["live_rows"],
                    "row_count": rows,
                    "table_width": 32 if selected else step["required_width"],
                    "bucket_id": int(selected) if selected else None,
                    "generation": initial["buckets"][selected]["setup_generation"]
                    + bucket_counts[selected]["prepared"]
                    if selected
                    else None,
                    "ticket_state": "committed" if selected else "cancelled" if count else None,
                }.items():
                    _require(dispatch[key] == expected, f"held dispatch {key} differs")
            for name in stable:
                _require(
                    set(record["buffer_hashes_before"][name]) == set(stable[name]["tensors"])
                    and set(record["buffer_hashes_after"][name]) == set(stable[name]["tensors"]),
                    "buffer hash inventory differs",
                )
                if name != selected:
                    _require(
                        record["buffer_hashes_before"][name] == record["buffer_hashes_after"][name],
                        "nonselected bucket changed during empty/fallback/other-bucket action",
                    )
            tensors = {}
            for operation in _tensor_keys(step):
                key = f"step{step['step_id']:02d}/{operation}"
                _require(
                    spool.read_metadata(key)
                    == {"evaluation_id": eid, "step_id": step["step_id"], "operation": operation},
                    "held tensor metadata differs",
                )
                tensor = spool.read(key)
                _require(bool(tensor.isfinite().all()), "nonfinite raw held tensor")
                raw_hashes[key] = (
                    str(tensor.dtype),
                    tuple(tensor.shape),
                    base._tensor_hash(tensor),
                )
                tensors[operation] = tensor
            if count:
                expected_hidden = _physical_hidden(step)
                _require(
                    torch.equal(tensors["input"], expected_hidden), "raw physical input differs"
                )
                for name, value in _metadata(step).items():
                    _require(
                        torch.equal(tensors["metadata/" + name], value), "raw held metadata differs"
                    )
                for layer in range(2):
                    q = expected_hidden.reshape(rows, 16, 128)
                    k, v = base._kv(expected_hidden, layer)
                    for name, value in (
                        ("query", q),
                        ("key", k.reshape_as(q)),
                        ("value", v.reshape_as(q)),
                    ):
                        _require(
                            torch.equal(tensors[f"layer{layer}/{name}"], value),
                            "raw held projection differs",
                        )
                    attention = tensors[f"layer{layer}/attention"]
                    _require(
                        attention.shape == q.shape and attention.dtype == torch.float32,
                        "held attention shape/dtype differs",
                    )
                    if selected:
                        inactive = attention[~tensors["metadata/active"]]
                        _require(
                            not bool(inactive.count_nonzero())
                            and not bool(inactive.signbit().any()),
                            "inactive attention is not positive zero",
                        )
                expected_output = (
                    tensors["layer0/attention"] + tensors["layer1/attention"]
                ).reshape(rows, 2048)
                _require(
                    torch.equal(tensors["physical_hidden"], expected_output)
                    and torch.equal(tensors["physical_gates"], expected_output[:, 0]),
                    "held output arithmetic differs",
                )
                _require(
                    torch.equal(tensors["publication_hidden"], expected_output[step["live_rows"]])
                    and torch.equal(
                        tensors["publication_gates"], expected_output[step["live_rows"], 0]
                    ),
                    "logical publication rows differ",
                )
                if replay and selected:
                    scratch = set(initial["setup"]["scratch"]["pages"])
                    if any(
                        int(x) not in scratch
                        for x in tensors["metadata/write_blocks"][step["live_rows"]]
                    ):
                        changed_pages.add(selected)
            elif step["kind"] == "empty":
                _require(
                    tensors["publication_hidden"].shape == (0, 2048)
                    and tensors["publication_gates"].shape == (0,),
                    "empty output shape differs",
                )
            if step["step_id"] == 2:
                held = result["held_initial"]
                _require(
                    held["request_id"] == "r2"
                    and held["stage"] == "coda"
                    and base._tensor_hash(tensors["held_hidden"]) == held["hidden_sha256"]
                    and base._tensor_hash(tensors["held_generator"]) == held["generator_sha256"]
                    and torch.equal(tensors["held_hidden"], tensors["publication_hidden"][2]),
                    "held coda raw evidence differs",
                )
        _require(
            result["held_checks"]
            == [{"step_id": n, "state": result["held_initial"]} for n in range(2, 12)],
            "held coda/RNG lifetime coverage differs",
        )
        _check_snapshot(result["graph_final"], initial, counters, fallbacks, bucket_counts)
        expected = plan["expected_executor"]
        _require(
            counters["calls"] == 10
            and counters["prepared"] == counters["committed"] == 7
            and counters["completed"] == 9
            and counters["empty"] == 1
            and fallbacks == expected["fallbacks"]
            and {k: v["prepared"] for k, v in bucket_counts.items()} == expected["bucket_visits"],
            "final lifecycle counts differ",
        )
        _require(
            not replay or changed_pages == {"4", "8"},
            "both graphs must use pages different from setup",
        )
    finally:
        spool.close()
    tensor_bytes, json_bytes = base._evidence_bytes(
        root / "evaluations" / eid, root / "outputs" / eid
    )
    _require(
        tensor_bytes <= 8 * MiB and json_bytes <= 4 * MiB, "lifecycle evidence byte cap exceeded"
    )
    return (
        {
            "evaluation_id": eid,
            "started_ns": start,
            "finished_ns": end,
            "deadline_ns": deadline,
            "result_sha256": base._file_hash(root / "evaluations" / eid / "result.json"),
            "tensor_records": len(raw_hashes),
            "guard_chunks": 480,
            "tensor_bytes": tensor_bytes,
            "json_bytes": json_bytes,
        },
        raw_hashes,
        probabilities,
    )


def _evaluation_prefix(root, plan, kind, expected_device):
    """Resolve a settled planned prefix, allowing one terminal failed evaluation."""
    order = plan["execution_order"]
    actual = {p.name for p in (root / "evaluations").glob("*") if p.is_dir()}
    _require(
        actual == {r["evaluation_id"] for r in order[: len(actual)]},
        "settled evaluation prefix has a hole or unknown row",
    )
    complete, stopped = [], None
    for row in order[: len(actual)]:
        _require(stopped is None, "execution continued after a failed evaluation")
        result = base._read(root / "evaluations" / row["evaluation_id"] / "result.json")
        if result.get("status") == "complete" and result.get("passed") is True:
            complete.append(row)
            continue
        field = f"{kind}_plan_sha256"
        _require(
            result["artifact_type"] == f"m3_capture_{kind}_result"
            and result["schema_version"] == 1
            and result["evaluation"] == row
            and result[field] == plan[field]
            and result["device"] == expected_device
            and result["status"] in {"failed", "incomplete"}
            and result["passed"] is False
            and isinstance(result["errors"], list)
            and result["errors"],
            "failed evaluation identity differs",
        )
        start, end, deadline = (result[k] for k in ("started_ns", "finished_ns", "deadline_ns"))
        _require(
            all(type(v) is int for v in (start, end, deadline))
            and 0 < start <= end
            and 0 < deadline - start <= 600 * 10**9,
            "failed evaluation chronology differs",
        )
        _require(
            base._read(root / "evaluations" / row["evaluation_id"] / "started.json")
            == {
                "evaluation": row,
                "started_ns": start,
                "deadline_ns": deadline,
                field: plan[field],
            },
            "failed start marker differs",
        )
        stopped = result
    return complete, stopped


def _failed_lifecycle_evidence(root, plan, result, prior):
    from vllm_lt.validation.diagnostics import TensorSpool

    known = False
    steps = result.get("steps", [])
    _require(
        [r["step_id"] for r in steps] == list(range(1, len(steps) + 1)) and len(steps) <= 11,
        "failed lifecycle action prefix differs",
    )
    for expected, row in zip(plan["steps"], steps):
        if "input_hashes" in row:
            _require(
                row["input_hashes"] == expected["input_hashes"],
                "failed lifecycle input identity differs",
            )
        chunks = row["guard_chunks"]
        _require(len(chunks) <= 40, "failed guard prefix exceeds plan")
        for actual, wanted in zip(chunks, expected["guard_chunks"]):
            _require(
                {k: v for k, v in actual.items() if k != "actual_sha256"} == wanted,
                "failed guard identity differs",
            )
            _require(
                isinstance(actual["actual_sha256"], str) and len(actual["actual_sha256"]) == 64,
                "failed guard hash missing",
            )
            known |= actual["actual_sha256"] != wanted["sha256"]
    eid = result["evaluation"]["evaluation_id"]
    index = root / "outputs" / eid / "evidence.index.json"
    if index.exists():
        spool = TensorSpool.open(root / "outputs", eid, "evidence")
        try:
            expected_keys = [
                f"step{s['step_id']:02d}/{op}" for s in plan["steps"] for op in _tensor_keys(s)
            ]
            _require(
                set(spool.index) == set(expected_keys[: len(spool.index)]),
                "failed raw tensor prefix differs",
            )
            for key in spool.index:
                value = spool.read(key)
                number, operation = key.split("/", 1)
                _require(
                    spool.read_metadata(key)
                    == {"evaluation_id": eid, "step_id": int(number[4:]), "operation": operation},
                    "failed raw tensor metadata differs",
                )
                known |= not bool(value.isfinite().all())
                if prior is not None:
                    known |= (
                        str(value.dtype),
                        tuple(value.shape),
                        base._tensor_hash(value),
                    ) != prior[0][key]
        finally:
            spool.close()
    return known


def audit_lifecycle_outputs(output_dir, plan, *, expected_device="cuda:0", allow_prefix=False):
    report = {
        "schema_version": 1,
        "artifact_type": "m3_capture_lifecycle_report",
        "complete": False,
        "passed": False,
        "errors": [],
        "evaluations": [],
        "completed_evaluations": [],
        "valid_prefix": False,
        "known_required_failure": False,
        "stopped_evaluation": None,
        "lifecycle_plan_sha256": plan.get("lifecycle_plan_sha256"),
    }
    root = Path(output_dir)
    try:
        validate_lifecycle_plan(plan)
        order, stopped = _evaluation_prefix(root, plan, "lifecycle", expected_device)
        _require(
            allow_prefix or (len(order) == 2 and stopped is None),
            "lifecycle evaluation coverage differs",
        )
        ids = {x["evaluation_id"] for x in order}
        expected_files = {
            f"{eid}/evidence{suffix}" for eid in ids for suffix in (".bin", ".index.json")
        }
        actual_files = {
            str(p.relative_to(root / "outputs"))
            for p in (root / "outputs").rglob("*")
            if p.is_file()
        }
        tail_files = (
            {
                f"{stopped['evaluation']['evaluation_id']}/evidence{suffix}"
                for suffix in (".bin", ".index.json")
            }
            if stopped
            else set()
        )
        _require(
            expected_files <= actual_files <= expected_files | tail_files,
            "lifecycle raw spool coverage differs",
        )
        prior, previous = None, 0
        for evaluation in order:
            row, hashes, probabilities = _audit_one(
                root, plan, evaluation, expected_device=expected_device
            )
            _require(previous <= row["started_ns"], "lifecycle A/B order differs")
            previous = row["finished_ns"]
            if prior is not None:
                _require(
                    (hashes, probabilities) == prior, "raw held A/B tensors or actual gates differ"
                )
            prior = hashes, probabilities
            report["evaluations"].append(row)
            report["completed_evaluations"].append(evaluation["evaluation_id"])
        if stopped is not None:
            _require(previous <= stopped["started_ns"], "failed lifecycle order differs")
            report["known_required_failure"] = _failed_lifecycle_evidence(
                root, plan, stopped, prior
            )
            report["stopped_evaluation"] = {
                "evaluation_id": stopped["evaluation"]["evaluation_id"],
                "started_ns": stopped["started_ns"],
                "finished_ns": stopped["finished_ns"],
                "deadline_ns": stopped["deadline_ns"],
                "errors": stopped["errors"],
                "result_sha256": base._file_hash(
                    root / "evaluations" / stopped["evaluation"]["evaluation_id"] / "result.json"
                ),
            }
        size = sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
        _require(
            size <= plan["config"]["suite_evidence_bytes"], "lifecycle suite byte cap exceeded"
        )
        report["artifact_bytes"] = size
        report["valid_prefix"] = True
        report["complete"] = report["passed"] = len(order) == 2 and stopped is None
    except (OSError, ValueError, KeyError, TypeError, IndexError, RuntimeError) as error:
        report["errors"].append({"type": type(error).__name__, "message": str(error)[:1024]})
    return report


def _physical_hidden(step):
    import torch

    hidden = base._hidden(step["step_id"], step["request_ids"])
    if step["kind"] != "supported":
        return hidden
    result = torch.zeros((step["row_count"], 2048), dtype=torch.float32)
    result[step["live_rows"]] = hidden
    return result


@lru_cache(maxsize=1)
def _resolved_plan():
    plan = base.build_lifecycle_plan()
    plan.pop("lifecycle_plan_sha256")
    plan.pop("expected_persistent")
    plan["artifact_type"] = "m3_capture_lifecycle_plan"
    plan["config"].pop("row_count")
    plan["config"]["row_counts"] = [4, 8]
    plan["scope"] = (
        "two-layer held tensor arithmetic; separate15 Ouro cases qualify real-model behavior"
    )
    plan["empty_policy"] = (
        "one explicit empty invocation per side, zero physical rows and no graph replay"
    )
    plan["observation_policy"] = (
        "read actual retained tensor-body outputs after completion, including intermediates; "
        "no replay-time Python hook claim"
    )
    plan["execution_order"] = [
        {
            "evaluation_id": f"LIFE-{side}-triton",
            "implementation_id": side,
            "backend": "triton",
            "use_graphs": side == "B",
        }
        for side in ("A", "B")
    ]
    plan["resource_estimates"].update(
        evaluations=2,
        tensor_bytes_upper_bound=16 * MiB,
        json_bytes_upper_bound=8 * MiB,
        artifact_bytes_upper_bound=48 * MiB,
    )
    plan["setup_budget"] = {
        "A": {"warmups": 0, "captures": 0, "verification_replays": 0},
        "B": {"warmups": 6, "captures": 2, "verification_replays": 2},
    }
    plan["expected_executor"] = {
        "supported": 7,
        "fallbacks": {"backend": 0, "live_count": 1, "table_width": 1},
        "bucket_visits": {"4": 4, "8": 3},
        "calls": 10,
        "completed": 9,
        "empty": 1,
        "device_payload_bytes": 198588,
        "cpu_staging_bytes": 1884,
    }
    writes = {}
    initial = plan["initial_allocations"]
    for position in range(510):
        block = initial["long"]["block_tables"][0][position // 16]
        for layer in range(2):
            writes[block, layer, position % 16] = ("seed", position)
    guards = {(r["component"], r["start_block"]): r["sha256"] for r in plan["initial_guard_chunks"]}
    prefixes = {name: [[0] * 2 for _ in range(4)] for name in initial}
    prefixes["long"][0] = [510, 510]
    for step in plan["steps"]:
        if step["step_id"] == 3:
            step.update(
                name="four_row_long_at_511",
                request_ids=["long"],
                depths=[0],
                positions=[511],
                required_width=32,
            )
        elif step["step_id"] == 5:
            step["positions"] = [2, 2]
        for action in step["actions"]:
            if action["action"] == "free":
                prefixes.pop(action["request_id"])
            else:
                prefixes[action["request_id"]] = [[0] * 2 for _ in range(4)]
        step["prefixes_before"] = deepcopy(prefixes)
        if step["kind"] == "supported":
            step["row_count"] = 4 if len(step["request_ids"]) <= 2 else 8
        dirty = set()
        if step["request_ids"]:
            step["input_hashes"] = {
                "logical_hidden": base._tensor_hash(
                    base._hidden(step["step_id"], step["request_ids"])
                ),
                "physical_hidden": base._tensor_hash(_physical_hidden(step)),
                **{key: base._tensor_hash(value) for key, value in _metadata(step).items()},
            }
            for key, depth, position in zip(step["request_ids"], step["depths"], step["positions"]):
                block = step["allocations"][key]["block_tables"][depth][position // 16]
                dirty.add(block // 8 * 8)
                for layer in range(2):
                    _require(
                        prefixes[key][depth][layer] >= position,
                        "held fixture has a preceding history hole",
                    )
                    prefixes[key][depth][layer] = max(prefixes[key][depth][layer], position + 1)
                    writes[block, layer, position % 16] = ("step", step["step_id"], key)
        for component in range(2):
            for start in dirty:
                guards[component, start] = base._tensor_hash(
                    base._expected_chunk(component, start, writes)
                )
        step["guard_chunks"] = [
            {**row, "sha256": guards[row["component"], row["start_block"]]}
            for row in step["guard_chunks"]
        ]
        step["prefixes_after"] = deepcopy(prefixes)
    plan["lifecycle_plan_sha256"] = base._digest(plan)
    return plan


def build_lifecycle_plan():
    return deepcopy(_resolved_plan())


def validate_lifecycle_plan(plan):
    _require(
        base._digest(plan) == base._digest(_resolved_plan()), "capture lifecycle matrix differs"
    )


def _prefixes(cache):
    _require(
        all(not p.pending for a in cache._allocations.values() for d in a.written for p in d),
        "held decode unexpectedly contains sparse written positions",
    )
    return {
        name: [[p.prefix for p in depth] for depth in a.written]
        for name, a in cache._allocations.items()
    }


def _buffer_hashes(executor):
    return {
        str(rows): {name: base._tensor_hash(tensor) for name, tensor in bucket["tensors"].items()}
        for rows, bucket in executor.buckets.items()
    }


def _probe_model(device):
    """Capture-safe held body; retained outputs are observed only after replay."""
    from types import SimpleNamespace

    import torch

    class Boundary(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.marker = torch.nn.Parameter(torch.zeros(1, device=device), requires_grad=False)
            self.config = SimpleNamespace(hidden_size=2048)
            self.observed = {}

        def recurrent(self, hidden, ids, depths, positions, cache):
            batch = cache._prepare_batch(ids, depths, positions)
            return self._body(hidden, batch, cache)

        def _recurrent_tensor(self, hidden, view):
            return self._body(hidden, view, view)

        def _body(self, hidden, batch, cache):
            active = batch.active
            clean = hidden if active is None else hidden.masked_fill(~active[:, None], 0)
            values = {
                "input": hidden,
                **{
                    "metadata/" + name: getattr(batch, name)
                    for name in _METADATA
                    if getattr(batch, name) is not None
                },
            }
            outputs = []
            for layer in range(2):
                query = clean.reshape(-1, 16, 128)
                key, value = base._kv(clean, layer)
                key, value = key.reshape(-1, 16, 128), value.reshape(-1, 16, 128)
                cache._write_prepared(layer, batch, key, value)
                output = cache._attend_prepared(layer, batch, query)
                values.update(
                    {
                        f"layer{layer}/query": query,
                        f"layer{layer}/key": key,
                        f"layer{layer}/value": value,
                        f"layer{layer}/attention": output,
                    }
                )
                outputs.append(output.reshape(-1, 2048))
            physical = outputs[0] + outputs[1]
            gates = physical[:, 0].clone(memory_format=torch.contiguous_format)
            values.update(physical_hidden=physical, physical_gates=gates)
            # Capture runs Python once; subsequent graph replays update this
            # bucket's retained tensors, not this dictionary or a Python hook.
            self.observed[batch.row_count] = values
            return physical, gates

        def coda(self, hidden):
            raise AssertionError("held coda request must remain paused")

    return Boundary()


def _create_cache(device):
    import torch

    from vllm_lt.core.kv_cache_manager import KVCacheManager

    return KVCacheManager(
        2, 16, 128, 160, 16, 4, device=device, dtype=torch.float32, backend="triton"
    )


def run_lifecycle_evaluation(plan, evaluation_id, output_dir, device="cuda", deadline_ns=None):
    started = time.perf_counter_ns()
    import gc
    import weakref

    import torch

    from vllm_lt.core.scheduler import ScheduledItem, SchedulerOutput
    from vllm_lt.request import Request, Stage
    from vllm_lt.sampling_params import SamplingParams
    from vllm_lt.validation.diagnostics import SpoolBudget, TensorSpool
    from vllm_lt.worker.model_runner import ModelRunner

    validate_lifecycle_plan(plan)
    matches = [row for row in plan["execution_order"] if row["evaluation_id"] == evaluation_id]
    _require(len(matches) == 1, "unknown capture lifecycle evaluation")
    evaluation = matches[0]
    root, target = Path(output_dir), torch.device(device)
    folder = root / "evaluations" / evaluation_id
    folder.mkdir(parents=True, exist_ok=False)
    deadline = min(
        started + 600 * 10**9, deadline_ns if deadline_ns is not None else started + 600 * 10**9
    )
    marker = {
        "evaluation": evaluation,
        "started_ns": started,
        "deadline_ns": deadline,
        "lifecycle_plan_sha256": plan["lifecycle_plan_sha256"],
    }
    base._write(folder / "started.json", marker)
    result = {
        "schema_version": 1,
        "artifact_type": "m3_capture_lifecycle_result",
        **marker,
        "device": str(target),
        "status": "incomplete",
        "passed": False,
        "errors": [],
        "steps": [],
        "initial_guard_chunks": [],
        "held_checks": [],
        "cleanup": {
            "completion_confirmed": False,
            "used_blocks": None,
            "active_requests": None,
            "cache_references_released": False,
            "buffer_references_released": False,
        },
    }
    cache = model = runner = spool = reference = held = hidden = out = observed = None
    old_host = old_batch = old_ticket = publication = selected = request = tensor = None
    requests, weak_cache, weak_buffers, state = {}, [], [], {}

    def check():
        if time.perf_counter_ns() >= deadline:
            raise TimeoutError("capture lifecycle case lifetime exceeded")

    def emit(step, operation, value):
        check()
        key = f"step{step['step_id']:02d}/{operation}"
        record = spool.write(
            key,
            value,
            {"evaluation_id": evaluation_id, "step_id": step["step_id"], "operation": operation},
        )
        _require(bool(value.isfinite().all()), "nonfinite held tensor")
        if reference is not None:
            prior = reference._verified_record(key)
            reference.read(key)
            _require(
                (record["dtype"], record["shape"], record["sha256"])
                == (prior["dtype"], prior["shape"], prior["sha256"]),
                f"held B/A tensor differs at {key}",
            )

    def guards(expected, destination):
        for item in expected:
            check()
            pool = (cache.key_cache, cache.value_cache)[item["component"]]
            digest = base._tensor_hash(pool[item["start_block"] : item["start_block"] + 8])
            destination.append({**item, "actual_sha256": digest})
            _require(digest == item["sha256"], "whole-pool active/guard bytes differ")

    def apply(actions):
        for action in actions:
            name = action["request_id"]
            if action["action"] == "free":
                cache.free(name)
                requests.pop(name, None)
            else:
                _require(cache.allocate(name, action["max_tokens"]), "held admission failed")
                if name.startswith("r") or name == "long":
                    requests[name] = Request(
                        name, [1] * (511 if name == "long" else 1), SamplingParams(max_tokens=600)
                    )

    try:
        check()
        cache = _create_cache(target)
        target = cache.device
        result["device"] = str(target)
        weak_cache = [weakref.ref(cache.key_cache), weakref.ref(cache.value_cache)]
        for component, tensor in enumerate((cache.key_cache, cache.value_cache)):
            for start in range(0, 160, 8):
                check()
                tensor[start : start + 8].copy_(base._base_chunk(component, start))
        tensor = None
        model, runner = _probe_model(target), None
        runner = ModelRunner(model, cache)
        runner._enable_recurrent_graph(use_graphs=evaluation["use_graphs"])
        executor = runner._decode_executor
        executor.record_dispatch_tensors = True
        result["graph_initial"] = runner._graph_snapshot()
        _require(
            not cache._allocations and cache._free_blocks == list(reversed(range(160))),
            "setup did not restore empty allocator identity",
        )
        weak_buffers = [
            weakref.ref(t) for b in executor.buckets.values() for t in b["tensors"].values()
        ]
        spool = TensorSpool(
            root / "outputs",
            evaluation_id,
            "evidence",
            SpoolBudget(8 * MiB, 8 * MiB),
            evaluation_id,
            max_tensor_bytes=MiB,
            matching_bytes=64 * MiB,
        )
        if evaluation["implementation_id"] == "B":
            prior = base._read(root / "evaluations" / "LIFE-A-triton" / "result.json")
            _require(
                prior["passed"] and prior["status"] == "complete",
                "successful held A reference missing",
            )
            reference = TensorSpool.open(root / "outputs", "LIFE-A-triton", "evidence")
        apply(plan["setup_actions"])
        result["initial_allocations"] = base._allocations(cache, plan["initial_allocations"])
        for layer in range(2):
            key, value = (base._seed(layer, component, range(510)) for component in range(2))
            cache.write(
                layer,
                ["long"] * 510,
                [0] * 510,
                list(range(510)),
                key.to(target).reshape(510, 16, 128),
                value.to(target).reshape(510, 16, 128),
            )
        key = value = None
        guards(plan["initial_guard_chunks"], result["initial_guard_chunks"])
        for step in plan["steps"]:
            check()
            record = {
                "step_id": step["step_id"],
                "kind": step["kind"],
                "status": "incomplete",
                "guard_chunks": [],
                "before": runner._graph_snapshot(),
                "buffer_hashes_before": _buffer_hashes(executor),
            }
            result["steps"].append(record)
            apply(step["actions"])
            record["allocations"] = base._allocations(cache, step["allocations"])
            record["prefixes_before"] = _prefixes(cache)
            _require(
                record["prefixes_before"] == step["prefixes_before"],
                "preceding prefixes differ from fixture",
            )
            if step["kind"] == "host_only":
                for label, operation in (
                    ("stale_allocation_rejected", lambda: cache._begin_decode_traversal(old_host)),
                    ("stale_generation_rejected", lambda: cache._require_live_batch(old_batch)),
                    (
                        "duplicate_commit_rejected",
                        lambda: cache._commit_decode_traversal(
                            old_ticket, completion_confirmed=True
                        ),
                    ),
                ):
                    try:
                        operation()
                    except RuntimeError:
                        record[label] = True
                    else:
                        raise ValueError(f"{label} did not reject")
                old_host = old_batch = old_ticket = None
            elif step["kind"] == "empty":
                out = runner._recurrent(base._hidden(step["step_id"], []).to(target), [], [], [])
                emit(step, "publication_hidden", out[0])
                emit(step, "publication_gates", out[1])
                out = None
            else:
                hidden = base._hidden(step["step_id"], step["request_ids"]).to(target)
                selected = []
                for index, (name, depth, position) in enumerate(
                    zip(step["request_ids"], step["depths"], step["positions"])
                ):
                    request = requests[name]
                    request.generated_token_ids = [1] * (
                        position - len(request.prompt_token_ids) + 1
                    )
                    request.loops_done, request.hidden_state, request.stage = (
                        depth,
                        hidden[index],
                        Stage.RECURRENT,
                    )
                    selected.append(ScheduledItem(request))
                original = runner._recurrent

                def publication(*args, **kwargs):
                    outputs = original(*args, **kwargs)
                    state["outputs"] = outputs
                    state["batch"], state["ticket"] = executor.batch, executor.ticket
                    record["at_publication"] = runner._graph_snapshot()
                    record["prefixes_at_publication"] = _prefixes(cache)
                    if step["kind"] == "supported":
                        _require(
                            record["prefixes_at_publication"] == step["prefixes_before"],
                            "tensor body prematurely committed prefixes",
                        )
                        try:
                            cache._prepare_into(executor.batch.storage, executor.ticket.host)
                        except RuntimeError:
                            record["busy_lease_rejected"] = True
                        else:
                            raise ValueError("in-flight lease accepted another preparation")
                    return outputs

                runner._recurrent = publication
                try:
                    record["gate_probabilities"] = runner.execute(
                        SchedulerOutput(Stage.RECURRENT, selected)
                    )
                finally:
                    del runner._recurrent
                out = state.pop("outputs")
                observed = model.observed[step["row_count"]]
                record["input_hashes"] = {
                    "logical_hidden": base._tensor_hash(hidden),
                    "physical_hidden": base._tensor_hash(observed["input"]),
                    **{
                        name: base._tensor_hash(observed["metadata/" + name])
                        for name in _METADATA
                        if "metadata/" + name in observed
                    },
                }
                _require(
                    record["input_hashes"] == step["input_hashes"],
                    "actual held inputs differ from frozen values",
                )
                record["actual_inputs"] = {
                    "hidden": base._description(observed["input"]),
                    **{
                        name: base._description(observed["metadata/" + name])
                        for name in _METADATA
                        if "metadata/" + name in observed
                    },
                }
                for operation, value in observed.items():
                    emit(step, operation, value)
                emit(step, "publication_hidden", out[0])
                emit(step, "publication_gates", out[1])
                record["publication"] = {
                    "hidden": base._description(out[0]),
                    "gates": base._description(out[1]),
                }
                if step["kind"] == "supported":
                    try:
                        cache._require_live_batch(state["batch"])
                    except RuntimeError:
                        record["expired_generation_rejected"] = True
                    else:
                        raise ValueError("completed lease remained usable")
                if step["step_id"] == 2:
                    held = requests["r2"]
                    held.stage = Stage.CODA
                    held.generator = torch.Generator(device="cpu").manual_seed(17)
                    result["held_initial"] = base._held_state(held)
                    emit(step, "held_hidden", held.hidden_state)
                    emit(step, "held_generator", held.generator.get_state())
                if step["step_id"] == 5:
                    old_host = cache._prepare_host_batch(
                        step["request_ids"], step["depths"], step["positions"]
                    )
                    old_batch, old_ticket = state["batch"], state["ticket"]
                selected = request = hidden = out = observed = value = original = publication = None
                state.clear()
            record["after"] = runner._graph_snapshot()
            record["buffer_hashes_after"] = _buffer_hashes(executor)
            record["prefixes_after"] = _prefixes(cache)
            _require(
                record["prefixes_after"] == step["prefixes_after"],
                "completed prefixes differ from fixture",
            )
            if held is not None:
                actual = base._held_state(held)
                _require(
                    actual == result["held_initial"], "held coda state/RNG/publication changed"
                )
                result["held_checks"].append({"step_id": step["step_id"], "state": actual})
            guards(step["guard_chunks"], record["guard_chunks"])
            record["status"] = "complete"
        result["graph_final"] = runner._graph_snapshot()
        result["status"], result["passed"] = "complete", True
    except BaseException as error:
        result["status"] = (
            "incomplete" if isinstance(error, (TimeoutError, KeyboardInterrupt)) else "failed"
        )
        result["errors"].append({"type": type(error).__name__, "message": str(error)[:1024]})
    finally:
        for writer in (spool, reference):
            if writer is not None:
                try:
                    writer.close()
                except BaseException as error:
                    result["errors"].append(
                        {"type": "evidence_cleanup", "message": str(error)[:1024]}
                    )
        try:
            if runner is not None and runner._decode_executor is not None:
                runner._decode_executor.synchronize()
            elif target.type == "cuda":
                torch.cuda.synchronize(target)
            result["cleanup"]["completion_confirmed"] = True
            if cache is not None:
                for name in list(cache._allocations):
                    cache.free(name)
                result["cleanup"]["used_blocks"] = cache.num_used_blocks
                requests.clear()
                result["cleanup"]["active_requests"] = 0
            if runner is not None:
                runner._close_recurrent_graph()
                result["graph_closed"] = runner._graph_snapshot()
        except BaseException as error:
            result["errors"].append({"type": "execution_cleanup", "message": str(error)[:1024]})
            _unsettled.append((cache, model, runner, dict(requests)))
        if model is not None:
            model.observed.clear()
        if runner is not None:
            runner.__dict__.pop("_recurrent", None)
        requests.clear()
        state.clear()
        cache = model = runner = executor = held = hidden = out = observed = None
        old_host = old_batch = old_ticket = publication = selected = request = tensor = None
        key = value = original = operation = None
        gc.collect()
        result["cleanup"]["cache_references_released"] = bool(weak_cache) and all(
            ref() is None for ref in weak_cache
        )
        result["cleanup"]["buffer_references_released"] = bool(weak_buffers) and all(
            ref() is None for ref in weak_buffers
        )
        if result["errors"] or result["cleanup"] != {
            "completion_confirmed": True,
            "used_blocks": 0,
            "active_requests": 0,
            "cache_references_released": True,
            "buffer_references_released": True,
        }:
            result["status"], result["passed"] = "failed", False
            if not result["errors"]:
                result["errors"].append(
                    {
                        "type": "lifecycle_cleanup",
                        "message": "lifecycle cleanup evidence is incomplete",
                    }
                )
        base._persist_result(folder, root / "outputs" / evaluation_id, result, deadline)
    return result
