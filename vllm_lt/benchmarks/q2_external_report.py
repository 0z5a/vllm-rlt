"""Bounded, portable CPU readback of cached Q2 timings and actual output identity."""

import hashlib
import json
import math
from pathlib import Path

from .ab_schema import equal, require
from .observe import summarize_records
from .q2_external_driver import cpu_control_result
from .q2_external_schema import LIMITS, canonical_gpu_uuid, validate_plan
from .schema import _integer, _keys, read_json


def _clock(value, label, lower=0, upper=None):
    _integer(value, label, minimum=lower)
    if upper is not None:
        require(value <= upper, label + " exceeds its enclosing lifetime")
    return value


def _memory(value):
    _keys(value, ("allocated_bytes", "reserved_bytes"), name="memory counters")
    for key, number in value.items():
        _integer(number, key)
    require(value["reserved_bytes"] >= value["allocated_bytes"], "reserved memory below allocated")


def _inventory(root):
    total, sizes, hashes = 0, {}, {}
    for path in sorted(root.rglob("*")):
        require(not path.is_symlink() and (path.is_file() or path.is_dir()), "unsafe artifact path")
        if not path.is_file():
            continue
        # Derived outputs must not become inputs to their own next generation.
        if path.parent == root and path.name in (
            "q2-external-report.json",
            "q2-external-report.md",
        ):
            continue
        size = path.stat().st_size
        total += size
        parts = path.relative_to(root).parts
        if len(parts) >= 3 and parts[0] == "runs":
            sizes[parts[1]] = sizes.get(parts[1], 0) + size
        require(total <= LIMITS["artifact_bytes_max"], "artifact byte cap exceeded")
        require(size <= LIMITS["artifact_bytes_max"], "individual artifact exceeds cap")
        if path.suffix == ".json":
            hashes[str(path.relative_to(root))] = {
                "size_bytes": size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
    require(all(v <= LIMITS["case_bytes_max"] for v in sizes.values()), "case byte cap exceeded")
    return {"total_bytes": total, "case_bytes": sizes, "json_files": hashes}


def _parameter_shapes(config):
    h, v, m = config["hidden_size"], config["vocab_size"], config["intermediate_size"]
    q, kv = (
        config["num_attention_heads"] * config["head_dim"],
        config["num_key_value_heads"] * config["head_dim"],
    )
    shapes = {
        "model.embed_tokens.weight": [v, h],
        "model.norm.weight": [h],
        "model.early_exit_gate.weight": [1, h],
        "model.early_exit_gate.bias": [1],
        "lm_head.weight": [v, h],
    }
    for layer in range(config["num_hidden_layers"]):
        prefix = f"model.layers.{layer}."
        for name, shape in {
            "self_attn.q_proj": [q, h],
            "self_attn.k_proj": [kv, h],
            "self_attn.v_proj": [kv, h],
            "self_attn.o_proj": [h, q],
            "mlp.gate_proj": [m, h],
            "mlp.up_proj": [m, h],
            "mlp.down_proj": [h, m],
            "input_layernorm": [h],
            "input_layernorm_2": [h],
            "post_attention_layernorm": [h],
            "post_attention_layernorm_2": [h],
        }.items():
            shapes[prefix + name + ".weight"] = shape
    return shapes


def _preparation(plan, worker):
    prep = worker["preparation"]
    _clock(prep["started_ns"], "preparation start", worker["started_ns"], worker["ended_ns"])
    _clock(prep["ended_ns"], "preparation end", prep["started_ns"], worker["ended_ns"])
    weights = prep["shared_weights"]
    equal(weights["all_storage_equal"], True, "shared weights")
    expected = _parameter_shapes(plan["model_config"])
    records = weights["records"]
    require(len(records) == len(expected), "shared parameter record count")
    require(len({r["name"] for r in records}) == len(records), "duplicate parameter records")
    equal(sorted(r["name"] for r in records), sorted(expected), "shared parameter names")
    pointers = set()
    for record in records:
        equal(record["shape"], expected[record["name"]], "shared parameter shape")
        equal(record["numel"], math.prod(record["shape"]), "shared parameter numel")
        equal(
            record["dtype"],
            "torch." + plan["contract"]["engine"]["dtype"],
            "shared parameter dtype",
        )
        equal(record["device"], "cuda:0", "shared parameter device")
        _integer(record["native_data_ptr"], "parameter pointer", minimum=1)
        equal(record["native_data_ptr"], record["official_data_ptr"], "parameter alias")
        require(record["native_data_ptr"] not in pointers, "unexpected tied parameter storage")
        pointers.add(record["native_data_ptr"])
        equal(record["native_requires_grad"], False, "loaded immutable native parameter")
        equal(record["official_requires_grad"], False, "official parameter grad flag")
    equal(
        weights["parameter_count"], sum(math.prod(s) for s in expected.values()), "parameter count"
    )
    pool = prep["native_pool"]
    c = plan["model_config"]
    equal(
        {k: pool[k] for k in ("num_blocks", "block_size", "shape", "bytes")},
        {
            "num_blocks": 1024,
            "block_size": 16,
            "shape": [1024, c["num_hidden_layers"], 16, c["num_key_value_heads"], c["head_dim"]],
            "bytes": plan["resource_estimates"]["native_pool_bytes"],
        },
        "resident native pool",
    )
    for field in ("key_data_ptr", "value_data_ptr"):
        _integer(pool[field], field, minimum=1)
        require(pool[field] not in pointers, "pool aliases a model parameter")
    require(pool["key_data_ptr"] != pool["value_data_ptr"], "K/V pool alias")
    return prep["ended_ns"]


def _controls(plan, worker):
    probe = worker["source_probe"]
    equal(
        probe,
        {key: plan[key] for key in ("source", "imports", "dependencies", "cached_official")},
        "worker actual source/dependencies/imports/cached baseline",
    )
    env, controls = worker["environment"], plan["contract"]["controls"]
    equal(env["cuda_visible_devices"], str(controls["gpu_ids"][0]), "reserved physical GPU")
    equal(canonical_gpu_uuid(env["gpu_uuid"]), controls["gpu_uuid"], "reserved GPU UUID")
    equal(env["logical_device"], "cuda:0", "logical device")
    equal(env["cpu_affinity"], controls["affinity"]["cpu_ids"], "CPU affinity")
    equal(env["numa_status"], controls["affinity"]["numa_status"], "allowed CPU/memory masks")
    equal(env["actual_torch_threads"], {"intraop": 1, "interop": 1}, "thread controls")
    equal(env["arithmetic"], plan["contract"]["arithmetic"], "actual arithmetic controls")
    equal(env["cudnn_allow_tf32"], False, "actual cuDNN TF32 disabled")
    equal(env["active_affinity"], controls["affinity"], "active NUMA policy")
    equal(env["runtime_environment"], plan["runtime_environment"], "all runtime variables")
    scheduler = env["scheduler"]
    require(
        len(scheduler) == 1
        and str(scheduler[0]["gpu_id"]) == env["cuda_visible_devices"]
        and scheduler[0]["type"] == "RUN"
        and scheduler[0]["user"] == env["account"],
        "task-owned scheduler RUN reservation",
    )


def _cached(result, dtype="float32"):
    cached = result["official_cache"]
    ids = result["requests"][0]["token_ids"]
    expected_calls = [
        {
            "output_index": j,
            "input_count": 128 if j == 0 else 1,
            "last_input_position": 127 + j,
            "token_id": ids[j],
        }
        for j in range(64)
    ]
    equal(cached["calls"], expected_calls, "driver actual cached generation calls")
    actual = cached["snapshot"]
    equal(cached["final_summary"], actual["final_summary"], "official summary/snapshot identity")
    expected_actual = [
        {
            "output_index": j,
            "input_count": 128 if j == 0 else 1,
            "position": 0 if j == 0 else 127 + j,
            "cache_length": 128 + j,
        }
        for j in range(64)
    ]
    equal(actual["calls"], expected_actual, "official actual incremental cache calls")
    equal(
        {
            k: actual[k]
            for k in (
                "status",
                "prompt_length",
                "max_outputs",
                "expected_position",
                "output_count",
                "forward_calls",
                "cache_slots",
                "failure",
            )
        },
        {
            "status": "awaiting_completion",
            "prompt_length": 128,
            "max_outputs": 64,
            "expected_position": 191,
            "output_count": 64,
            "forward_calls": 64,
            "cache_slots": 96,
            "failure": None,
        },
        "official final cache lifecycle",
    )
    shape = [1, 16, 191, 128]
    summary = actual["final_summary"]
    equal(
        summary,
        {
            "slot_count": 96,
            "max_cache_size": 96,
            "lengths": [191] * 96,
            "key_shapes": [shape] * 96,
            "value_shapes": [shape] * 96,
            "dtype": "torch." + dtype,
            "device": "cuda:0",
            "distinct_storage": True,
            "all_finite": True if result["run"]["phase"] == "feasibility" else None,
        },
        "actual final 96 independent K/V slots",
    )


def _record(
    plan,
    row,
    result,
    marker,
    completion,
    acknowledged,
    worker,
    previous_ns,
    *,
    failed_after_completion=False,
):
    equal(result["schema_version"], 1, "result schema version")
    equal(result["artifact_type"], "q2_external_result", "result type")
    values = (result, marker, completion) + (() if acknowledged is None else (acknowledged,))
    for value in values:
        equal(value["plan_sha256"], plan["plan_sha256"], "case plan hash")
        equal(value["run"], row, "case planned row")
        equal(value["started_ns"], marker["started_ns"], "case start marker")
        equal(value["deadline_ns"], marker["deadline_ns"], "case deadline marker")
    start = _clock(result["started_ns"], "case start", previous_ns, worker["ended_ns"])
    equal(
        result["deadline_ns"],
        min(worker["deadline_ns"], start + LIMITS["case_timeout_s"] * 10**9),
        "case deadline",
    )
    deadline = result["deadline_ns"]
    arrival = _clock(result["arrival_ns"], "arrival", start, deadline)
    sync = _clock(result["synchronized_ns"], "synchronization", arrival + 1, deadline)
    ended = _clock(result["ended_ns"], "case export start", sync, deadline)
    # A separately recorded post-generation failure can cross the case deadline
    # during serialization/ACK. Validate the original generation, but never make
    # that case timing-eligible or invent a missing successful acknowledgment.
    export_bound = worker["ended_ns"] if failed_after_completion else deadline
    completed = _clock(completion["completed_ns"], "completed serialization", ended, export_bound)
    ack = None
    if acknowledged is not None:
        ack = _clock(
            acknowledged["acknowledged_ns"], "case acknowledgment", completed, export_bound
        )
        _clock(ack, "case within worker", upper=worker["ended_ns"])
    if not failed_after_completion:
        require(ack is not None and ack < deadline, "acknowledgment exhausted case deadline")
    equal(result["status"], "complete", "complete acknowledged case")
    equal(completion["status"], "complete", "completion status")
    equal(result["failures"], [], "complete case failures")
    if plan["schema_version"] == 2 and row["phase"] == "measured":
        cpu = result["cpu_control"]
        equal(cpu, cpu_control_result(cpu["before"], cpu["after"]), "CPU contention reconstruction")
        equal(cpu["passed"], True, "CPU contention screen")
        equal(
            cpu["before"]["affinity"],
            plan["contract"]["controls"]["affinity"]["cpu_ids"],
            "timed CPU affinity",
        )
        _clock(cpu["before"]["time_ns"], "CPU observation start", start, arrival)
        _clock(cpu["after"]["time_ns"], "CPU observation end", sync, ended)
    empty = {"native_requests": 0, "native_used_blocks": 0, "official_cache_slots": 0}
    equal(result["before_request"], empty, "fresh request state")
    equal(result["cleanup"], empty, "request cleanup")
    _clock(result["setup_ns"], "setup interval", upper=arrival - start)
    requests = result["requests"]
    require(len(requests) == 1, "one actual request required")
    request = requests[0]
    equal(request["request_id"], "W1", "request ID")
    equal(request["finished"], True, "completed output history")
    equal(request["finish_reason"], "length", "ignore-EOS output cap")
    require(len(request["token_ids"]) == 64, "64 actual outputs required")
    for token in request["token_ids"]:
        _integer(token, "actual token ID")
        require(token < plan["model_config"]["vocab_size"], "actual token outside vocabulary")
    equal(request["exit_depths"], [4] * 64, "four-loop output history")
    for key in ("admitted_ns", "enqueue_start_ns", "enqueue_end_ns", "first_prefill_ns"):
        equal(request.get(key), None, "common observer does not invent " + key)
    metrics = summarize_records(requests, arrival_ns=arrival, synchronized_ns=sync)
    equal(result["metrics"], metrics, "raw-token metric reconstruction")
    events = result["events"]
    require(len(events) == 65, "64 token events plus one finish event required")
    native = row["implementation_id"] == "native"
    for j, event in enumerate(events):
        equal(event["event_seq"], j, "event sequence")
        equal(event["schema_version"], 1, "event schema")
        equal(event["artifact_type"], "event", "event type")
        equal(event["request_id"], "W1", "event request")
        output = min(j, 63)
        equal(
            event["host_offset_ns"],
            request["token_timestamps_ns"][output] - arrival,
            "event delivery timestamp",
        )
        equal(event["step_id"], 1 + 6 * output if native else output, "actual output step")
        if j < 64:
            equal(
                {k: event[k] for k in ("kind", "output_index", "token_id", "exit_depth")},
                {
                    "kind": "token_emitted",
                    "output_index": j,
                    "token_id": request["token_ids"][j],
                    "exit_depth": 4,
                },
                "token event",
            )
        else:
            equal(
                [event["kind"], event["finish_reason"]],
                ["request_finished", "length"],
                "finish event",
            )
    counts = result["counts"]
    if native:
        stages = ["prefill", "coda"] + [
            "prelude",
            "recurrent",
            "recurrent",
            "recurrent",
            "recurrent",
            "coda",
        ] * 63
        equal(counts["steps"], 380, "native step count")
        equal(
            counts["stage_counts"],
            {"prefill": 1, "coda": 64, "prelude": 63, "recurrent": 252},
            "native stage work",
        )
        equal(
            counts["dispatches"],
            [
                {"step_id": j, "stage": stage, "num_tokens": 128 if j == 0 else 1}
                for j, stage in enumerate(stages)
            ],
            "actual native dispatch work",
        )
        equal(result["official_cache"], None, "native row has no official cache work")
    else:
        equal(counts["steps"], 64, "official call count")
        equal(counts["stage_counts"], {"prefill": 1, "cached_decode": 63}, "official call work")
        equal(
            counts["official_logits_checked"],
            64 if row["phase"] == "feasibility" else 0,
            "official finite-check coverage",
        )
        _cached(result, plan["contract"]["engine"]["dtype"])
    if row["phase"] == "feasibility":
        equal(
            result["feasibility"],
            {"finite_checks": {"recurrent": 256, "coda": 64} if native else {}, "passed": True},
            "finite feasibility coverage",
        )
    else:
        equal(result["feasibility"], None, "excluded finite observers absent in timing")
    memory = result["memory"]
    for key in ("before", "after_requests"):
        _memory(memory[key])
    for key in ("peak_allocated_bytes", "peak_reserved_bytes", "pool_bytes"):
        _integer(memory[key], key)
    equal(
        memory["pool_bytes"], plan["resource_estimates"]["native_pool_bytes"], "resident pool bytes"
    )
    require(memory["peak_reserved_bytes"] >= memory["peak_allocated_bytes"], "peak reserved memory")
    for state in (memory["before"], memory["after_requests"]):
        require(
            state["allocated_bytes"] >= memory["pool_bytes"],
            "native pool missing from resident floor",
        )
        for kind in ("allocated", "reserved"):
            require(
                memory[f"peak_{kind}_bytes"] >= state[f"{kind}_bytes"], "peak below observed memory"
            )
    return {
        "run_id": row["run_id"],
        "run": row,
        "token_ids": request["token_ids"],
        "metrics": {**metrics, "synchronized_tokens_per_second": 64e9 / (sync - arrival)},
        "memory": memory,
        "setup_ns": result["setup_ns"],
        "acknowledged_ns": ack,
        "started_ns": start,
        "ended_ns": ended,
    }


def _pairs(records, repetitions=2):
    by_id = {r["run_id"]: r for r in records}
    pairs = []
    for number in range(1, repetitions + 1):
        n, o = by_id.get(f"N{number}"), by_id.get(f"O{number}")
        row = {"pair_id": f"pair-{number}", "complete": n is not None and o is not None}
        if row["complete"]:
            nv, ov = (
                n["metrics"]["generated_tokens_per_second"],
                o["metrics"]["generated_tokens_per_second"],
            )
            row.update(
                native_tokens_per_s=nv,
                official_tokens_per_s=ov,
                native_over_official=nv / ov,
                native_metrics=n["metrics"],
                official_metrics=o["metrics"],
                native_memory=n["memory"],
                official_memory=o["memory"],
                native_setup_ns=n["setup_ns"],
                official_setup_ns=o["setup_ns"],
            )
        pairs.append(row)
    if not all(p["complete"] for p in pairs):
        return pairs, {"status": "incomplete"}
    native = [p["native_tokens_per_s"] for p in pairs]
    official = [p["official_tokens_per_s"] for p in pairs]
    if repetitions == 1:
        return pairs, {
            "status": "single_observation",
            "interpretation": "one measured request per backend; repeatability not established",
        }
    status = (
        "native_faster"
        if min(native) > max(official)
        else "official_faster"
        if min(official) > max(native)
        else "overlapping"
    )
    return pairs, {
        "status": status,
        "native_range": [min(native), max(native)],
        "official_range": [min(official), max(official)],
        "interpretation": "descriptive two-run ranges; no speed acceptance threshold",
    }


def build_report(output_dir):
    root = Path(output_dir)
    report = {
        "schema_version": 1,
        "artifact_type": "q2_external_report",
        "complete": False,
        "passed": False,
        "evidence_status": "incomplete",
        "decision": "inconclusive",
        "errors": [],
        "missing": [],
        "failures": [],
        "runs": [],
        "retained_failed_runs": [],
        "pairs": [],
    }
    try:
        inventory = _inventory(root)
        report["artifacts"] = inventory
        plan = read_json(root / "plan.json")
        validate_plan(plan)
        report["plan_sha256"] = plan["plan_sha256"]
        if plan["schema_version"] == 2:
            report.update(
                schema_version=2, dtype="bfloat16", expected_runs=4, expected_measured_runs=2
            )
        manifest = read_json(root / "manifest.json")
        expected = [r["run_id"] for r in plan["execution_order"]]
        for value, label in ((manifest, "manifest"),):
            equal(value["plan_sha256"], plan["plan_sha256"], label + " plan")
            equal(
                value["completed_runs"],
                expected[: len(value["completed_runs"])],
                label + " ACK prefix",
            )
        if not (root / "worker.json").exists():
            report["missing"].append("worker.json")
            report["failures"].extend(manifest.get("failures", []))
            return _finish(report)
        worker = read_json(root / "worker.json")
        equal(worker["plan_sha256"], plan["plan_sha256"], "worker plan")
        prefix = worker["completed_runs"]
        equal(prefix, expected[: len(prefix)], "worker ACK prefix")
        parent_prefix = manifest["completed_runs"]
        equal(parent_prefix, prefix[: len(parent_prefix)], "parent acknowledged prefix")
        if len(parent_prefix) != len(prefix):
            report["missing"].append("parent did not acknowledge full worker prefix")
        for value in (manifest, worker):
            require(value["status"] in ("running", "complete", "failed"), "terminal status")
            report["failures"].extend(value.get("failures", []))
        for name in inventory["case_bytes"]:
            require(name in expected, "unplanned case directory")
            require(
                expected.index(name) <= len(prefix),
                "case execution beyond the next unacknowledged planned row",
            )
        # A terminated worker can lack a final record. Its acknowledged rows are
        # still bounded by the controller's observed child return, not a made-up
        # successful worker teardown.
        if "ended_ns" not in manifest:
            report["missing"].append("terminal controller lifetime")
            return _finish(report)
        if "ended_ns" not in worker:
            report["missing"].append("terminal worker lifetime")
            if manifest.get("launch", {}).get("ended_ns") is None:
                return _finish(report)
            worker = {**worker, "ended_ns": manifest["launch"]["ended_ns"]}
        if any(v["status"] == "running" for v in (manifest, worker)):
            report["missing"].append("terminal controller/worker status")
        start = _clock(manifest["started_ns"], "controller start")
        equal(manifest["deadline_ns"], start + LIMITS["total_timeout_s"] * 10**9, "global deadline")
        end = _clock(manifest["ended_ns"], "controller end", start)
        launch = manifest["launch"]
        ls = _clock(launch["started_ns"], "launch start", start, end)
        le = _clock(launch["ended_ns"], "launch end", ls, end)
        ws = _clock(worker["started_ns"], "worker start", ls, le)
        we = _clock(worker["ended_ns"], "worker end", ws, le)
        equal(worker["deadline_ns"], manifest["deadline_ns"], "worker deadline")
        if end > manifest["deadline_ns"] or we > worker["deadline_ns"]:
            report["failures"].append(
                {"type": "TimeoutError", "message": "global deadline exceeded"}
            )
        if "source_probe" not in worker or "environment" not in worker:
            report["missing"].append("worker source/environment controls")
            return _finish(report)
        _controls(plan, worker)
        if "preparation" not in worker or "ended_ns" not in worker["preparation"]:
            report["missing"].append("completed common preparation")
            return _finish(report)
        previous = _preparation(plan, worker)
        by_id = {}
        for row in plan["execution_order"]:
            name = row["run_id"]
            path = root / "runs" / name
            failure_path = path / "failure.json"
            if failure_path.exists():
                late = read_json(failure_path)
                _keys(late, ("run", "plan_sha256", "failure", "occurred_ns"), name="case failure")
                equal(late["run"], row, "failed case row")
                equal(late["plan_sha256"], plan["plan_sha256"], "failed case plan")
                _keys(late["failure"], ("type", "message"), name="case failure details")
                require(
                    all(isinstance(v, str) and v for v in late["failure"].values()),
                    "failure type/message must be nonempty strings",
                )
                require(
                    not any(
                        expected.index(other) > expected.index(name)
                        for other in inventory["case_bytes"]
                    ),
                    "execution continued after a recorded case failure",
                )
                result, marker, completion = (
                    read_json(path / filename)
                    for filename in ("result.json", "started.json", "completed.json")
                )
                ack_path = path / "acknowledged.json"
                ack = read_json(ack_path) if ack_path.exists() else None
                digest = inventory["json_files"][str((path / "result.json").relative_to(root))]
                equal(completion["result"], digest, "failed-tail completed result bytes")
                if ack is not None:
                    equal(ack["result"], digest, "failed-tail acknowledged result bytes")
                    equal(
                        ack["completion"],
                        inventory["json_files"][str((path / "completed.json").relative_to(root))],
                        "ACK completion bytes",
                    )
                item = _record(
                    plan,
                    row,
                    result,
                    marker,
                    completion,
                    ack,
                    worker,
                    previous,
                    failed_after_completion=True,
                )
                lower = max(
                    item["ended_ns"], completion["completed_ns"], item["acknowledged_ns"] or 0
                )
                _clock(late["occurred_ns"], "post-completion failure", lower, worker["ended_ns"])
                item.update(failure=late, comparison_eligible=False, generation_valid=True)
                report["retained_failed_runs"].append(item)
                report["failures"].append(late["failure"])
                report["missing"].append(
                    name + ": complete successful lifetime/ACK not established"
                )
                continue
            if name not in parent_prefix:
                report["missing"].append(name)
                if (path / "result.json").exists():
                    retained = read_json(path / "result.json")
                    equal(retained["run"], row, "retained unacknowledged row")
                    equal(retained["plan_sha256"], plan["plan_sha256"], "retained row plan")
                    if retained.get("status") == "failed":
                        report["failures"].extend(retained.get("failures", []))
                continue
            result, marker, completion, ack = (
                read_json(path / name)
                for name in ("result.json", "started.json", "completed.json", "acknowledged.json")
            )
            digest = inventory["json_files"][str((path / "result.json").relative_to(root))]
            equal(completion["result"], digest, "completed result bytes")
            equal(ack["result"], digest, "acknowledged result bytes")
            equal(
                ack["completion"],
                inventory["json_files"][str((path / "completed.json").relative_to(root))],
                "ACK completion bytes",
            )
            item = _record(plan, row, result, marker, completion, ack, worker, previous)
            previous = item["acknowledged_ns"]
            report["runs"].append(item)
            by_id[name] = item
        n, o = by_id.get("N-feas"), by_id.get("O-feas")
        if n is not None and o is not None:
            same = n["token_ids"] == o["token_ids"]
            report["equivalence"] = {"complete": True, "passed": same}
            if not same and plan["schema_version"] == 1:
                report["failures"].append(
                    {"type": "EquivalenceFailure", "message": "feasibility IDs differ"}
                )
            if same or plan["schema_version"] == 2:
                for name, item in by_id.items():
                    reference = n if item["run"]["implementation_id"] == "native" else o
                    if item["token_ids"] != reference["token_ids"]:
                        report["failures"].append(
                            {
                                "type": "EquivalenceFailure",
                                "message": name + " differs from feasibility",
                            }
                        )
                gate = worker["equivalence"]
                equal(
                    [gate["passed"], gate["token_ids"], gate["exit_depths"]],
                    [same, o["token_ids"], [4] * 64],
                    "recorded actual equivalence",
                )
                _clock(
                    gate["checked_ns"],
                    "equivalence gate",
                    lower=o["ended_ns"],
                    upper=by_id.get(plan["execution_order"][2]["run_id"], {"started_ns": we})[
                        "started_ns"
                    ],
                )
        else:
            report["missing"].append("two completed feasibility outputs")
        cleanup = worker.get("teardown_after_workspace_release")
        if cleanup is None:
            report["missing"].append("final memory cleanup")
        else:
            _memory(cleanup)
            if any(cleanup.values()):
                report["failures"].append(
                    {"type": "CleanupFailure", "message": "worker memory remains"}
                )
        if worker["status"] == "complete" or manifest["status"] == "complete":
            equal(prefix, expected, "complete worker coverage")
            equal(parent_prefix, expected, "complete parent coverage")
            equal(launch["returncode"], 0, "successful worker return")
        report["complete"] = len(report["runs"]) == len(expected) and not report["missing"]
        report["pairs"], report["variation"] = _pairs(
            report["runs"], plan["contract"]["limits"]["measured_executions"] // 2
        )
    except (OSError, ValueError, TypeError, KeyError, StopIteration) as error:
        report["errors"].append(str(error))
    return _finish(report)


def _finish(report):
    report["counts"] = {
        "acknowledged_valid_runs": len(report["runs"]),
        "valid_measured_runs": sum(r["run"]["phase"] == "measured" for r in report["runs"]),
        "validated_generation_runs": len(report["runs"]) + len(report["retained_failed_runs"]),
        "retained_failed_completed_runs": len(report["retained_failed_runs"]),
    }
    if report["errors"]:
        report.update(
            complete=False, evidence_status="invalid", decision="inconclusive", passed=False
        )
    elif report["failures"]:
        report.update(
            evidence_status="complete" if report["complete"] else "incomplete",
            decision="failed",
            passed=False,
        )
    elif report["complete"]:
        report.update(evidence_status="complete", decision="passed", passed=True)
    else:
        report.update(evidence_status="incomplete", decision="inconclusive", passed=False)
    report["scope"] = (
        "BF16 fixed-four cached W1 latency spot check; token agreement is diagnostic, "
        "not an accuracy gate; no repeatability, quality or adaptive claim"
        if report.get("dtype") == "bfloat16"
        else "FP32 fixed-four cached W1 only; no quality/adaptive claim, no required speedup"
    )
    return report


def write_report(output_dir):
    """Write derived reports only; raw case/worker records are never modified."""
    root = Path(output_dir)
    result = build_report(root)
    lines = [
        "# Q2 external "
        + ("BF16" if result.get("dtype") == "bfloat16" else "FP32")
        + " W1 comparison",
        "",
        f"Evidence: **{result['evidence_status']}**; decision: **{result['decision']}**.",
        "",
        result["scope"],
        "",
        f"Valid acknowledged executions: {result['counts']['acknowledged_valid_runs']}/"
        f"{result.get('expected_runs', 8)}; "
        f"measured: {result['counts']['valid_measured_runs']}/"
        f"{result.get('expected_measured_runs', 4)}.",
        "",
    ]
    if result.get("plan_sha256"):
        lines.extend([f"Canonical plan: `{result['plan_sha256']}`.", ""])
    if result["pairs"]:
        lines.extend(
            [
                "| Pair | Native tokens/s | Official tokens/s | Native / official |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for pair in result["pairs"]:
            if pair["complete"]:
                lines.append(
                    f"| {pair['pair_id']} | {pair['native_tokens_per_s']:.6f} | "
                    f"{pair['official_tokens_per_s']:.6f} | {pair['native_over_official']:.6f} |"
                )
            else:
                lines.append(f"| {pair['pair_id']} | incomplete | incomplete | — |")
        lines.extend(
            [
                "",
                "Delivery throughput uses the last actual token return. Final synchronized "
                "throughput, TTFT, TPOT, raw intervals, memory and setup are separate JSON fields.",
                "",
            ]
        )
    for label in ("errors", "failures", "missing"):
        if result[label]:
            lines.extend([label.capitalize() + ":", ""])
            lines.extend(
                "- " + (item if isinstance(item, str) else json.dumps(item, sort_keys=True))
                for item in result[label]
            )
            lines.append("")
    payloads = {
        "q2-external-report.json": (json.dumps(result, indent=2, allow_nan=False) + "\n").encode(),
        "q2-external-report.md": ("\n".join(lines) + "\n").encode(),
    }
    current = sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
    replaced = sum((root / name).stat().st_size for name in payloads if (root / name).exists())
    require(
        current - replaced + sum(map(len, payloads.values())) <= LIMITS["artifact_bytes_max"],
        "derived reports exceed artifact cap",
    )
    for name, payload in payloads.items():
        (root / name).write_bytes(payload)
    return result
