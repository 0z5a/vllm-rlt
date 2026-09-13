"""Build a reproducible M1 report from saved records, without loading a model."""

import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from benchmarks.observe import summarize_records
from benchmarks.schema import (
    validate_plan_integrity,
)
from vllm_lt.validation.common import (
    read_json,
)

_IDENTITY_FIELDS = (
    "run_id",
    "cell_id",
    "workload_id",
    "mode",
    "phase",
    "repetition",
    "pair_id",
    "workload_sha256",
    "controls_sha256",
    "instrumentation",
)
_COMPARISON_METRICS = (
    "delivery_wall_ns",
    "synchronized_wall_ns",
    "generated_tokens_per_second",
    "token_weighted_tpot_ns",
)


def _reject_constant(value):
    raise ValueError(f"nonfinite JSON value: {value}")


def _read_json(path):
    return read_json(path)


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _integer(value):
    return type(value) is int


def _differences(stored, computed, path="metrics"):
    """Integer timestamps/counts are exact; derived float values allow rounding."""
    if isinstance(computed, dict):
        if not isinstance(stored, dict):
            return [f"{path}: expected an object"]
        problems = []
        for key, value in computed.items():
            if key not in stored:
                problems.append(f"{path}.{key}: missing")
            else:
                problems.extend(_differences(stored[key], value, f"{path}.{key}"))
        return problems
    if isinstance(computed, list):
        if not isinstance(stored, list) or len(stored) != len(computed):
            return [f"{path}: list length differs"]
        return [
            problem
            for index, (left, right) in enumerate(zip(stored, computed))
            for problem in _differences(left, right, f"{path}[{index}]")
        ]
    if type(computed) is float:
        equal = (
            type(stored) in (int, float)
            and math.isfinite(stored)
            and math.isclose(stored, computed, rel_tol=1e-9, abs_tol=1e-6)
        )
    else:
        equal = type(stored) is type(computed) and stored == computed
    return [] if equal else [f"{path}: stored value {stored!r} differs from {computed!r}"]


def _read_events(path):
    def object_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON field: {key}")
            result[key] = value
        return result

    def finite_float(value):
        number = float(value)
        if not math.isfinite(number):
            _reject_constant(value)
        return number

    events = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                raise ValueError(f"events.jsonl:{line_number}: empty event")
            event = json.loads(
                line,
                parse_constant=_reject_constant,
                parse_float=finite_float,
                object_pairs_hook=object_pairs,
            )
            if not isinstance(event, dict):
                raise ValueError(f"events.jsonl:{line_number}: expected an object")
            events.append(event)
    return events


def _event_problems(events, result):
    problems = []
    expected = {}
    for request in result["requests"]:
        request_id = request["request_id"]
        for index, (token, depth, timestamp) in enumerate(
            zip(request["token_ids"], request["exit_depths"], request["token_timestamps_ns"])
        ):
            expected[request_id, index] = (token, depth, timestamp)
    observed = {}
    step_times = {}
    request_steps = set()
    previous_seq = None
    for event in events:
        sequence = event.get("event_seq")
        if (
            not _integer(sequence)
            or sequence < 0
            or (previous_seq is not None and sequence != previous_seq + 1)
        ):
            problems.append("event_seq must be consecutive nonnegative integers")
        previous_seq = sequence if _integer(sequence) else previous_seq
        if (
            not _integer(event.get("schema_version"))
            or event["schema_version"] != 1
            or event.get("artifact_type") != "event"
            or event.get("run_id") != result["run_id"]
        ):
            problems.append("event schema or run identity differs")
        offset = event.get("host_offset_ns")
        if not _integer(offset) or offset < 0:
            problems.append("event host_offset_ns must be a nonnegative _integer")
            continue
        if event.get("kind") != "token_emitted":
            continue
        request_id = event.get("request_id")
        index = event.get("output_index")
        step = event.get("step_id")
        if (
            not isinstance(request_id, str)
            or not _integer(index)
            or index < 0
            or not _integer(step)
            or step < 0
            or not _integer(event.get("token_id"))
            or not _integer(event.get("exit_depth"))
        ):
            problems.append("token event requires request_id and _integer output_index/step_id")
            continue
        key = (request_id, index)
        if key in observed:
            problems.append(f"duplicate emitted output {key}")
        if (request_id, step) in request_steps:
            problems.append(f"multiple outputs for request {request_id!r} in step {step}")
        request_steps.add((request_id, step))
        timestamp = result["arrival_ns"] + offset
        observed[key] = (event.get("token_id"), event.get("exit_depth"), timestamp)
        if step in step_times and step_times[step] != timestamp:
            problems.append(f"step {step} has inconsistent output timestamps")
        step_times[step] = timestamp
    if observed != expected:
        problems.append("emitted events do not match raw request token/depth/timestamp histories")
    return problems


def _workload(plan, workload_id):
    for workload in plan.get("suite", {}).get("workloads", []):
        if workload.get("workload_id") == workload_id:
            return workload
    return None


def _completion_problems(result, plan):
    problems = []
    requests = result["requests"]
    if not requests or any(request.get("finished") is not True for request in requests):
        problems.append("complete run contains unfinished or missing requests")
    workload = _workload(plan, result["workload_id"])
    if workload is None:
        return problems + ["workload is absent from the frozen suite"]
    fixtures = {request["request_id"]: request for request in workload["requests"]}
    counts = result.get("counts", {})
    work = counts.get("request_work", {})
    totals = Counter()
    if set(work) != set(fixtures):
        problems.append("per-request work identities differ from the frozen workload")
    if {request["request_id"] for request in requests} != set(fixtures):
        problems.append("request identities differ from the frozen workload")
    for request in requests:
        fixture = fixtures.get(request["request_id"])
        if fixture and len(request["token_ids"]) != fixture["max_output_tokens"]:
            problems.append(f"request {request['request_id']!r} output count differs from fixture")
        depths = request["exit_depths"]
        if fixture is not None:
            prompt_length = len(fixture["prompt_token_ids"])
            expected_work = {
                "prefill": prompt_length,
                "prelude": fixture["max_output_tokens"] - 1,
                "recurrent": sum(depths[1:]),
                "coda": fixture["max_output_tokens"],
                "prefill_token_traversals": 4 * prompt_length,
            }
            totals.update(expected_work)
            observed_work = work.get(request["request_id"], {})
            for stage, expected in expected_work.items():
                actual = observed_work.get(stage, 0)
                if not _integer(actual) or actual != expected:
                    problems.append(f"request {request['request_id']!r} {stage} work differs")
        if depths and depths[0] != 4:
            problems.append("first output must come from depth-four prefill")
        if workload.get("kind") == "fixed_depth_generation" and any(d != 4 for d in depths):
            problems.append("fixed-depth generation contains a non-four exit depth")
        if workload.get("kind") == "scheduler_replay":
            replay = workload.get("replay", {}).get(request["request_id"])
            if replay is None or (
                request["token_ids"] != replay["output_token_ids"]
                or depths != replay["exit_depths"]
            ):
                problems.append("raw replay history differs from the frozen token/depth trace")
    for stage in ("prefill", "prelude", "recurrent", "coda"):
        actual = counts.get("stage_tokens", {}).get(stage, 0)
        if not _integer(actual) or actual != totals[stage]:
            problems.append(f"stage_tokens.{stage} differs from expected executed work")
    probabilities = counts.get("gate_probabilities")
    if (
        not isinstance(probabilities, list)
        or len(probabilities) != totals["recurrent"]
        or any(
            type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1
            for value in probabilities
        )
    ):
        problems.append("actual finite gate readback count must equal recurrent token traversals")
    if result["phase"] == "feasibility":
        checks = result.get("feasibility_finite_checks", {})
        for stage in ("recurrent", "coda"):
            if not _integer(checks.get(stage)) or checks[stage] <= 0:
                problems.append(f"feasibility is missing positive {stage} finite-check counts")
    return problems


def _run_record(run_dir, specification, plan, hashes):
    run_id = specification["run_id"]
    folder = run_dir / "runs" / run_id
    record = {
        "run_id": run_id,
        "planned": specification,
        "status": "incomplete",
        "comparison_eligible": False,
        "validation_errors": [],
        "failures": [],
        "started": (folder / "started.json").is_file(),
        "result": None,
        "recomputed_metrics": None,
    }
    for name in ("started.json", "events.jsonl", "result.json"):
        path = folder / name
        if path.is_file():
            hashes[str(path.relative_to(run_dir))] = _sha256(path)
    result_path = folder / "result.json"
    if not result_path.is_file():
        record["validation_errors"].append(
            "run started but result is missing"
            if record["started"]
            else "planned run did not start"
        )
        return record
    try:
        result = _read_json(result_path)
        if not isinstance(result, dict):
            raise ValueError("result.json must contain an object")
        record["result"] = result
        if result.get("schema_version") != 1 or result.get("artifact_type") != "run_result":
            raise ValueError("unsupported result schema")
        status = result.get("status")
        if status not in {"complete", "failed", "incomplete"}:
            raise ValueError("unsupported run status")
        record["status"] = status
        record["failures"] = result.get("failures", [])
        for field in _IDENTITY_FIELDS:
            if field not in result or result[field] != specification.get(field):
                record["validation_errors"].append(f"{field} differs from the frozen plan")
        if result.get("plan_sha256") != plan["plan_sha256"]:
            record["validation_errors"].append("result plan_sha256 differs from the frozen plan")
        if not record["started"]:
            record["validation_errors"].append("run result has no started.json marker")
        if result.get("synchronized_ns") is None:
            raise ValueError("final synchronization boundary is unavailable")
        computed = summarize_records(
            result["requests"],
            arrival_ns=result["arrival_ns"],
            synchronized_ns=result["synchronized_ns"],
        )
        record["recomputed_metrics"] = computed
        record["validation_errors"].extend(_differences(result.get("metrics"), computed))
        events = _read_events(folder / "events.jsonl")
        record["validation_errors"].extend(_event_problems(events, result))
        if status == "complete":
            record["validation_errors"].extend(_completion_problems(result, plan))
            cleanup = result.get("cleanup", {})
            for field in ("active_requests", "used_kv_blocks"):
                if not _integer(cleanup.get(field)) or cleanup[field] != 0:
                    record["validation_errors"].append(f"cleanup.{field} must be zero")
            if record["failures"]:
                record["validation_errors"].append("complete result records failures")
        record["comparison_eligible"] = (
            status == "complete"
            and result.get("comparison_eligible") is True
            and specification["phase"] == "measured"
            and not record["validation_errors"]
        )
    except (OSError, ValueError, KeyError, TypeError) as error:
        record["validation_errors"].append(str(error))
    return record


def _ranges(values):
    return {
        metric: {"values": observations, "min": min(observations), "max": max(observations)}
        for metric in _COMPARISON_METRICS
        if (observations := [row[metric] for row in values if row.get(metric) is not None])
    }


def _comparisons(records):
    comparisons = []
    for workload_id in ("W4", "W5"):
        rows = [
            record
            for record in records
            if record["planned"]["workload_id"] == workload_id
            and record["planned"]["phase"] == "measured"
        ]
        if not rows:
            continue
        order_valid = [row["planned"]["mode"] for row in rows] == [
            "refill",
            "no_refill",
            "no_refill",
            "refill",
        ]
        arrivals = [
            row["result"].get("arrival_ns")
            for row in rows
            if row["result"] is not None and _integer(row["result"].get("arrival_ns"))
        ]
        if any(left >= right for left, right in zip(arrivals, arrivals[1:])):
            order_valid = False
        pairs = defaultdict(list)
        for row in rows:
            pairs[row["planned"].get("pair_id")].append(row)
        pair_reports = []
        accepted = {"refill": [], "no_refill": []}
        for pair_id, members in pairs.items():
            reasons = []
            if not order_valid:
                reasons.append(
                    "planned or observed measured run order does not satisfy A1/B1/B2/A2"
                )
            if (
                pair_id is None
                or len(members) != 2
                or {member["planned"]["mode"] for member in members} != {"refill", "no_refill"}
            ):
                reasons.append("pair must contain one planned measured run per mode")
            for field in ("controls_sha256", "workload_sha256", "instrumentation", "repetition"):
                first = members[0]["planned"].get(field)
                if first is None or any(
                    member["planned"].get(field) != first for member in members
                ):
                    reasons.append(f"paired {field} values differ or are missing")
            for member in members:
                if not member["comparison_eligible"]:
                    reasons.append(f"{member['run_id']} is not a valid complete measured run")
            raw = {
                member["planned"]["mode"]: {
                    "run_id": member["run_id"],
                    "status": member["status"],
                    "metrics": member["recomputed_metrics"],
                }
                for member in members
            }
            ratios = {}
            if not reasons:
                for mode in accepted:
                    accepted[mode].append(raw[mode]["metrics"])
                for metric in _COMPARISON_METRICS:
                    numerator = raw["refill"]["metrics"].get(metric)
                    denominator = raw["no_refill"]["metrics"].get(metric)
                    ratios[metric] = (
                        numerator / denominator
                        if numerator is not None and denominator is not None and denominator > 0
                        else None
                    )
            pair_reports.append(
                {
                    "pair_id": pair_id,
                    "status": "complete" if not reasons else "excluded",
                    "reasons": reasons,
                    "observations": raw,
                    "refill_over_no_refill": ratios,
                }
            )
        comparisons.append(
            {
                "workload_id": workload_id,
                "workload_kind": "scheduler_replay",
                "pairs": pair_reports,
                "complete_pairs": sum(pair["status"] == "complete" for pair in pair_reports),
                "expected_pairs": 2,
                "mode_ranges": {mode: _ranges(values) for mode, values in accepted.items()},
                "interpretation": (
                    "Descriptive synthetic replay measurements only. Two runs do not establish "
                    "statistical significance, live-gate quality, or a general serving advantage."
                ),
            }
        )
    return comparisons


def _cells(records, comparisons):
    complete_pair_ids = {
        pair["pair_id"]
        for comparison in comparisons
        for pair in comparison["pairs"]
        if pair["status"] == "complete"
    }
    grouped = defaultdict(list)
    for record in records:
        if record["planned"]["phase"] == "measured":
            grouped[record["planned"]["cell_id"]].append(record)
    cells = []
    for cell_id, rows in grouped.items():
        eligible = [
            row["recomputed_metrics"]
            for row in rows
            if row["comparison_eligible"]
            and (
                row["planned"]["pair_id"] is None or row["planned"]["pair_id"] in complete_pair_ids
            )
        ]
        cells.append(
            {
                "cell_id": cell_id,
                "workload_id": rows[0]["planned"]["workload_id"],
                "mode": rows[0]["planned"]["mode"],
                "run_ids": [row["run_id"] for row in rows],
                "included_observations": len(eligible),
                "measured_ranges": _ranges(eligible),
            }
        )
    return cells


def _profiles(run_dir, hashes, records, plan):
    profiles = []
    folder = run_dir / "profiles"
    if not folder.is_dir():
        return profiles
    for capture in sorted(path for path in folder.iterdir() if path.is_dir()):
        item = {
            "capture_id": capture.name,
            "metadata": None,
            "limitations": [],
            "validation_errors": [],
        }
        for name in ("metadata.json", "trace.json"):
            path = capture / name
            if path.is_file():
                hashes[str(path.relative_to(run_dir))] = _sha256(path)
        try:
            item["metadata"] = _read_json(capture / "metadata.json")
            metadata = item["metadata"]
            if (
                metadata.get("schema_version") != 1
                or metadata.get("artifact_type") != "profile_metadata"
            ):
                item["validation_errors"].append("unsupported profile metadata schema")
            run = next((row for row in records if row["run_id"] == capture.name), None)
            if run is None or run["planned"]["phase"] != "profile":
                item["validation_errors"].append("capture does not identify a planned profile run")
            else:
                for field in (
                    "cell_id",
                    "workload_id",
                    "mode",
                    "controls_sha256",
                    "workload_sha256",
                ):
                    if metadata.get(field) != run["planned"][field]:
                        item["validation_errors"].append(f"profile {field} differs from plan")
            start, end = metadata.get("start_step"), metadata.get("end_step")
            if (
                metadata.get("capture_id") != capture.name
                or metadata.get("complete_window") is not True
                or not _integer(start)
                or not _integer(end)
                or not 0 <= start <= end
            ):
                item["validation_errors"].append("capture identity/window is incomplete or invalid")
            elif run is not None and run["result"] is not None:
                emitted = _read_events(run_dir / "runs" / capture.name / "events.jsonl")
                observed = sum(
                    event.get("kind") == "token_emitted"
                    and event.get("output_index", 0) > 0
                    and start <= event.get("step_id", -1) <= end
                    for event in emitted
                )
                limit = plan.get("contract", {}).get("limits", {}).get("profile_decode_outputs", 16)
                if observed != metadata.get("subsequent_outputs") or observed < limit:
                    item["validation_errors"].append(
                        "capture output window disagrees with raw events"
                    )
                snapshots = run["result"].get("counts", {}).get("snapshots", [])
                captured = [
                    row["batch"] for row in snapshots if start <= row["batch"]["step_id"] <= end
                ]
                item["stage_dispatches"] = dict(Counter(row["stage"] for row in captured))
                item["interleaved_prefill_tokens"] = (
                    sum(
                        token["token_count"]
                        for row in captured
                        if row["stage"] == "prefill"
                        for token in row["rows"]
                    )
                    if captured
                    else None
                )
                if not captured:
                    item["limitations"].append(
                        "Stage snapshots unavailable for exact prefill coverage."
                    )
            trace = _read_json(capture / "trace.json")
            events = trace.get("traceEvents", []) if isinstance(trace, dict) else trace
            groups = {
                "cpu_inclusive_scopes": defaultdict(lambda: [0, 0.0]),
                "benchmark_overhead": defaultdict(lambda: [0, 0.0]),
                "gpu_kernels": defaultdict(lambda: [0, 0.0]),
                "gpu_memcpy": defaultdict(lambda: [0, 0.0]),
            }
            for event in events:
                if event.get("ph") != "X":
                    continue
                duration = event.get("dur")
                if (
                    type(duration) not in (int, float)
                    or not math.isfinite(duration)
                    or duration < 0
                ):
                    raise ValueError("trace duration must be finite and nonnegative")
                category = event.get("cat", "").lower()
                target = (
                    "gpu_kernels"
                    if category == "kernel"
                    else "gpu_memcpy"
                    if "memcpy" in category
                    else "benchmark_overhead"
                    if category == "user_annotation"
                    and event.get("name", "").startswith("vllm_lt::observer_")
                    else "cpu_inclusive_scopes"
                    if category == "user_annotation"
                    else None
                )
                if target:
                    aggregate = groups[target][event.get("name", "unnamed")]
                    aggregate[0] += 1
                    aggregate[1] += duration
            for target, values in groups.items():
                item[target] = sorted(
                    (
                        {"name": name, "count": value[0], "total_us": value[1]}
                        for name, value in values.items()
                    ),
                    key=lambda row: row["total_us"],
                    reverse=True,
                )
            item["gpu_trace_available"] = bool(item["gpu_kernels"] or item["gpu_memcpy"])
            if not item["gpu_kernels"]:
                item["limitations"].append("No actual GPU kernel events; GPU cost is unavailable.")
            if not item["gpu_memcpy"]:
                item["limitations"].append(
                    "No actual GPU memcpy events; transfer attribution is unavailable."
                )
            if not item["cpu_inclusive_scopes"]:
                item["limitations"].append(
                    "No labeled CPU user scopes; stage attribution is unavailable."
                )
            item["limitations"].append(
                "CPU scopes are inclusive and may overlap; do not add them or equate them with "
                "metadata time. GPU event sums are not end-to-end time or device busy-time unions. "
                "The bounded decode window may contain interleaved prefill. Observer scans are "
                "benchmark overhead, not production optimization targets; enclosing scopes "
                "include them."
            )
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
            item["gpu_trace_available"] = False
            item["validation_errors"].append(f"Capture could not be summarized: {error}")
        profiles.append(item)
    return profiles


def _next_action(profiles):
    for profile in profiles:
        kernels = profile.get("gpu_kernels", [])
        if not kernels or profile["validation_errors"]:
            continue
        largest = kernels[0]
        evidence = (
            f"Kernel-time ranking only: capture {profile['capture_id']} records "
            f"{largest['name']!r} as its largest "
            f"GPU kernel group by aggregate duration ({largest['total_us']:.6g} microseconds). "
        )
        return evidence + (
            "This ranking does not nominate the next optimization. Correlate this kernel "
            "group with the captured stage and nested "
            "CPU operations to distinguish compute cost from dispatch/readback gaps before "
            "selecting a runtime change. Use a separately declared capture budget if the saved "
            "trace lacks that correlation; observer scans are excluded from production candidates."
        )
    return (
        "Obtain the missing planned CPU/CUDA profiler capture with actual GPU events "
        "and a verified window before ranking bottlenecks or selecting an optimization."
    )


def _display(value):
    if value is None:
        return "unavailable"
    return f"{value:.6g}" if isinstance(value, float) else str(value)


def _markdown(report):
    lines = [
        "# M1 benchmark report",
        "",
        f"Coverage status: **{report['status']}**. This is a baseline measurement report; "
        "no optimization target or statistical win is inferred.",
        "",
        f"Frozen experiment: [manifest](<{report['run_dir']}/manifest.json>); "
        "its plan identity and source provenance are retained in report.json.",
        "",
        "## Planned run inventory",
        "",
        "Only valid, complete measured pairs enter mode comparisons. Failed, incomplete, "
        "and excluded observations remain visible below.",
        "",
        "| Run | Cell | Phase | Status | Delivery ms | Output tokens/s | Eligible |",
        "| --- | --- | --- | --- | ---: | ---: | --- |",
    ]
    for run in report["runs"]:
        metrics = run["recomputed_metrics"] or {}
        delivery = metrics.get("delivery_wall_ns")
        lines.append(
            f"| {run['run_id']} | {run['planned']['cell_id']} | {run['planned']['phase']} | "
            f"{run['status']} | {_display(delivery / 1e6 if delivery is not None else None)} | "
            f"{_display(metrics.get('generated_tokens_per_second'))} | "
            f"{run['comparison_eligible']} |"
        )
    for run in report["runs"]:
        if run["validation_errors"] or run["failures"]:
            lines.extend(
                [
                    "",
                    f"**{run['run_id']}**: "
                    + "; ".join(str(value) for value in run["validation_errors"] + run["failures"]),
                ]
            )
    lines.extend(
        [
            "",
            "## Measured cell observations",
            "",
            "Invalid runs and unmatched pairs are excluded from these ranges.",
            "",
            "| Cell | Metric | Raw values | Min | Max |",
            "| --- | --- | --- | ---: | ---: |",
        ]
    )
    for cell in report["cells"]:
        for metric, values in cell["measured_ranges"].items():
            lines.append(
                f"| {cell['cell_id']} | {metric} | "
                f"{', '.join(map(_display, values['values']))} | "
                f"{_display(values['min'])} | {_display(values['max'])} |"
            )
    lines.extend(["", "## Paired synthetic replay observations", ""])
    for comparison in report["comparisons"]:
        lines.extend(
            [
                f"### {comparison['workload_id']}",
                "",
                f"Complete pairs: {comparison['complete_pairs']}/{comparison['expected_pairs']}.",
                "",
                comparison["interpretation"],
                "",
            ]
        )
        for pair in comparison["pairs"]:
            lines.append(
                f"- {pair['pair_id']}: {pair['status']}"
                + ("; " + "; ".join(pair["reasons"]) if pair["reasons"] else "")
            )
        lines.extend(
            ["", "| Mode | Metric | Raw values | Min | Max |", "| --- | --- | --- | ---: | ---: |"]
        )
        for mode, metrics in comparison["mode_ranges"].items():
            for name, values in metrics.items():
                lines.append(
                    f"| {mode} | {name} | {', '.join(map(_display, values['values']))} | "
                    f"{_display(values['min'])} | {_display(values['max'])} |"
                )
    lines.extend(["", "## Bounded profiler evidence", ""])
    if not report["profiles"]:
        lines.append("No profiler artifacts were available; no hotspot ranking can be supported.")
    for profile in report["profiles"]:
        lines.extend(["", f"### {profile['capture_id']}", ""])
        path = f"{report['run_dir']}/profiles/{profile['capture_id']}"
        lines.append(
            f"Evidence: [trace](<{path}/trace.json>), [capture metadata](<{path}/metadata.json>)."
        )
        metadata = profile.get("metadata") or {}
        lines.extend(
            [
                "",
                f"Window: steps {_display(metadata.get('start_step'))}–"
                f"{_display(metadata.get('end_step'))}; subsequent outputs: "
                f"{_display(metadata.get('subsequent_outputs'))}; interleaved prefill tokens: "
                f"{_display(profile.get('interleaved_prefill_tokens'))}.",
            ]
        )
        for family in ("cpu_inclusive_scopes", "benchmark_overhead", "gpu_kernels", "gpu_memcpy"):
            rows = profile.get(family, [])
            if rows:
                lines.extend(
                    [
                        "",
                        f"**{family}**, ranked within this family:",
                        "",
                        "| Scope/event | Count | Total μs |",
                        "| --- | ---: | ---: |",
                    ]
                )
                for row in rows[:10]:
                    name = row["name"].replace("|", "\\|").replace("\n", " ")
                    lines.append(f"| {name} | {row['count']} | {_display(row['total_us'])} |")
        lines.extend(
            ["", *[f"- {text}" for text in (profile["validation_errors"] + profile["limitations"])]]
        )
    lines.extend(["", "## Next measurement and limits", "", report["next_action"], ""])
    lines.extend(f"- {limitation}" for limitation in report["limitations"])
    lines.extend(
        [
            "",
            "Raw per-request metrics, memory, counters, failures, frozen provenance, "
            "and input artifact SHA-256 hashes are retained in report.json.",
            "",
        ]
    )
    return "\n".join(lines)


def build_report(run_dir: Path, output_dir: Path) -> dict:
    """Recompute and report a saved experiment; never overwrite output artifacts."""
    run_dir, output_dir = Path(run_dir), Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"report output already exists: {output_dir}")
    manifest_path = run_dir / "manifest.json"
    plan_path = run_dir / "plan.json"
    if not plan_path.is_file():
        plan_path = run_dir / "inputs" / "plan.json"
    manifest, plan = _read_json(manifest_path), _read_json(plan_path)
    for name, value in (("manifest", manifest), ("plan", plan)):
        if not isinstance(value, dict) or value.get("schema_version") != 1:
            raise ValueError(f"unsupported {name} schema")
    validate_plan_integrity(plan)
    if manifest.get("plan_sha256") != plan["plan_sha256"]:
        raise ValueError("experiment manifest does not identify the saved plan hash")
    order = plan.get("execution_order")
    if not isinstance(order, list) or not order:
        raise ValueError("plan requires a nonempty execution_order")
    ids = []
    for entry in order:
        run_id = entry.get("run_id")
        if (
            not isinstance(run_id, str)
            or not run_id
            or Path(run_id).name != run_id
            or run_id in {".", ".."}
        ):
            raise ValueError("run_id must be a nonempty directory basename")
        if any(field not in entry for field in _IDENTITY_FIELDS):
            raise ValueError(f"incomplete plan identity for {run_id}")
        ids.append(run_id)
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate planned run IDs")
    hashes = {
        "manifest.json": _sha256(manifest_path),
        str(plan_path.relative_to(run_dir)): _sha256(plan_path),
    }
    records = [_run_record(run_dir, entry, plan, hashes) for entry in order]
    profiles = _profiles(run_dir, hashes, records, plan)
    unexpected = (
        sorted(
            path.name
            for path in (run_dir / "runs").iterdir()
            if path.is_dir() and path.name not in ids
        )
        if (run_dir / "runs").is_dir()
        else []
    )
    comparisons = _comparisons(records)
    complete = (
        all(row["status"] == "complete" and not row["validation_errors"] for row in records)
        and not unexpected
    )
    phase_counts = dict(Counter(entry["phase"] for entry in order))
    matrix_complete = phase_counts == {"feasibility": 7, "warmup": 7, "measured": 14, "profile": 2}
    complete = complete and matrix_complete
    complete = complete and manifest.get("status") == "complete" and not manifest.get("failures")
    complete = complete and all(item["complete_pairs"] == 2 for item in comparisons)
    planned_profiles = sum(entry["phase"] == "profile" for entry in order)
    profile_coverage = len(profiles) == planned_profiles and all(
        profile.get("gpu_kernels")
        and profile.get("gpu_memcpy")
        and profile.get("cpu_inclusive_scopes")
        and not profile["validation_errors"]
        for profile in profiles
    )
    complete = complete and profile_coverage
    report = {
        "schema_version": 1,
        "artifact_type": "benchmark_report",
        "run_dir": str(run_dir.resolve()),
        "status": "complete" if complete else "partial",
        "manifest": manifest,
        "plan_sha256": plan.get("plan_sha256"),
        "source": plan.get("source"),
        "input_artifact_sha256": hashes,
        "runs": records,
        "cells": _cells(records, comparisons),
        "comparisons": comparisons,
        "profiles": profiles,
        "coverage": {
            "planned_runs": len(order),
            "planned_phase_counts": phase_counts,
            "m1_matrix_complete": matrix_complete,
            "complete_valid_runs": sum(
                row["status"] == "complete" and not row["validation_errors"] for row in records
            ),
            "eligible_measured_runs": sum(row["comparison_eligible"] for row in records),
            "planned_profiles": planned_profiles,
            "observed_profiles": len(profiles),
            "profile_coverage_complete": bool(profile_coverage),
            "unplanned_run_directories": unexpected,
        },
        "next_action": _next_action(profiles),
        "limitations": [
            "Feasibility, warmup, and profiling are excluded from measured comparisons.",
            "Partial/failed pairs remain visible and are excluded; no observations are imputed.",
            "Two measured runs are a bounded descriptive screen, not statistical significance.",
            "W4 and W5 are separate synthetic replay workloads; their depth traces differ.",
            "Delivery throughput includes enqueue, queueing, prefill, and decode.",
            "Small request distributions do not establish stable serving-tail percentiles.",
            "Request/KV cleanup is distinct from persistent model/pool/allocator memory.",
        ],
    }
    if manifest.get("status") != "complete" or manifest.get("failures"):
        report["limitations"].append(
            f"Experiment manifest status is {manifest.get('status', 'unavailable')}; "
            f"recorded failures: {manifest.get('failures', [])}."
        )
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (output_dir / "report.md").write_text(_markdown(report), encoding="utf-8")
    return report
