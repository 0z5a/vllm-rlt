"""Six fixed four-row/width32 kernel checks, independent of graph construction."""

import time
from copy import deepcopy
from functools import lru_cache
from pathlib import Path

from vllm_lt.validation import m3_inactive_kernels as base
from vllm_lt.validation.m3_persistent_lifecycle import _persist_result, _require

_unsettled = []


@lru_cache(maxsize=1)
def _resolved_plan():
    prior = base.build_kernel_plan()
    layouts = [
        {
            "layout_id": name,
            "row_count": 4,
            "table_width": 32,
            "live_rows": live,
            "lengths": lengths,
        }
        for name, live, lengths in (
            ("K4-zero", [], []),
            ("K4-one-limit", [1], [512]),
            ("K4-two-crossing", [1, 3], [16, 17]),
        )
    ]
    plan = {
        "schema_version": 1,
        "artifact_type": "m3_capture_kernel_plan",
        "config": prior["config"],
        "layouts": layouts,
        "execution_order": [
            {
                "evaluation_id": f"K-{side}-triton-{layout['layout_id']}",
                "implementation_id": side,
                "backend": "triton",
                "layout_id": layout["layout_id"],
            }
            for side in ("A", "B")
            for layout in layouts
        ],
        "checker": {**prior["checker"], "intended_layout_ids": [x["layout_id"] for x in layouts]},
        "resource_estimates": {
            "evaluations": 6,
            "checker_evaluations": 0,
            "cache_bytes": 80 * 1024**2,
            "cache_chunks_per_evaluation": 40,
            "output_payload_bytes": 122880,
            "output_index_records": 6,
            "max_output_tensor_bytes": 32768,
            "auxiliary_evidence_bytes_upper_bound": 1024**2,
            "artifact_bytes_upper_bound": 256 * 1024**2,
        },
        "input_identities": {},
    }
    for layout in layouts:
        identities = {}
        for side in ("A", "B"):
            _, _, physical, logical, _ = base._fixture(plan, layout, side)
            identities[side] = {"input_hashes": physical, "logical_input_hashes": logical}
        _require(
            identities["A"]["logical_input_hashes"] == identities["B"]["logical_input_hashes"],
            "compact and padded logical inputs differ",
        )
        plan["input_identities"][layout["layout_id"]] = identities
    plan["kernel_plan_sha256"] = base._digest(plan)
    return plan


def build_kernel_plan():
    return deepcopy(_resolved_plan())


def validate_kernel_plan(plan):
    _require(base._digest(plan) == base._digest(_resolved_plan()), "capture kernel matrix differs")


def _device_kernels(device):
    if device.type != "cuda":
        raise ValueError("declared Triton evaluation requires its reserved CUDA device")
    from vllm_lt.kernels.paged_attention import triton_paged_attention
    from vllm_lt.kernels.triton_kv_write import masked_kv_write

    return triton_paged_attention, masked_kv_write


def _chunk_records(plan, layout):
    rows = []
    for component in ("key", "value"):
        for start in range(0, 160, 8):
            tensor = base._cache_chunk(plan, layout, component, start, start + 8, written=True)
            digest = base._tensor_hash(tensor)
            rows.append(
                {
                    "component": component,
                    "block_start": start,
                    "block_stop": start + 8,
                    "shape": list(tensor.shape),
                    "dtype": "float32",
                    "size_bytes": tensor.numel() * 4,
                    "actual_sha256": digest,
                    "expected_sha256": digest,
                }
            )
    return rows


def _valid_output(output, live, shape):
    stats = base._output_stats(output, live)
    _require(
        stats["shape"] == shape
        and stats["dtype"] == "float32"
        and stats["finite_count"] == stats["numel"]
        and stats["inactive_zero_count"] == stats["inactive_numel"]
        and stats["inactive_negative_zero_count"] == 0,
        "kernel raw output violates shape, finiteness or positive-zero padding",
    )
    return stats


def run_kernel_evaluation(plan, evaluation_id, output_dir, *, device, deadline_ns):
    started = time.perf_counter_ns()
    import gc
    import weakref

    import torch

    from vllm_lt.validation.diagnostics import SpoolBudget, TensorSpool

    validate_kernel_plan(plan)
    matches = [x for x in plan["execution_order"] if x["evaluation_id"] == evaluation_id]
    _require(len(matches) == 1, "unknown capture kernel evaluation")
    row = matches[0]
    layout = next(x for x in plan["layouts"] if x["layout_id"] == row["layout_id"])
    root, target = Path(output_dir), torch.device(device)
    folder = root / "evaluations" / evaluation_id
    folder.mkdir(parents=True, exist_ok=False)
    deadline = min(deadline_ns, started + 600 * 10**9)
    marker = {
        "evaluation": row,
        "started_ns": started,
        "deadline_ns": deadline,
        "kernel_plan_sha256": plan["kernel_plan_sha256"],
    }
    base._write(folder / "started.json", marker)
    result = {
        "schema_version": 1,
        "artifact_type": "m3_capture_kernel_result",
        **marker,
        "device": str(target),
        "status": "incomplete",
        "passed": False,
        "errors": [],
        "cache_chunks": [],
        "completion_confirmed": False,
        "cache_references_released": False,
    }
    pool, tensors, refs = {}, {}, []
    spool = output = host = args = actual = reference = None

    def check():
        if time.perf_counter_ns() >= deadline:
            raise TimeoutError("capture kernel case lifetime exceeded")

    try:
        check()
        attention, scatter = _device_kernels(target)
        cpu, live, hashes, logical, _ = base._fixture(plan, layout, row["implementation_id"])
        _require(
            plan["input_identities"][layout["layout_id"]][row["implementation_id"]]
            == {"input_hashes": hashes, "logical_input_hashes": logical},
            "frozen kernel inputs differ",
        )
        result.update(input_hashes=hashes, logical_input_hashes=logical)
        tensors = {k: v.to(target) for k, v in cpu.items()}
        del cpu
        pool = {
            name: torch.empty((160, 2, 16, 16, 128), dtype=torch.float32, device=target)
            for name in ("key", "value")
        }
        refs = [weakref.ref(tensor) for tensor in pool.values()]
        target = pool["key"].device
        result["device"] = str(target)
        for component in pool:
            for start in range(0, 160, 8):
                check()
                pool[component][start : start + 8].copy_(
                    base._cache_chunk(plan, layout, component, start, start + 8, written=False)
                )
        if row["implementation_id"] == "B":
            scatter(
                pool["key"][:, 1],
                pool["value"][:, 1],
                tensors["write_blocks"],
                tensors["write_offsets"],
                tensors["key"],
                tensors["value"],
                tensors["active"],
            )
        elif live:
            pool["key"][tensors["write_blocks"], 1, tensors["write_offsets"]] = tensors["key"]
            pool["value"][tensors["write_blocks"], 1, tensors["write_offsets"]] = tensors["value"]
        args = (
            tensors["query"],
            pool["key"][:, 1],
            pool["value"][:, 1],
            tensors["block_tables"],
            tensors["context_lengths"],
        )
        output = (
            attention(*args)
            if row["implementation_id"] == "A"
            else attention(*args, tensors["active"])
        )
        host = output.to("cpu")
        count = len(live) if row["implementation_id"] == "A" else 4
        result["output_stats"] = base._output_stats(host, live)
        result["output_sha256"] = base._tensor_hash(host)
        spool = TensorSpool(
            root / "outputs",
            evaluation_id,
            layout["layout_id"],
            SpoolBudget(1024**2, 1024**2),
            evaluation_id,
        )
        spool.write(
            "attention_output",
            host,
            {"evaluation_id": evaluation_id, "layout_id": layout["layout_id"]},
        )
        spool.close()
        _valid_output(host, live, [count, 16, 128])
        for expected in _chunk_records(plan, layout):
            check()
            actual = pool[expected["component"]][
                expected["block_start"] : expected["block_stop"]
            ].to("cpu")
            observed = {**expected, "actual_sha256": base._tensor_hash(actual)}
            result["cache_chunks"].append(observed)
            _require(observed == expected, "whole-pool active/neighbor/guard bytes differ")
        if row["implementation_id"] == "B":
            prior_id = evaluation_id.replace("K-B-", "K-A-", 1)
            prior = base._read(root / "evaluations" / prior_id / "result.json")
            _require(
                prior["passed"] and prior["status"] == "complete",
                "successful A kernel reference missing",
            )
            reference = TensorSpool.open(root / "outputs", prior_id, layout["layout_id"])
            _require(
                torch.equal(host[live], reference.read("attention_output")),
                "active B/A kernel output differs",
            )
            result.update(reference_evaluation_id=prior_id, active_exact_equal=True)
        check()
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
                    result["passed"] = False
                    result["errors"].append(
                        {"type": "evidence_cleanup", "message": str(error)[:1024]}
                    )
        try:
            if target.type == "cuda":
                torch.cuda.synchronize(target)
            result["completion_confirmed"] = True
        except BaseException as error:
            result["errors"].append({"type": "device_cleanup", "message": str(error)[:1024]})
            _unsettled.append((pool, tensors, output, args))
        pool, tensors = {}, {}
        output = host = args = actual = None
        gc.collect()
        result["cache_references_released"] = bool(refs) and all(ref() is None for ref in refs)
        if (
            result["errors"]
            or not result["completion_confirmed"]
            or not result["cache_references_released"]
        ):
            result["status"], result["passed"] = "failed", False
        _persist_result(folder, root / "outputs" / evaluation_id, result, deadline)
    return result


def _failed_kernel_evidence(root, plan, result, outputs):
    import torch

    from vllm_lt.validation.diagnostics import TensorSpool

    row = result["evaluation"]
    layout = next(x for x in plan["layouts"] if x["layout_id"] == row["layout_id"])
    _, live, hashes, logical, _ = base._fixture(plan, layout, row["implementation_id"])
    known = False
    if "input_hashes" in result:
        _require(
            result["input_hashes"] == hashes and result["logical_input_hashes"] == logical,
            "failed kernel input identity differs",
        )
    chunks = result["cache_chunks"]
    _require(len(chunks) <= 40, "failed kernel guard coverage exceeds plan")
    for actual, wanted in zip(chunks, _chunk_records(plan, layout)):
        _require(
            {k: v for k, v in actual.items() if k != "actual_sha256"}
            == {k: v for k, v in wanted.items() if k != "actual_sha256"},
            "failed kernel guard identity differs",
        )
        _require(
            isinstance(actual["actual_sha256"], str) and len(actual["actual_sha256"]) == 64,
            "failed kernel guard hash missing",
        )
        known |= actual["actual_sha256"] != wanted["actual_sha256"]
    eid = row["evaluation_id"]
    if (root / "outputs" / eid / (layout["layout_id"] + ".index.json")).exists():
        spool = TensorSpool.open(root / "outputs", eid, layout["layout_id"])
        try:
            _require(
                list(spool.index) == ["attention_output"]
                and spool.read_metadata("attention_output")
                == {"evaluation_id": eid, "layout_id": layout["layout_id"]},
                "failed kernel output identity differs",
            )
            value = spool.read("attention_output")
            stats = base._output_stats(value, live)
            _require(
                stats == result["output_stats"]
                and base._tensor_hash(value) == result["output_sha256"],
                "failed kernel output statistics differ",
            )
            known |= (
                stats["finite_count"] != stats["numel"]
                or stats["inactive_zero_count"] != stats["inactive_numel"]
                or bool(stats["inactive_negative_zero_count"])
            )
            if row["implementation_id"] == "B":
                prior = outputs[eid.replace("K-B-", "K-A-", 1)]
                known |= not torch.equal(value[live], prior)
        finally:
            spool.close()
    return known


def audit_kernel_outputs(output_dir, plan, *, expected_device="cuda:0", allow_prefix=False):
    import torch

    from benchmarks.capture.lifecycle import _evaluation_prefix
    from vllm_lt.validation.diagnostics import TensorSpool
    from vllm_lt.validation.m3_persistent_lifecycle import _file_hash

    report = {
        "schema_version": 1,
        "artifact_type": "m3_capture_kernel_report",
        "complete": False,
        "passed": False,
        "errors": [],
        "completed_evaluations": [],
        "evaluations": [],
        "valid_prefix": False,
        "known_required_failure": False,
        "stopped_evaluation": None,
        "kernel_plan_sha256": plan.get("kernel_plan_sha256"),
        "checker": plan.get("checker"),
    }
    root = Path(output_dir)
    try:
        validate_kernel_plan(plan)
        order, stopped = _evaluation_prefix(root, plan, "kernel", expected_device)
        _require(
            allow_prefix or (len(order) == 6 and stopped is None),
            "kernel evaluation coverage differs",
        )
        files = {
            f"{x['evaluation_id']}/{x['layout_id']}{suffix}"
            for x in order
            for suffix in (".bin", ".index.json")
        }
        actual_files = {
            str(p.relative_to(root / "outputs"))
            for p in (root / "outputs").rglob("*")
            if p.is_file()
        }
        tail_files = (
            {
                f"{stopped['evaluation']['evaluation_id']}/{stopped['evaluation']['layout_id']}{suffix}"
                for suffix in (".bin", ".index.json")
            }
            if stopped
            else set()
        )
        _require(
            files <= actual_files <= files | tail_files,
            "kernel raw spool coverage differs",
        )
        previous, outputs = 0, {}
        for row in order:
            eid = row["evaluation_id"]
            layout = next(x for x in plan["layouts"] if x["layout_id"] == row["layout_id"])
            result = base._read(root / "evaluations" / eid / "result.json")
            _require(
                result["artifact_type"] == "m3_capture_kernel_result"
                and result["schema_version"] == 1
                and result["evaluation"] == row
                and result["kernel_plan_sha256"] == plan["kernel_plan_sha256"]
                and result["status"] == "complete"
                and result["passed"] is True
                and not result["errors"]
                and result["device"] == expected_device
                and result["completion_confirmed"] is True
                and result["cache_references_released"] is True,
                "kernel result identity/status/cleanup differs",
            )
            start, end, deadline = (result[k] for k in ("started_ns", "finished_ns", "deadline_ns"))
            _require(
                all(type(v) is int for v in (start, end, deadline))
                and previous <= start <= end < deadline
                and deadline - start <= 600 * 10**9,
                "kernel lifetime/order differs",
            )
            previous = end
            _require(
                base._read(root / "evaluations" / eid / "started.json")
                == {
                    "evaluation": row,
                    "started_ns": start,
                    "deadline_ns": deadline,
                    "kernel_plan_sha256": plan["kernel_plan_sha256"],
                },
                "kernel start marker differs",
            )
            _, live, hashes, logical, _ = base._fixture(plan, layout, row["implementation_id"])
            _require(
                result["input_hashes"] == hashes and result["logical_input_hashes"] == logical,
                "kernel input identities differ",
            )
            _require(
                result["cache_chunks"] == _chunk_records(plan, layout),
                "whole-pool40-chunk evidence differs",
            )
            spool = TensorSpool.open(root / "outputs", eid, layout["layout_id"])
            try:
                _require(list(spool.index) == ["attention_output"], "kernel tensor keys differ")
                _require(
                    spool.read_metadata("attention_output")
                    == {"evaluation_id": eid, "layout_id": layout["layout_id"]},
                    "kernel tensor metadata differs",
                )
                output = spool.read("attention_output")
            finally:
                spool.close()
            count = len(live) if row["implementation_id"] == "A" else 4
            _require(
                _valid_output(output, live, [count, 16, 128]) == result["output_stats"]
                and base._tensor_hash(output) == result["output_sha256"],
                "kernel output statistics differ",
            )
            if row["implementation_id"] == "B":
                prior_id = eid.replace("K-B-", "K-A-", 1)
                _require(
                    result["reference_evaluation_id"] == prior_id
                    and result["active_exact_equal"] is True
                    and torch.equal(output[live], outputs[prior_id]),
                    "active raw B/A outputs differ",
                )
            else:
                outputs[eid] = output
            report["completed_evaluations"].append(eid)
            report["evaluations"].append(
                {
                    "evaluation_id": eid,
                    "started_ns": start,
                    "finished_ns": end,
                    "deadline_ns": deadline,
                    "result_sha256": _file_hash(root / "evaluations" / eid / "result.json"),
                }
            )
        if stopped is not None:
            _require(previous <= stopped["started_ns"], "failed kernel order differs")
            report["known_required_failure"] = _failed_kernel_evidence(root, plan, stopped, outputs)
            report["stopped_evaluation"] = {
                "evaluation_id": stopped["evaluation"]["evaluation_id"],
                "started_ns": stopped["started_ns"],
                "finished_ns": stopped["finished_ns"],
                "deadline_ns": stopped["deadline_ns"],
                "errors": stopped["errors"],
                "result_sha256": _file_hash(
                    root / "evaluations" / stopped["evaluation"]["evaluation_id"] / "result.json"
                ),
            }
        report["artifact_bytes"] = base._usage(root, plan["config"]["evidence_bytes_max"])
        report["valid_prefix"] = True
        report["complete"] = report["passed"] = len(order) == 6 and stopped is None
    except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
        report["errors"].append({"type": type(error).__name__, "message": str(error)[:1024]})
    return report
