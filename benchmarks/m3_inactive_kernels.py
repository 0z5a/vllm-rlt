"""The finite M3 held-input matrix and portable evidence; no imports initialize CUDA."""

import hashlib
import json
import math
import time
from pathlib import Path

_LAYOUTS = (
    (0, 1, (), ()),
    (1, 1, (), ()),
    (1, 1, (0,), (1,)),
    (4, 1, (), ()),
    (4, 1, (0,), (16,)),
    (4, 1, (3,), (15,)),
    (4, 1, (0, 2), (1, 16)),
    (4, 1, (0, 1, 2, 3), (1, 2, 15, 16)),
    (4, 2, (0, 1, 2, 3), (15, 16, 17, 31)),
    (8, 2, (0, 1, 2, 3), (16, 17, 31, 32)),
    (8, 2, (1, 3, 5, 7), (16, 17, 31, 32)),
    (8, 3, (1, 3, 5, 7), (17, 31, 32, 33)),
    (8, 32, (1, 3, 5, 7), (33, 511, 512, 16)),
)


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _read(path):
    from vllm_lt.validation.schema import read_json

    return read_json(path)


def _write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def build_kernel_plan():
    """Resolve 52 normal evaluations and CPU input hashes, without weights or CUDA."""
    config = {
        "dtype": "float32",
        "query_heads": 16,
        "kv_heads": 16,
        "head_dim": 128,
        "num_blocks": 160,
        "layers": 2,
        "target_layer": 1,
        "block_size": 16,
        "chunk_blocks": 8,
        "case_timeout_s": 600,
        "matching_bytes": 64 * 1024**2,
        "evidence_bytes_max": 256 * 1024**2,
        "cache_formula": {"key": [1, 23, 257, 128, 64], "value": [3, 31, 257, 128, 64]},
        "row_formula": {
            "query": [3, 11, 251, 125, 128],
            "key": [5, 17, 127, 63, 64],
            "value": [7, 19, 127, 63, 64],
        },
        "formula_order": "((flat_index*multiplier+offset)%modulus-center)/divisor; float32",
        "page_mapping": "consecutive unique logical pages map to physical2*j+1",
        "inactive": "NaN query/key/value; length0; addresses -1/even-row or2**30/odd-row",
        "comparison": "same-backend exact active output; positive-zero padding; whole-cache exact",
    }
    layouts = [
        {
            "layout_id": f"L{index:02d}",
            "row_count": count,
            "table_width": width,
            "live_rows": list(rows),
            "lengths": list(lengths),
        }
        for index, (count, width, rows, lengths) in enumerate(_LAYOUTS)
    ]
    executions = [
        {
            "evaluation_id": f"K-{implementation}-{backend}-{layout['layout_id']}",
            "implementation_id": implementation,
            "backend": backend,
            "layout_id": layout["layout_id"],
        }
        for implementation in ("A", "B")
        for backend in ("torch", "triton")
        for layout in layouts
    ]
    output_rows = 2 * sum(len(row["live_rows"]) + row["row_count"] for row in layouts)
    plan = {
        "schema_version": 1,
        "artifact_type": "m3_inactive_kernel_plan",
        "config": config,
        "layouts": layouts,
        "execution_order": executions,
        "checker": {
            "status": "unavailable",
            "execution_count": 0,
            "intended_layout_ids": [row["layout_id"] for row in layouts],
            "limitation": (
                "No compute-sanitizer/cuda-memcheck installed; guards do not replace "
                "a device memory-access checker."
            ),
        },
        "resource_estimates": {
            "evaluations": 52,
            "checker_evaluations": 0,
            "cache_bytes": 2 * 160 * 2 * 16 * 16 * 128 * 4,
            "output_payload_bytes": output_rows * 16 * 128 * 4,
            "cache_chunks_per_evaluation": 40,
            "output_index_records": 52,
            "max_output_tensor_bytes": 8 * 16 * 128 * 4,
            "auxiliary_evidence_bytes_upper_bound": 8 * 1024**2,
        },
    }
    plan["input_identities"] = {}
    for layout in layouts:
        identities = {}
        for implementation in ("A", "B"):
            _, _, hashes, logical_hashes, _ = _fixture(plan, layout, implementation)
            identities[implementation] = {
                "input_hashes": hashes,
                "logical_input_hashes": logical_hashes,
            }
        if identities["A"]["logical_input_hashes"] != identities["B"]["logical_input_hashes"]:
            raise ValueError("compact and padded logical kernel inputs differ")
        plan["input_identities"][layout["layout_id"]] = identities
    plan["kernel_plan_sha256"] = _digest(plan)
    return plan


def validate_kernel_plan(plan):
    if _digest(plan) != _digest(build_kernel_plan()):
        raise ValueError("kernel plan differs from the frozen13-layout/52-evaluation matrix")


def _tensor_hash(tensor):
    import torch

    view = tensor.detach().contiguous().reshape(-1).view(torch.uint8)
    return hashlib.sha256(memoryview(view.numpy()).cast("B")).hexdigest()


def _values(start, count, formula):
    import torch

    multiplier, offset, modulus, center, divisor = formula
    values = torch.arange(start, start + count, dtype=torch.int64, device="cpu")
    return ((values * multiplier + offset) % modulus - center).to(torch.float32) / divisor


def _fixture(plan, layout, implementation):
    import torch

    config = plan["config"]
    count = len(layout["live_rows"]) if implementation == "A" else layout["row_count"]
    live_rows = list(range(count)) if implementation == "A" else layout["live_rows"]
    query = torch.full(
        (count, config["query_heads"], config["head_dim"]),
        float("nan"),
        dtype=torch.float32,
        device="cpu",
    )
    key = torch.full(
        (count, config["kv_heads"], config["head_dim"]),
        float("nan"),
        dtype=torch.float32,
        device="cpu",
    )
    value = key.clone()
    lengths = torch.zeros(count, dtype=torch.int32, device="cpu")
    active = torch.zeros(count, dtype=torch.bool, device="cpu")
    tables = (
        torch.tensor(
            [-1 if row % 2 == 0 else 2**30 for row in range(count)], dtype=torch.int32, device="cpu"
        )
        .unsqueeze(1)
        .expand(count, layout["table_width"])
        .clone()
    )
    blocks = torch.tensor(
        [-1 if row % 2 == 0 else 2**30 for row in range(count)], dtype=torch.int64, device="cpu"
    )
    offsets = blocks.clone()
    logical_pages = []
    page_index = 0
    for logical, (physical, length) in enumerate(zip(live_rows, layout["lengths"])):
        active[physical], lengths[physical] = True, length
        pages = [
            2 * j + 1
            for j in range(page_index, page_index + math.ceil(length / config["block_size"]))
        ]
        page_index += len(pages)
        logical_pages.append(pages)
        tables[physical, : len(pages)] = torch.tensor(pages, dtype=torch.int32, device="cpu")
        blocks[physical], offsets[physical] = pages[-1], (length - 1) % config["block_size"]
        for tensor, name in ((query, "query"), (key, "key"), (value, "value")):
            width = tensor.shape[1] * tensor.shape[2]
            tensor[physical] = _values(logical * width, width, config["row_formula"][name]).reshape(
                tensor.shape[1:]
            )
    assert page_index * 2 <= config["num_blocks"]
    tensors = {
        "query": query,
        "key": key,
        "value": value,
        "block_tables": tables,
        "context_lengths": lengths,
        "write_blocks": blocks,
        "write_offsets": offsets,
        "active": active,
    }
    hashes = {name: _tensor_hash(tensor) for name, tensor in tensors.items()}
    # Effective rows and their page histories match even when physical row positions differ.
    logical_hashes = {
        name: _tensor_hash(tensor[live_rows])
        for name, tensor in tensors.items()
        if name != "active"
    }
    # Unused table columns are inaccessible; give both sides the same canonical sentinel.
    canonical_tables = tables[live_rows].clone()
    for index, length in enumerate(layout["lengths"]):
        canonical_tables[index, math.ceil(length / config["block_size"]) :] = -1
    logical_hashes["block_tables"] = _tensor_hash(canonical_tables)
    return tensors, live_rows, hashes, logical_hashes, logical_pages


def _cache_chunk(plan, layout, component, start, stop, *, written):
    config = plan["config"]
    shape = (
        stop - start,
        config["layers"],
        config["block_size"],
        config["kv_heads"],
        config["head_dim"],
    )
    per_block = math.prod(shape[1:])
    chunk = _values(
        start * per_block, (stop - start) * per_block, config["cache_formula"][component]
    ).reshape(shape)
    if written:
        page_offset = 0
        width = config["kv_heads"] * config["head_dim"]
        for logical, length in enumerate(layout["lengths"]):
            page_count = math.ceil(length / config["block_size"])
            block = 2 * (page_offset + page_count - 1) + 1
            page_offset += page_count
            if start <= block < stop:
                chunk[
                    block - start, config["target_layer"], (length - 1) % config["block_size"]
                ] = _values(logical * width, width, config["row_formula"][component]).reshape(
                    config["kv_heads"], config["head_dim"]
                )
    return chunk


def _usage(root, limit):
    size = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
    if size > limit:
        raise ValueError("kernel evidence exceeds its256MiB budget")
    return size


def _output_stats(output, live_rows):
    import torch

    inactive = [row for row in range(output.shape[0]) if row not in live_rows]
    padding = output[inactive]
    return {
        "shape": list(output.shape),
        "dtype": str(output.dtype).removeprefix("torch."),
        "numel": output.numel(),
        "finite_count": int(output.isfinite().sum()),
        "inactive_numel": padding.numel(),
        "inactive_zero_count": int((padding == 0).sum()),
        "inactive_negative_zero_count": int(torch.signbit(padding).sum()),
    }


def run_kernel_evaluation(plan, evaluation_id, output_dir, *, device, deadline_ns):
    """Execute one declared row. Device reservation/lifetime belongs to the parent."""
    import torch

    from vllm_lt.kernels.paged_attention import torch_paged_attention, triton_paged_attention
    from vllm_lt.validation.diagnostics import SpoolBudget, TensorSpool

    validate_kernel_plan(plan)
    row = next(item for item in plan["execution_order"] if item["evaluation_id"] == evaluation_id)
    layout = next(item for item in plan["layouts"] if item["layout_id"] == row["layout_id"])
    config = plan["config"]
    root = Path(output_dir)
    folder = root / "evaluations" / evaluation_id
    folder.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter_ns()
    deadline = min(deadline_ns, started + config["case_timeout_s"] * 10**9)
    marker = {
        "schema_version": 1,
        "kernel_plan_sha256": plan["kernel_plan_sha256"],
        "evaluation": row,
        "started_ns": started,
        "deadline_ns": deadline,
    }
    _write(folder / "started.json", marker)
    result = {
        "schema_version": 1,
        "artifact_type": "m3_kernel_result",
        **marker,
        "status": "incomplete",
        "passed": False,
        "errors": [],
        "cache_chunks": [],
    }
    target = torch.device(device)
    cache = {}
    tensors, output, host_output, spool = None, None, None, None
    arguments = indices = blocks = offsets = actual = expected = None

    def check_time():
        if time.perf_counter_ns() >= deadline:
            raise TimeoutError("kernel case lifetime deadline exhausted")

    try:
        check_time()
        if row["backend"] == "triton" and target.type != "cuda":
            raise ValueError("declared Triton evaluation requires its reserved CUDA device")
        cpu, live_rows, physical_hashes, logical_hashes, _ = _fixture(
            plan, layout, row["implementation_id"]
        )
        result.update(input_hashes=physical_hashes, logical_input_hashes=logical_hashes)
        if plan["input_identities"][layout["layout_id"]][row["implementation_id"]] != {
            "input_hashes": physical_hashes,
            "logical_input_hashes": logical_hashes,
        }:
            raise ValueError("runtime kernel inputs differ from prelaunch CPU hashes")
        tensors = {name: tensor.to(target) for name, tensor in cpu.items()}
        del cpu
        shape = (
            config["num_blocks"],
            config["layers"],
            config["block_size"],
            config["kv_heads"],
            config["head_dim"],
        )
        cache = {
            component: torch.empty(shape, dtype=torch.float32, device=target)
            for component in ("key", "value")
        }
        for component in cache:
            for start in range(0, config["num_blocks"], config["chunk_blocks"]):
                check_time()
                stop = start + config["chunk_blocks"]
                cache[component][start:stop].copy_(
                    _cache_chunk(plan, layout, component, start, stop, written=False)
                )
        layer = config["target_layer"]
        if row["implementation_id"] == "B" and row["backend"] == "triton":
            from vllm_lt.kernels.triton_kv_write import masked_kv_write

            masked_kv_write(
                cache["key"][:, layer],
                cache["value"][:, layer],
                tensors["write_blocks"],
                tensors["write_offsets"],
                tensors["key"],
                tensors["value"],
                tensors["active"],
            )
        elif live_rows:
            indices = torch.tensor(live_rows, dtype=torch.long, device=target)
            blocks, offsets = tensors["write_blocks"][indices], tensors["write_offsets"][indices]
            cache["key"][blocks, layer, offsets] = tensors["key"][indices]
            cache["value"][blocks, layer, offsets] = tensors["value"][indices]
            del indices, blocks, offsets
        attention = torch_paged_attention if row["backend"] == "torch" else triton_paged_attention
        arguments = (
            tensors["query"],
            cache["key"][:, layer],
            cache["value"][:, layer],
            tensors["block_tables"],
            tensors["context_lengths"],
        )
        output = (
            attention(*arguments)
            if row["implementation_id"] == "A"
            else attention(*arguments, tensors["active"])
        )
        del arguments
        host_output = output.to("cpu")
        result["output_stats"] = _output_stats(host_output, live_rows)
        stats = result["output_stats"]
        if (
            stats["finite_count"] != stats["numel"]
            or stats["inactive_zero_count"] != stats["inactive_numel"]
            or stats["inactive_negative_zero_count"]
        ):
            raise ValueError("kernel output is nonfinite or inactive output is not positive zero")
        spool = TensorSpool(
            root / "outputs",
            evaluation_id,
            layout["layout_id"],
            SpoolBudget(config["evidence_bytes_max"], config["evidence_bytes_max"]),
            evaluation_id,
        )
        spool.write(
            "attention_output",
            host_output,
            {"evaluation_id": evaluation_id, "layout_id": layout["layout_id"]},
        )
        spool.close()
        result["output_sha256"] = _tensor_hash(host_output)
        for component in cache:
            for start in range(0, config["num_blocks"], config["chunk_blocks"]):
                check_time()
                stop = start + config["chunk_blocks"]
                actual = cache[component][start:stop].to("cpu")
                expected = _cache_chunk(plan, layout, component, start, stop, written=True)
                state = {
                    "component": component,
                    "block_start": start,
                    "block_stop": stop,
                    "shape": list(actual.shape),
                    "dtype": "float32",
                    "size_bytes": actual.numel() * 4,
                    "actual_sha256": _tensor_hash(actual),
                    "expected_sha256": _tensor_hash(expected),
                }
                result["cache_chunks"].append(state)
                del actual, expected
                if state["actual_sha256"] != state["expected_sha256"]:
                    raise ValueError("active write or guard/neighbor cache bytes differ")
        if row["implementation_id"] == "B":
            reference_id = evaluation_id.replace("K-B-", "K-A-", 1)
            reference = TensorSpool.open(root / "outputs", reference_id, layout["layout_id"])
            try:
                expected = reference.read("attention_output")
            finally:
                reference.close()
            result["reference_evaluation_id"] = reference_id
            result["active_exact_equal"] = torch.equal(host_output[live_rows], expected)
            if not result["active_exact_equal"]:
                raise ValueError(
                    "masked active attention differs from compact same-backend reference"
                )
        result["status"], result["passed"] = "complete", True
        result["artifact_bytes"] = _usage(root, config["evidence_bytes_max"])
        check_time()
    except (Exception, KeyboardInterrupt) as exc:
        result["status"] = (
            "incomplete" if isinstance(exc, (TimeoutError, KeyboardInterrupt)) else "failed"
        )
        result["passed"] = False
        result["errors"].append({"type": type(exc).__name__, "message": str(exc)})
    finally:
        if spool is not None:
            try:
                spool.close()
            except Exception as exc:
                result["status"], result["passed"] = "failed", False
                result["errors"].append({"type": "evidence_cleanup", "message": str(exc)})
        cache, tensors, output, host_output = {}, None, None, None
        arguments = indices = blocks = offsets = actual = expected = None
        del arguments, indices, blocks, offsets, actual, expected
        if target.type == "cuda":
            try:
                torch.cuda.synchronize()
            except Exception as exc:
                result["status"], result["passed"] = "failed", False
                result["errors"].append({"type": "device_cleanup", "message": str(exc)})
        result["finished_ns"] = time.perf_counter_ns()
        result["cache_references_released"] = True
        if result["finished_ns"] >= deadline:
            result["status"], result["passed"] = "incomplete", False
            result["errors"].append({"type": "deadline", "message": "kernel lifetime exceeded"})
        _write(folder / "result.json", result)
    return result


def audit_kernel_outputs(output_dir, plan):
    """Read raw typed outputs and reconstruct fixture/full-cache hashes without CUDA."""
    import torch

    from vllm_lt.validation.diagnostics import TensorSpool

    validate_kernel_plan(plan)
    root, config = Path(output_dir), plan["config"]
    report = {
        "schema_version": 1,
        "artifact_type": "m3_kernel_report",
        "kernel_plan_sha256": plan["kernel_plan_sha256"],
        "complete": False,
        "passed": False,
        "completed_evaluations": [],
        "errors": [],
        "evaluations": [],
        "checker": plan["checker"],
    }
    try:
        actual_ids = sorted(path.name for path in (root / "evaluations").iterdir() if path.is_dir())
        if actual_ids != sorted(row["evaluation_id"] for row in plan["execution_order"]):
            raise ValueError("kernel evidence lacks exact52 evaluation directories")
        expected_output_files = {
            f"{row['evaluation_id']}/{row['layout_id']}{suffix}"
            for row in plan["execution_order"]
            for suffix in (".bin", ".index.json")
        }
        actual_output_files = {
            str(path.relative_to(root / "outputs"))
            for path in (root / "outputs").rglob("*")
            if path.is_file()
        }
        if actual_output_files != expected_output_files:
            raise ValueError("kernel raw output files differ from the frozen52 spools")
        outputs, expected_cache = {}, {}
        previous = 0
        for row in plan["execution_order"]:
            evaluation_id = row["evaluation_id"]
            layout = next(item for item in plan["layouts"] if item["layout_id"] == row["layout_id"])
            result = _read(root / "evaluations" / evaluation_id / "result.json")
            if (
                result["schema_version"] != 1
                or result["artifact_type"] != "m3_kernel_result"
                or result["evaluation"] != row
                or result["kernel_plan_sha256"] != plan["kernel_plan_sha256"]
                or result["status"] != "complete"
                or result["passed"] is not True
                or result["errors"]
            ):
                raise ValueError("kernel result identity/status differs")
            start, end, deadline = (
                result[name] for name in ("started_ns", "finished_ns", "deadline_ns")
            )
            if (
                any(type(value) is not int for value in (start, end, deadline))
                or not previous <= start <= end < deadline
                or deadline - start > config["case_timeout_s"] * 10**9
            ):
                raise ValueError("kernel lifetime/order differs")
            previous = end
            marker = {
                "schema_version": 1,
                "kernel_plan_sha256": plan["kernel_plan_sha256"],
                "evaluation": row,
                "started_ns": start,
                "deadline_ns": deadline,
            }
            if (
                _read(root / "evaluations" / evaluation_id / "started.json") != marker
                or result["cache_references_released"] is not True
            ):
                raise ValueError("kernel marker or cache cleanup differs")
            tensors, live_rows, hashes, logical, _ = _fixture(
                plan, layout, row["implementation_id"]
            )
            del tensors
            if result["input_hashes"] != hashes or result["logical_input_hashes"] != logical:
                raise ValueError("kernel supplied input hashes differ from deterministic fixture")
            if layout["layout_id"] not in expected_cache:
                expected_chunks = []
                for component in ("key", "value"):
                    for begin in range(0, config["num_blocks"], config["chunk_blocks"]):
                        stop = begin + config["chunk_blocks"]
                        expected = _cache_chunk(plan, layout, component, begin, stop, written=True)
                        digest = _tensor_hash(expected)
                        expected_chunks.append(
                            {
                                "component": component,
                                "block_start": begin,
                                "block_stop": stop,
                                "shape": list(expected.shape),
                                "dtype": "float32",
                                "size_bytes": expected.numel() * 4,
                                "actual_sha256": digest,
                                "expected_sha256": digest,
                            }
                        )
                        del expected
                expected_cache[layout["layout_id"]] = expected_chunks
            expected_chunks = expected_cache[layout["layout_id"]]
            if result["cache_chunks"] != expected_chunks:
                raise ValueError("whole-cache chunk coverage or active/guard hashes differ")
            spool = TensorSpool.open(root / "outputs", evaluation_id, layout["layout_id"])
            try:
                if list(spool.index) != ["attention_output"]:
                    raise ValueError("kernel output boundary coverage differs")
                if spool.read_metadata("attention_output") != {
                    "evaluation_id": evaluation_id,
                    "layout_id": layout["layout_id"],
                }:
                    raise ValueError("kernel raw output metadata differs")
                output = spool.read("attention_output")
            finally:
                spool.close()
            stats = _output_stats(output, live_rows)
            count = len(live_rows) if row["implementation_id"] == "A" else layout["row_count"]
            if (
                stats["shape"] != [count, config["query_heads"], config["head_dim"]]
                or stats["dtype"] != "float32"
                or stats != result["output_stats"]
                or _tensor_hash(output) != result["output_sha256"]
            ):
                raise ValueError("kernel raw output identity/statistics differ")
            if (
                stats["finite_count"] != stats["numel"]
                or stats["inactive_zero_count"] != stats["inactive_numel"]
                or stats["inactive_negative_zero_count"]
            ):
                raise ValueError("kernel raw output violates finite/zero padding")
            if row["implementation_id"] == "B":
                reference_id = evaluation_id.replace("K-B-", "K-A-", 1)
                reference = outputs[reference_id]
                if (
                    result["reference_evaluation_id"] != reference_id
                    or result["active_exact_equal"] is not True
                    or not torch.equal(output[live_rows], reference)
                ):
                    raise ValueError("kernel masked output differs from raw compact reference")
            else:
                outputs[evaluation_id] = output
            report["evaluations"].append(
                {
                    "evaluation_id": evaluation_id,
                    "started_ns": start,
                    "finished_ns": end,
                    "deadline_ns": deadline,
                    "output_sha256": result["output_sha256"],
                }
            )
            report["completed_evaluations"].append(evaluation_id)
        report["artifact_bytes"] = _usage(root, config["evidence_bytes_max"])
        report["complete"] = report["passed"] = True
    except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
        report["errors"].append({"type": type(exc).__name__, "message": str(exc)})
    return report
