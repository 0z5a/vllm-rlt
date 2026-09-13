"""Bounded, typed tensor evidence and CPU comparisons for the Q1 pass.

Spools are append-only raw bytes plus a JSON index. They preserve model dtype,
including BF16 bit patterns, and never use pickle. Budget counters are cumulative
and are not reset by closing or deleting a spool.
"""

import hashlib
import json
import math
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType

import torch

MiB = 1024**2
GiB = 1024**3
MATCHING_BYTES = 64 * MiB
INDEX_RECORD_BYTES = 1024
COMPARISON_RECORD_BYTES = 2048
_DTYPES = {
    "float32": (torch.float32, 4),
    "bfloat16": (torch.bfloat16, 2),
    "float16": (torch.float16, 2),
    "float64": (torch.float64, 8),
    "int64": (torch.int64, 8),
    "int32": (torch.int32, 4),
    "uint8": (torch.uint8, 1),
    "bool": (torch.bool, 1),
}


class BudgetExceededError(ValueError):
    """A requested write or comparison exceeds its declared byte budget."""


class NonfiniteComparisonError(ValueError):
    """Fatal comparison with its serializable evidence retained in ``stats``."""

    def __init__(self, stats):
        self.stats = stats
        super().__init__(f"Nonfinite values at {stats['operation']}; device work must stop")


def _integer(value, name, *, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an _integer >= {minimum}")
    return value


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def comparison_json(record: Mapping) -> str:
    """Encode one complete driver envelope, including its bounded JSONL newline."""
    encoded = _json(dict(record)) + "\n"
    if len(encoded.encode("utf-8")) > COMPARISON_RECORD_BYTES:
        raise BudgetExceededError("Comparison JSONL record exceeds 2048 bytes")
    return encoded


def _no_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON field {key!r}")
        result[key] = value
    return result


def _identity(value, name):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise ValueError(f"{name} must be a nonempty path-safe identifier")
    return value


def _key(value):
    if not isinstance(value, str) or not value:
        raise ValueError("A boundary key must be a nonempty string")


def _tensor_size(tensor):
    if not isinstance(tensor, torch.Tensor) or tensor.layout != torch.strided:
        raise ValueError("A strided tensor is required")
    dtype = str(tensor.dtype).removeprefix("torch.")
    if dtype not in _DTYPES:
        raise ValueError(f"Unsupported evidence dtype: {dtype}")
    return dtype, tensor.numel() * tensor.element_size()


class SpoolBudget:
    """One ledger shared by every namespace and dtype pass in an experiment."""

    def __init__(self, total_bytes=8 * GiB, per_group_bytes=2 * GiB):
        self.total_bytes = _integer(total_bytes, "total_bytes")
        self.per_group_bytes = _integer(per_group_bytes, "per_group_bytes")
        self.written_bytes = 0
        self.group_written_bytes = {}

    def check(self, group_id: str, size: int):
        _key(group_id)
        _integer(size, "size")
        if self.written_bytes + size > self.total_bytes:
            raise BudgetExceededError("Cumulative tensor spool byte budget exceeded")
        if self.group_written_bytes.get(group_id, 0) + size > self.per_group_bytes:
            raise BudgetExceededError(f"Tensor spool group byte budget exceeded: {group_id}")

    def claim(self, group_id: str, size: int):
        self.check(group_id, size)
        self.written_bytes += size
        self.group_written_bytes[group_id] = self.group_written_bytes.get(group_id, 0) + size


class TensorSpool:
    """One ``directory/namespace/fixture_id.{bin,index.json}`` pair.

    Metadata must be finite JSON data; tuples such as positions become lists.
    Index records are appended as writes complete. ``close()`` finalizes the JSON
    container; it does not claim that the fixture's planned coverage completed.
    The in-memory index permits immediate reads while a writer is open.
    """

    def __init__(
        self,
        directory: Path,
        namespace: str,
        fixture_id: str,
        budget: SpoolBudget,
        group_id: str,
        *,
        max_tensor_bytes=32 * MiB,
        matching_bytes=MATCHING_BYTES,
    ):
        self.namespace = _identity(namespace, "namespace")
        self.fixture_id = _identity(fixture_id, "fixture_id")
        self._budget, self._group_id = budget, group_id
        budget.check(group_id, 0)
        self.max_tensor_bytes = _integer(max_tensor_bytes, "max_tensor_bytes")
        self.matching_bytes = _integer(matching_bytes, "matching_bytes")
        folder = Path(directory) / namespace
        folder.mkdir(parents=True, exist_ok=True)
        self.data_path = folder / f"{fixture_id}.bin"
        self.index_path = folder / f"{fixture_id}.index.json"
        if self.data_path.exists() or self.index_path.exists():
            raise FileExistsError("Tensor spool files must be new")
        self._data = self.data_path.open("xb")
        self._index_file = self.index_path.open("x", encoding="utf-8")
        self._records, self._size, self._closed = {}, 0, False
        header = {
            "schema_version": 1,
            "artifact_type": "tensor_spool_index",
            "namespace": namespace,
            "fixture_id": fixture_id,
            "byte_order": sys.byteorder,
        }
        self._index_file.write(_json(header)[:-1] + ',"records":[\n')

    @property
    def index(self):
        return MappingProxyType(self._records)

    def write(self, key: str, tensor: torch.Tensor, metadata: Mapping):
        if self._closed:
            raise ValueError("Tensor spool is closed or read-only")
        _key(key)
        if key in self._records:
            raise ValueError(f"Duplicate tensor boundary: {key}")
        dtype, size = _tensor_size(tensor)
        if size > self.max_tensor_bytes or size > self.matching_bytes:
            raise BudgetExceededError("Tensor snapshot exceeds the single-boundary byte budget")
        self._budget.check(self._group_id, size)
        if not isinstance(metadata, Mapping):
            raise ValueError("Tensor metadata must be a mapping")
        metadata = json.loads(_json(dict(metadata)))
        record = {
            "key": key,
            "dtype": dtype,
            "shape": list(tensor.shape),
            "offset": self._size,
            "size_bytes": size,
            "sha256": "0" * 64,
            "metadata": metadata,
        }
        # Both hashes have fixed ASCII length. Check the complete record before
        # any tensor staging or byte-budget claim; reserve comma/newline overhead.
        if len(_json({**record, "record_sha256": "0" * 64}).encode()) + 2 > INDEX_RECORD_BYTES:
            raise BudgetExceededError("Tensor index record exceeds 1024 bytes")
        staging_bytes = size * (
            1 + int(tensor.device.type != "cpu") + int(not tensor.is_contiguous())
        )
        if staging_bytes > self.matching_bytes:
            raise BudgetExceededError("Tensor staging exceeds the live matching byte budget")
        cpu = tensor.detach().to(device="cpu").contiguous()
        # NumPy sees uint8, never BF16. memoryview avoids a second full byte copy.
        payload = memoryview(cpu.reshape(-1).view(torch.uint8).numpy()).cast("B")
        record["sha256"] = hashlib.sha256(payload).hexdigest()
        record["record_sha256"] = hashlib.sha256(_json(record).encode()).hexdigest()
        encoded = _json(record)
        self._budget.claim(self._group_id, size)
        if self._data.write(payload) != size:
            raise OSError("Incomplete tensor spool write")
        self._data.flush()
        self._index_file.write((",\n" if self._records else "") + encoded)
        self._index_file.flush()
        self._records[key] = record
        self._size += size
        return record

    def _verified_record(self, key):
        record = self._records[key]
        unhashed = {name: value for name, value in record.items() if name != "record_sha256"}
        if hashlib.sha256(_json(unhashed).encode()).hexdigest() != record["record_sha256"]:
            raise ValueError(f"Tensor index record SHA-256 mismatch: {key}")
        return record

    def read_metadata(self, key: str) -> dict:
        """Return an independent JSON copy of verified boundary metadata."""
        return json.loads(_json(self._verified_record(key)["metadata"]))

    def read(self, key: str) -> torch.Tensor:
        record = self._verified_record(key)
        size = record["size_bytes"]
        if size > self.matching_bytes:
            raise BudgetExceededError("Tensor read exceeds the matching byte budget")
        payload = bytearray(size)
        with self.data_path.open("rb") as source:
            source.seek(record["offset"])
            if source.readinto(payload) != size:
                raise ValueError(f"Truncated tensor evidence: {key}")
        if hashlib.sha256(payload).hexdigest() != record["sha256"]:
            raise ValueError(f"Tensor SHA-256 mismatch: {key}")
        dtype = _DTYPES[record["dtype"]][0]
        if size == 0:
            return torch.empty(record["shape"], dtype=dtype)
        # The resulting tensor owns a reference to the writable bytearray. It is
        # independent of the file and of all other read() calls.
        return torch.frombuffer(payload, dtype=dtype).reshape(record["shape"])

    def close(self):
        if not self._closed:
            self._data.close()
            self._index_file.write(f'\n],"size_bytes":{self._size}}}\n')
            self._index_file.close()
            self._closed = True

    @classmethod
    def open(
        cls, directory: Path, namespace: str, fixture_id: str, *, matching_bytes=MATCHING_BYTES
    ):
        namespace, fixture_id = (
            _identity(namespace, "namespace"),
            _identity(fixture_id, "fixture_id"),
        )
        self = cls.__new__(cls)
        folder = Path(directory) / namespace
        self.data_path, self.index_path = (
            folder / f"{fixture_id}.bin",
            folder / f"{fixture_id}.index.json",
        )
        document = json.loads(self.index_path.read_text(), object_pairs_hook=_no_duplicates)
        _json(document)  # Reject nonfinite JSON, including overflowing exponents.
        expected = {
            "schema_version",
            "artifact_type",
            "namespace",
            "fixture_id",
            "byte_order",
            "records",
            "size_bytes",
        }
        if not isinstance(document, dict) or set(document) != expected:
            raise ValueError("Malformed tensor spool index")
        if (
            type(document["schema_version"]) is not int
            or document["schema_version"] != 1
            or document["artifact_type"] != "tensor_spool_index"
            or document["namespace"] != namespace
            or document["fixture_id"] != fixture_id
            or document["byte_order"] != sys.byteorder
            or not isinstance(document["records"], list)
        ):
            raise ValueError("Unsupported tensor spool identity, version, or byte order")
        self._records, offset = {}, 0
        for record in document["records"]:
            if len(_json(record).encode()) + 2 > INDEX_RECORD_BYTES:
                raise BudgetExceededError("Tensor index record exceeds 1024 bytes")
            fields = {
                "key",
                "dtype",
                "shape",
                "offset",
                "size_bytes",
                "sha256",
                "metadata",
                "record_sha256",
            }
            if not isinstance(record, dict) or set(record) != fields:
                raise ValueError("Malformed tensor index record")
            unhashed = {name: value for name, value in record.items() if name != "record_sha256"}
            if hashlib.sha256(_json(unhashed).encode()).hexdigest() != record["record_sha256"]:
                raise ValueError("Tensor index record SHA-256 mismatch")
            _key(record["key"])
            shape, dtype = record["shape"], record["dtype"]
            if not isinstance(shape, list) or not isinstance(dtype, str) or dtype not in _DTYPES:
                raise ValueError("Invalid tensor shape or dtype")
            for dimension in shape:
                _integer(dimension, "tensor dimension")
            size = _integer(record["size_bytes"], "size_bytes")
            if (
                record["key"] in self._records
                or _integer(record["offset"], "offset") != offset
                or size != math.prod(shape) * _DTYPES[dtype][1]
                or not isinstance(record["metadata"], dict)
                or not isinstance(record["sha256"], str)
                or not re.fullmatch("[0-9a-f]{64}", record["sha256"])
            ):
                raise ValueError(
                    "Duplicate boundary or invalid tensor offset, size, hash, or metadata"
                )
            self._records[record["key"]] = record
            offset += size
        if (
            _integer(document["size_bytes"], "spool size") != offset
            or self.data_path.stat().st_size != offset
        ):
            raise ValueError("Tensor spool file size disagrees with its index")
        self.namespace, self.fixture_id = namespace, fixture_id
        self._size, self._closed = offset, True
        self.matching_bytes = _integer(matching_bytes, "matching_bytes")
        return self


def _coordinate(index, shape):
    coordinates = []
    for dimension in reversed(shape):
        coordinates.append(index % dimension)
        index //= dimension
    return list(reversed(coordinates))


def compare(
    actual: torch.Tensor,
    reference: torch.Tensor,
    operation: str,
    policy: Mapping,
    *,
    matching_bytes=MATCHING_BYTES,
):
    """Compare corresponding values with bounded FP64 CPU working chunks.

    Only same-dtype final ``logits`` acquire allclose and exact top-1 gates.
    Other operations pass required finiteness only; their numerical deltas are
    diagnostic, and ``allclose`` remains null. Nonfinite evidence is attached to
    the raised exception so a driver can persist it before stopping device work.
    """
    _key(operation)
    _key(policy.get("policy_id"))
    policy_hash = hashlib.sha256(_json(dict(policy)).encode()).hexdigest()
    actual_dtype, actual_bytes = _tensor_size(actual)
    reference_dtype, reference_bytes = _tensor_size(reference)
    if actual.shape != reference.shape or actual.numel() == 0:
        raise ValueError("Comparison requires matching, nonempty tensor shapes")
    if actual.dtype not in (
        torch.float32,
        torch.bfloat16,
        torch.float16,
    ) or reference.dtype not in (torch.float32, torch.bfloat16, torch.float16):
        raise ValueError("Comparison requires FP32, BF16, or FP16 tensors")
    numeric_required = operation == "logits"
    bounds = None
    if numeric_required:
        if actual_dtype != reference_dtype or actual.ndim == 0 or actual.shape[-1] < 2:
            raise ValueError("Logit comparisons require matching dtypes and a vocabulary axis")
        bounds = policy["logits"][actual_dtype]
        for name in ("atol", "rtol"):
            value = bounds[name]
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"Invalid logit {name}")
    # Include input-sized CPU staging/contiguous copies conservatively. The
    # 96-byte workspace allowance per element bounds promoted operands, deltas,
    # squared terms, masks, and tolerance temporaries. Top-k uses one row at a time.
    resident = actual_bytes + reference_bytes
    resident += actual_bytes if actual.device.type != "cpu" or not actual.is_contiguous() else 0
    resident += (
        reference_bytes if reference.device.type != "cpu" or not reference.is_contiguous() else 0
    )
    topk_workspace = actual.shape[-1] * 32 if numeric_required else 0
    available = _integer(matching_bytes, "matching_bytes") - resident - topk_workspace - 4096
    chunk_size = min(actual.numel(), 65536, available // 96)
    if chunk_size < 1:
        raise BudgetExceededError("Comparison exceeds the live tensor matching byte budget")
    a = actual.detach().to(device="cpu").contiguous().reshape(-1)
    r = reference.detach().to(device="cpu").contiguous().reshape(-1)
    finite_a = finite_r = 0
    first_a = first_r = None
    max_error, max_index, squared_error, squared_reference = -1.0, 0, 0.0, 0.0
    allclose = True
    for start in range(0, a.numel(), chunk_size):
        av, rv = a[start : start + chunk_size].double(), r[start : start + chunk_size].double()
        fa, fr = torch.isfinite(av), torch.isfinite(rv)
        ca, cr = int(fa.sum()), int(fr.sum())
        finite_a += ca
        finite_r += cr
        if ca != av.numel() and first_a is None:
            first_a = start + int((~fa).nonzero()[0, 0])
        if cr != rv.numel() and first_r is None:
            first_r = start + int((~fr).nonzero()[0, 0])
        if ca != av.numel() or cr != rv.numel():
            continue
        delta = (av - rv).abs()
        local_index = int(delta.argmax())
        local_error = float(delta[local_index])
        if local_error > max_error:
            max_error, max_index = local_error, start + local_index
        squared_error += float(delta.square().sum())
        squared_reference += float(rv.square().sum())
        if bounds is not None:
            allclose = allclose and bool(
                (delta <= bounds["atol"] + bounds["rtol"] * rv.abs()).all()
            )
    all_finite = finite_a == finite_r == a.numel()
    stats = {
        "schema_version": 1,
        "artifact_type": "tensor_comparison",
        "operation": operation,
        "policy_id": policy["policy_id"],
        "policy_sha256": policy_hash,
        "shape": list(actual.shape),
        "actual_dtype": actual_dtype,
        "reference_dtype": reference_dtype,
        "numel": a.numel(),
        "actual_finite_count": finite_a,
        "reference_finite_count": finite_r,
        "all_finite": all_finite,
        "first_nonfinite_actual_coordinate": None
        if first_a is None
        else _coordinate(first_a, actual.shape),
        "first_nonfinite_reference_coordinate": None
        if first_r is None
        else _coordinate(first_r, actual.shape),
        "max_abs_error": max_error if all_finite else None,
        "rms_error": math.sqrt(squared_error / a.numel()) if all_finite else None,
        "reference_rms": math.sqrt(squared_reference / r.numel()) if all_finite else None,
        "max_error_coordinate": _coordinate(max_index, actual.shape) if all_finite else None,
        "exact_equal": all_finite and actual_dtype == reference_dtype and torch.equal(a, r),
        "numeric_required": numeric_required,
        "allclose": allclose if all_finite and numeric_required else None,
        "top1_equal": None,
        "passed": all_finite,
        "bounds": dict(bounds) if bounds is not None else None,
    }
    if not all_finite:
        stats["status"] = "nonfinite"
        comparison_json(stats)
        raise NonfiniteComparisonError(stats)
    if numeric_required:
        vocabulary = actual.shape[-1]
        actual_ids, reference_ids, actual_margins, reference_margins = [], [], [], []
        for start in range(0, a.numel(), vocabulary):
            av, rv = a[start : start + vocabulary], r[start : start + vocabulary]
            actual_ids.append(int(av.argmax()))
            reference_ids.append(int(rv.argmax()))
            at, rt = av.topk(2).values.double(), rv.topk(2).values.double()
            actual_margins.append(float(at[0] - at[1]))
            reference_margins.append(float(rt[0] - rt[1]))
        stats.update(
            {
                "actual_top1_ids": actual_ids,
                "reference_top1_ids": reference_ids,
                "actual_top2_margins": actual_margins,
                "reference_top2_margins": reference_margins,
                "top1_equal": actual_ids == reference_ids,
                "passed": allclose and actual_ids == reference_ids,
            }
        )
    stats["status"] = (
        ("passed" if stats["passed"] else "failed") if numeric_required else "diagnostic_only"
    )
    comparison_json(stats)
    return stats


class DiagnosticDump:
    """Bounded first-boundary snapshots, selected within the original pass.

    Call failure notifications in frozen execution/fixture/boundary order.
    ``can_write`` implements the declared until-cap selection; once a caller
    requests ``write``, exceeding any budget is an error rather than a silent skip.
    Share ``budget`` with matching spools to include dumps in experiment totals.
    """

    def __init__(
        self,
        directory: Path,
        preselected_fixture_ids: Sequence[str],
        *,
        budget=None,
        per_fixture_bytes=64 * MiB,
        total_bytes=256 * MiB,
        max_snapshot_bytes=32 * MiB,
    ):
        if len(preselected_fixture_ids) != 2 or len(set(preselected_fixture_ids)) != 2:
            raise ValueError("Exactly two distinct diagnostic fixture IDs must be preselected")
        self.directory = Path(directory)
        self._preselected_fixture_ids = tuple(
            _identity(value, "fixture_id") for value in preselected_fixture_ids
        )
        self.selected_fixture_ids = list(self._preselected_fixture_ids)
        self.failure_fixture_ids = []
        self._quota = SpoolBudget(total_bytes, per_fixture_bytes)
        self._budget = budget if budget is not None else self._quota
        self.max_snapshot_bytes = _integer(max_snapshot_bytes, "max_snapshot_bytes")
        self._spools = {}
        self._closed = False

    @property
    def preselected_fixture_ids(self) -> tuple[str, ...]:
        """Immutable planned selection, separate from later failure selections."""
        return self._preselected_fixture_ids

    def observe_failure(self, fixture_id: str, *, finite=True):
        _identity(fixture_id, "fixture_id")
        if (
            finite
            and fixture_id not in self.selected_fixture_ids
            and len(self.failure_fixture_ids) < 2
        ):
            self.failure_fixture_ids.append(fixture_id)
            self.selected_fixture_ids.append(fixture_id)
        return fixture_id in self.selected_fixture_ids

    def can_write(self, fixture_id: str, size_bytes: int):
        if self._closed:
            raise ValueError("Diagnostic dump is closed")
        _integer(size_bytes, "size_bytes")
        if fixture_id not in self.selected_fixture_ids:
            return False
        if size_bytes > self.max_snapshot_bytes:
            raise BudgetExceededError("Diagnostic snapshot exceeds the single-snapshot byte budget")
        try:
            self._quota.check(fixture_id, size_bytes)
        except BudgetExceededError:
            return False
        return True

    def write(self, fixture_id: str, key: str, tensor: torch.Tensor, metadata: Mapping):
        if self._closed:
            raise ValueError("Diagnostic dump is closed")
        if fixture_id not in self.selected_fixture_ids:
            raise ValueError("Fixture was not selected for diagnostic snapshots")
        _, size = _tensor_size(tensor)
        self._quota.check(fixture_id, size)
        self._budget.check(fixture_id, size)
        if fixture_id not in self._spools:
            self._spools[fixture_id] = TensorSpool(
                self.directory,
                "diagnostic",
                fixture_id,
                self._budget,
                fixture_id,
                max_tensor_bytes=self.max_snapshot_bytes,
            )
        record = self._spools[fixture_id].write(key, tensor, metadata)
        if self._budget is not self._quota:
            self._quota.claim(fixture_id, size)
        return record

    @property
    def written_bytes(self):
        return self._quota.written_bytes

    @property
    def fixture_written_bytes(self):
        return MappingProxyType(self._quota.group_written_bytes)

    def close(self):
        for spool in self._spools.values():
            spool.close()
        self._closed = True
