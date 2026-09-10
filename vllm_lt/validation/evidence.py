"""Stream validation comparisons without retaining native activation histories."""

import hashlib
import json
import math
from pathlib import Path

import torch

from .diagnostics import (
    MATCHING_BYTES,
    BudgetExceededError,
    MiB,
    NonfiniteComparisonError,
    TensorSpool,
    compare,
    comparison_json,
)
from .schema import read_json, write_json


def boundary_key(metadata):
    fields = [metadata[name] for name in ("operation", "positions", "depth", "layer")]
    if metadata["operation"] == "populated_kv":
        fields.append(metadata["component"])
    return json.dumps(fields, separators=(",", ":"))


def required_pass(record):
    if record["status"] == "incomparable":
        return record["stats"]["all_finite"]
    stats = record["stats"]
    return stats["all_finite"] and (
        not stats["numeric_required"]
        or (stats["allclose"] and (not record["top1_required"] or stats["top1_equal"]))
    )


def accumulate(summary, record):
    """Shared arithmetic for online writing and independent offline reconstruction."""
    summary["count"] += 1
    if record["status"] == "incomparable":
        summary["incomparable_count"] += 1
        if not required_pass(record):
            summary["required_failures"] += 1
            if summary["first_failure"] is None:
                summary["first_failure"] = record
        return
    summary["compared_count"] += 1
    stats = record["stats"]
    operation = record["metadata"]["operation"]
    group = summary["by_operation"].setdefault(
        operation,
        {
            "count": 0,
            "required_failures": 0,
            "max_abs_error": 0.0,
            "max_error_boundary": None,
            "first_nonexact_boundary": None,
        },
    )
    group["count"] += 1
    if stats["max_abs_error"] is not None and stats["max_abs_error"] > group["max_abs_error"]:
        group["max_abs_error"] = stats["max_abs_error"]
        group["max_error_boundary"] = record["key"]
    if not stats["exact_equal"] and group["first_nonexact_boundary"] is None:
        group["first_nonexact_boundary"] = record["key"]
    if not required_pass(record):
        summary["required_failures"] += 1
        group["required_failures"] += 1
        if summary["first_failure"] is None:
            summary["first_failure"] = record


def empty_summary(comparison):
    return {
        **comparison,
        "count": 0,
        "compared_count": 0,
        "incomparable_count": 0,
        "required_failures": 0,
        "first_failure": None,
        "by_operation": {},
        "behavior": [],
        "behavior_failures": 0,
        "first_behavior_failure": None,
        "complete": False,
    }


def _gates(row):
    values = [
        row[name] for name in ("gate_logits", "gate_probabilities", "cumulative_probabilities")
    ]
    if any(len(value) != row["exit_depth"] for value in values):
        raise ValueError("gate evidence does not cover the executed depths")
    if any(type(v) not in (int, float) or not math.isfinite(v) for value in values for v in value):
        raise ValueError("nonfinite or invalid gate evidence")
    if any(not 0 <= value <= 1 for series in values[1:] for value in series):
        raise ValueError("gate probability or CDF is outside [0, 1]")
    return values


def compare_behavior(actual, reference, *, live, threshold=None, route_control=None):
    if len(actual) != 9 or len(reference) != 9:
        raise ValueError("behavior comparison requires nine genuine prediction points")
    if route_control is None:
        route_control = "live" if live else "forced"
    if route_control not in ("live", "fixed", "forced") or (route_control == "live") != live:
        raise ValueError("route control differs from the declared comparison mode")
    threshold = (0.7 if threshold is None else threshold) if live else None
    if threshold is not None and (
        type(threshold) not in (int, float)
        or not math.isfinite(threshold)
        or not 0 <= threshold <= 1
    ):
        raise ValueError("invalid live gate threshold")
    records, diverged = [], False
    for index, (candidate, expected) in enumerate(zip(actual, reference)):
        if candidate["output_index"] != index or expected["output_index"] != index:
            raise ValueError("prediction trace order is invalid")
        same_history = candidate["history_sha256"] == expected["history_sha256"]
        if not same_history and (not live or not diverged):
            raise ValueError("input histories differ before a recorded live decision divergence")
        actual_gates, reference_gates = _gates(candidate), _gates(expected)
        reason = None
        if live and diverged:
            status, reason = "incomparable", "earlier live decision changed the trajectory"
        elif (candidate["exit_depth"], candidate["actual_token_id"]) != (
            expected["exit_depth"],
            expected["actual_token_id"],
        ):
            status, reason = "diverged", "actual token or selected exit differs on matched prefix"
            diverged = live
        else:
            status = "matched"
        gate_comparison = None
        if status != "incomparable":
            rows = []
            for depth in range(min(candidate["exit_depth"], expected["exit_depth"])):
                rows.append(
                    {
                        "depth": depth + 1,
                        "logit_delta": actual_gates[0][depth] - reference_gates[0][depth],
                        "probability_delta": actual_gates[1][depth] - reference_gates[1][depth],
                        "cdf_delta": actual_gates[2][depth] - reference_gates[2][depth],
                        "actual_threshold_distance": None
                        if threshold is None
                        else actual_gates[2][depth] - threshold,
                        "reference_threshold_distance": None
                        if threshold is None
                        else reference_gates[2][depth] - threshold,
                    }
                )
            gate_comparison = {
                "diagnostic_only": True,
                "threshold": threshold,
                "per_depth": rows,
                "actual_depths": candidate["exit_depth"],
                "reference_depths": expected["exit_depth"],
            }
        records.append(
            {
                "output_index": index,
                "status": status,
                "reason": reason,
                "actual": candidate,
                "reference": expected,
                "route_control": route_control,
                "gate_comparison": gate_comparison,
            }
        )
    return records


def _pair_preflight(actual, reference_bytes):
    size = actual.numel() * actual.element_size()
    staging = size * (1 + int(actual.device.type != "cpu") + int(not actual.is_contiguous()))
    # Covers the comparator's 65,536-element promoted chunk, masks and a single
    # Ouro vocabulary row's top-k workspace, and subsequent dump staging.
    if staging + reference_bytes + 8 * MiB > MATCHING_BYTES:
        raise BudgetExceededError("Comparison/dump pair exceeds the live matching byte budget")


def _candidate_finiteness(actual, operation):
    _pair_preflight(actual, 0)
    values = actual.detach().to(device="cpu").contiguous().reshape(-1)
    count, first = 0, None
    for start in range(0, values.numel(), 65536):
        finite = torch.isfinite(values[start : start + 65536])
        count += int(finite.sum())
        if first is None and not bool(finite.all()):
            first = start + int((~finite).nonzero()[0, 0])
    return {
        "operation": operation,
        "check": "candidate_finiteness_only",
        "all_finite": count == values.numel(),
        "actual_finite_count": count,
        "numel": values.numel(),
        "shape": list(actual.shape),
        "actual_dtype": str(actual.dtype).removeprefix("torch."),
        "reference_available": False,
        "first_nonfinite_flat_index": first,
    }


class ComparisonStream:
    def __init__(self, output, comparison, policy, dumps, plan_sha256, *, diagnostics=None):
        self.output, self.comparison, self.policy = Path(output), comparison, policy
        selection = {
            "preselected_dump_dtype": "bfloat16",
            "preselected_dump_comparison_kind": "same_dtype_fidelity",
        }
        if diagnostics is not None and any(
            diagnostics.get(name) != value for name, value in selection.items()
        ):
            raise ValueError("preselected dumps require the frozen BF16 same-dtype policy")
        self.dump_selection = selection
        self.summary = empty_summary(comparison)
        self.summary.update(
            schema_version=1, artifact_type="validation_comparison_summary", plan_sha256=plan_sha256
        )
        self.dumps = dumps
        self.reference = TensorSpool.open(
            self.output / "spools", comparison["reference_case_id"], comparison["fixture_id"]
        )
        self.seen = set()
        self._last_observation_index = 0
        self._live_divergence_output = None
        self._live_divergence_kind = None
        self._logit_outputs = set()
        folder = self.output / "comparisons"
        folder.mkdir(exist_ok=True)
        self.path = folder / (comparison["comparison_id"] + ".jsonl")
        self.file = self.path.open("xb")
        self.digest, self.size = hashlib.sha256(), 0

    def _record(self, record):
        encoded = comparison_json(record).encode()
        self.file.write(encoded)
        self.digest.update(encoded)
        self.size += len(encoded)
        accumulate(self.summary, record)
        self._last_observation_index = record["observation_index"]

    def observe(self, metadata, actual, *, observation_index=None):
        if self.file.closed:
            raise ValueError("Comparison stream is closed")
        if observation_index is None:
            observation_index = self.summary["count"] + 1
        if type(observation_index) is not int or observation_index <= self._last_observation_index:
            raise ValueError("observation_index must be a positive, strictly increasing integer")
        if metadata["fixture_id"] != self.comparison["fixture_id"]:
            raise ValueError("candidate boundary belongs to a different fixture")
        if str(actual.dtype).removeprefix("torch.") != self.comparison["dtype"]:
            raise ValueError("candidate tensor dtype differs from the frozen comparison")
        if self.comparison["family"] != "original" and (
            type(metadata["output_index"]) is not int or not 0 <= metadata["output_index"] <= 8
        ):
            raise ValueError("candidate output index is outside the nine prediction points")
        if self.comparison["family"] == "official" and (
            metadata["operation"] != "logits"
            or metadata["depth"] != 4
            or metadata["layer"] is not None
        ):
            raise ValueError("official comparison requires final four-loop logits")
        key = boundary_key(metadata)
        if key in self.seen:
            raise ValueError(f"duplicate candidate boundary: {key}")
        if len(self.seen) >= self.comparison["expected_boundary_records"]:
            raise BudgetExceededError("Candidate boundary count exceeds its frozen upper bound")
        self.seen.add(key)
        entry = self.reference.index.get(key)
        live = self.comparison["family"] == "live_gate"
        reason = None
        if entry is not None:
            expected_metadata = self.reference.read_metadata(key)
            for field in (
                "fixture_id",
                "output_index",
                "operation",
                "positions",
                "depth",
                "layer",
                "component",
                "axes",
                "depth_range",
                "layer_range",
            ):
                candidate_value = metadata.get(field)
                expected_value = expected_metadata.get(field)
                if field in ("positions", "axes", "depth_range", "layer_range"):
                    candidate_value = list(candidate_value) if candidate_value is not None else None
                if candidate_value != expected_value:
                    raise ValueError(f"corresponding boundary {field} metadata differs")
        if (
            live
            and self._live_divergence_output is not None
            and (
                metadata["output_index"] > self._live_divergence_output
                or (
                    metadata["operation"] == "populated_kv"
                    and (self._live_divergence_kind == "exit" or self._live_divergence_output < 8)
                )
            )
        ):
            reason = "earlier live decision changed token or KV history"
        elif entry is None:
            if not live:
                raise ValueError(f"missing reference boundary: {key}")
            reason = "different live exit depth omitted this reference boundary"
            self._live_divergence_output = metadata["output_index"]
            self._live_divergence_kind = "exit"
        elif entry["metadata"]["history_sha256"] != metadata["history_sha256"]:
            if not live or self._live_divergence_output is None:
                raise ValueError("corresponding boundary has a different supplied input history")
            reason = "different live input history"
        if metadata["operation"] == "logits":
            if (
                self.comparison["family"] != "original"
                and metadata["output_index"] in self._logit_outputs
            ):
                raise ValueError("duplicate final prediction output index")
            self._logit_outputs.add(metadata["output_index"])
        record = {
            "observation_index": observation_index,
            "key": key,
            "metadata": metadata,
            "stats": None,
            "status": "incomparable" if reason else "compared",
            "reason": reason,
            "top1_required": not (
                self.comparison["family"] == "original" and metadata["depth"] < 4
            ),
        }
        if reason:
            # Incomparable histories still must remain finite.
            record["stats"] = _candidate_finiteness(actual, metadata["operation"])
            self._record(record)
            if not record["stats"]["all_finite"]:
                self.file.flush()
                raise NonfiniteComparisonError(record["stats"])
            return
        _pair_preflight(actual, entry["size_bytes"])
        reference = self.reference.read(key)
        operation = (
            "bf16_fp32_sensitivity"
            if self.comparison["comparison_kind"] == "bf16_fp32_sensitivity"
            else metadata["operation"]
        )
        try:
            record["stats"] = compare(actual, reference, operation, self.policy)
        except NonfiniteComparisonError as exc:
            record["stats"] = exc.stats
            self._record(record)
            self.file.flush()
            raise
        self._record(record)
        if live and metadata["operation"] == "logits" and record["stats"]["top1_equal"] is False:
            self._live_divergence_output = metadata["output_index"]
            self._live_divergence_kind = "token"
        if not required_pass(record):
            self.dumps.observe_failure(metadata["fixture_id"], finite=True)
        fixture_id = metadata["fixture_id"]
        # Preserve the planned fixtures' quota for BF16 paired layer evidence.
        # Additional failure fixtures begin capture at their first finite failure
        # in execution order and keep that selection across later dtype passes.
        preselected = fixture_id in self.dumps.preselected_fixture_ids
        eligible = not preselected or (
            self.comparison["dtype"] == self.dump_selection["preselected_dump_dtype"]
            and self.comparison["comparison_kind"]
            == self.dump_selection["preselected_dump_comparison_kind"]
        )
        # One pair per observed comparison, bounded by the predeclared dump caps.
        pair_bytes = (
            actual.numel() * actual.element_size() + reference.numel() * reference.element_size()
        )
        if eligible and self.dumps.can_write(fixture_id, pair_bytes):
            for side, tensor in (("actual", actual), ("reference", reference)):
                dump_key = hashlib.sha256(
                    (self.comparison["comparison_id"] + key + side).encode()
                ).hexdigest()
                self.dumps.write(
                    fixture_id,
                    dump_key,
                    tensor,
                    {
                        **metadata,
                        "side": side,
                        "comparison_sha256": hashlib.sha256(
                            self.comparison["comparison_id"].encode()
                        ).hexdigest(),
                    },
                )

    def finish(self, traces=None):
        if self.comparison["family"] not in ("original", "official"):
            reference_result = read_json(
                self.output / "cases" / self.comparison["reference_case_id"] / "result.json"
            )
            if self.comparison["comparison_kind"] != "bf16_fp32_sensitivity":
                self.summary["behavior"] = compare_behavior(
                    traces,
                    reference_result["traces"][self.comparison["fixture_id"]],
                    live=self.comparison["family"] == "live_gate",
                    threshold=reference_result["case"]["exit_policy"]["threshold"],
                    route_control=reference_result["case"]["exit_policy"]["mode"],
                )
                failures = [row for row in self.summary["behavior"] if row["status"] == "diverged"]
                self.summary["behavior_failures"] = len(failures)
                self.summary["first_behavior_failure"] = failures[0] if failures else None
        expected = self.comparison["expected_boundary_records"]
        if self.comparison["counts_are_upper_bounds"]:
            if self.summary["count"] > expected:
                raise ValueError("live boundary count exceeds the frozen bound")
            # The driver checks all executed token/depth boundaries and all final KV chunks.
        elif self.summary["count"] != expected:
            raise ValueError(f"boundary coverage {self.summary['count']} != {expected}")
        if not self.comparison["counts_are_upper_bounds"]:
            expected_keys = {
                key
                for key, entry in self.reference.index.items()
                if self.comparison["family"] != "official"
                or entry["metadata"]["operation"] == "logits"
            }
            if self.seen != expected_keys:
                raise ValueError(
                    "candidate boundaries do not cover the declared reference boundaries"
                )
        if self.comparison["family"] != "original" and self._logit_outputs != set(range(9)):
            raise ValueError("final prediction coverage does not contain all nine output indices")
        self.summary["complete"] = True
        self.close()
        return self.summary

    def close(self):
        if not self.file.closed:
            self.file.close()
        self.reference.close()
        self.summary.update(sha256=self.digest.hexdigest(), size_bytes=self.size)
        write_json(self.path.with_suffix(".summary.json"), self.summary)
