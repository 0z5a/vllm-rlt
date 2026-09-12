"""Real bounded evidence writers/auditors with CPU kernel and graph substitutes."""

import time

import pytest
import torch
from test_recurrent_graph import fake_runtime

from benchmarks.capture import kernels, lifecycle
from vllm_lt.core.kv_cache_manager import KVCacheManager
from vllm_lt.kernels.paged_attention import torch_paged_attention
from vllm_lt.validation.schema import read_json, write_json

pytestmark = pytest.mark.usefixtures("forbid_cuda")


@pytest.fixture(params=["kernel", "lifecycle"])
def harness(request, monkeypatch):
    fake_runtime(monkeypatch)

    def scatter(keys, values, blocks, offsets, k, v, active):
        keys[blocks[active], offsets[active]] = k[active]
        values[blocks[active], offsets[active]] = v[active]

    def cache(device):
        assert device.type == "cpu"
        value = KVCacheManager(2, 16, 128, 160, 16, 4)
        value.backend = "triton"
        return value

    monkeypatch.setattr(kernels, "_device_kernels", lambda device: (torch_paged_attention, scatter))
    monkeypatch.setattr(lifecycle, "_create_cache", cache)
    if request.param == "kernel":
        plan = kernels.build_kernel_plan()
        assert [(r["live_rows"], r["lengths"]) for r in plan["layouts"]] == [
            ([], []),
            ([1], [512]),
            ([1, 3], [16, 17]),
        ]
        assert len(plan["execution_order"]) == 6
        for value in plan["input_identities"].values():
            assert value["A"]["logical_input_hashes"] == value["B"]["logical_input_hashes"]
        return kernels, plan, kernels.run_kernel_evaluation, kernels.audit_kernel_outputs
    plan = lifecycle.build_lifecycle_plan()
    assert plan["steps"][2]["positions"] == [511]
    assert plan["expected_executor"]["bucket_visits"] == {"4": 4, "8": 3}
    assert len(plan["execution_order"]) == 2
    return lifecycle, plan, lifecycle.run_lifecycle_evaluation, lifecycle.audit_lifecycle_outputs


def test_evidence_roundtrip_and_corruption(harness, tmp_path):
    module, plan, execute, audit = harness
    for row in plan["execution_order"]:
        result = execute(
            plan,
            row["evaluation_id"],
            tmp_path,
            device="cpu",
            deadline_ns=time.perf_counter_ns() + 600 * 10**9,
        )
        assert result["passed"], result
    checked = audit(tmp_path, plan, expected_device="cpu")
    assert checked["passed"] and len(checked["evaluations"]) == len(plan["execution_order"])
    assert not audit(tmp_path, plan)["passed"]  # CPU evidence cannot qualify CUDA.
    path = tmp_path / "evaluations" / plan["execution_order"][-1]["evaluation_id"] / "result.json"
    result = read_json(path)
    if module is kernels:
        result["cache_chunks"][-1]["actual_sha256"] = "0" * 64
    else:
        result["steps"][2]["prefixes_at_publication"]["long"][0][1] = 0
    write_json(path, result)
    checked = audit(tmp_path, plan, expected_device="cpu")
    assert not checked["passed"] and checked["errors"]


@pytest.mark.parametrize("failure", ["prepare", "nonfinite"])
def test_failed_prefix_preserves_actual_evidence(harness, tmp_path, monkeypatch, failure):
    module, plan, execute, audit = harness
    factory = "_device_kernels" if module is kernels else "_probe_model"
    original = getattr(module, factory)

    def broken(device):
        if failure == "prepare":
            raise KeyboardInterrupt("injected preparation failure")
        value = original(device)
        if module is kernels:
            attention, scatter = value

            def bad(*args):
                output = attention(*args)
                if output.numel():
                    output[0, 0, 0] = torch.nan
                return output

            return bad, scatter
        recurrent = value._recurrent_tensor

        def bad(hidden, view):
            output = recurrent(hidden, view)
            value.observed[view.row_count]["layer0/attention"][1, 0, 0] = torch.nan
            return output

        value._recurrent_tensor = bad
        return value

    monkeypatch.setattr(module, factory, broken)
    for row in plan["execution_order"]:
        result = execute(
            plan,
            row["evaluation_id"],
            tmp_path,
            device="cpu",
            deadline_ns=time.perf_counter_ns() + 600 * 10**9,
        )
        if not result["passed"]:
            break
    else:
        pytest.fail("injected failure was not observed")
    path = tmp_path / "evaluations" / row["evaluation_id"] / "result.json"
    assert path.is_file() and result["errors"]
    checked = audit(tmp_path, plan, expected_device="cpu", allow_prefix=True)
    assert checked["valid_prefix"] and not checked["passed"] and not checked["complete"], checked
    assert checked["known_required_failure"] is (failure == "nonfinite")
