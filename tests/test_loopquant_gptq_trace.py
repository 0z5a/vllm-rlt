import hashlib

import pytest
import torch

from loopquant.gptq_trace import HessianTrace
from vllm_rlt.core.kv_cache_manager import KVCacheManager


def test_first_all_matched_hessians_preserve_row_budgets_and_ignore_padding():
    trace = HessianTrace({"a": 3, "b": 3}, 4)
    cache = KVCacheManager(1, 1, 4, 16, 16, 4, device="cpu", dtype=torch.float32)
    for request in ("a", "b"):
        assert cache.allocate(request, 3)
    rows = [
        (request, loop, position)
        for request in ("a", "b")
        for loop in range(4)
        for position in range(3)
    ]
    order = torch.randperm(len(rows), generator=torch.Generator().manual_seed(3)).tolist()
    rows = [rows[index] for index in order]
    batch = cache._prepare_batch([r[0] for r in rows], [r[1] for r in rows], [r[2] for r in rows])
    trace.prepare(batch, len(rows) + 2)
    values = torch.arange((len(rows) + 2) * 4, dtype=torch.float32).reshape(-1, 4) / 10
    values[-2:] = float("nan")
    layer = torch.nn.Linear(4, 4, bias=False)
    names = [f"model.layers.0.self_attn.{name}_proj" for name in ("q", "k", "v")]
    for name in names:
        trace.record(name, "input", layer, values)
    canonical = names[0]
    expected_rows = {
        "all": list(range(len(rows))),
        "first": [i for i, row in enumerate(rows) if row[1] == 0],
        "matched": [
            i
            for i, row in enumerate(rows)
            if row[1]
            == int.from_bytes(hashlib.sha256(f"17:{row[0]}:{row[2]}".encode()).digest()[:8], "big")
            % 4
        ],
    }
    for policy, indices in expected_rows.items():
        selected = values[indices]
        stats = trace.hessians[canonical, policy]
        torch.testing.assert_close(stats.gram, selected.T @ selected)
        assert sum(stats.loop_rows.values()) == len(indices)
        assert sum(stats.loop_energy.values()) == pytest.approx(float(selected.square().sum()))
    summary = {row["policy"]: row for row in trace.summary()}
    assert [summary[policy]["total_rows"] for policy in ("first", "all", "matched")] == [6, 24, 6]
    assert all(row["physical_projections"] == sorted(names) for row in summary.values())
    assert trace.calls == 1
    for request in ("a", "b"):
        cache.free(request)


def test_incomplete_fixed_depth_collection_cannot_claim_matched_budget():
    trace = HessianTrace({"a": 1}, 2)
    cache = KVCacheManager(1, 1, 4, 4, 16, 2, device="cpu", dtype=torch.float32)
    assert cache.allocate("a", 1)
    for loop in range(2):
        trace.prepare(cache._prepare_batch(["a"], [loop], [0]), 1)
        trace.record(
            "model.layers.0.mlp.down_proj",
            "input",
            torch.nn.Linear(4, 4, bias=False),
            torch.ones(1, 4),
        )
    trace.hessians["model.layers.0.mlp.down_proj", "matched"].loop_rows.clear()
    with pytest.raises(ValueError, match="fixed-depth row budget"):
        trace.summary()
    cache.free("a")


def test_hessian_hooks_leave_native_generation_and_kv_drain_unchanged():
    from loopquant.bench import run_cohort
    from loopquant.trace import TracedOuroForCausalLM
    from tests.helpers import tiny_ouro_config
    from vllm_rlt import LLM, CacheConfig, SchedulerConfig
    from vllm_rlt.models.ouro import OuroForCausalLM

    torch.manual_seed(17)
    prompts = [[5, 7, 2], [3, 9]]
    trace = HessianTrace({str(index): len(tokens) for index, tokens in enumerate(prompts)}, 4)
    traced = TracedOuroForCausalLM(tiny_ouro_config(), trace)
    plain = OuroForCausalLM(tiny_ouro_config())
    plain.load_state_dict(traced.state_dict())
    outputs = []
    for model in (plain, traced):
        engine = LLM(
            model,
            cache_config=CacheConfig(num_blocks=64, block_size=2),
            scheduler_config=SchedulerConfig(max_num_seqs=2, max_num_batched_tokens=8),
        ).engine
        records = []
        result = run_cohort(
            engine, prompts, concurrency=2, output_tokens=3, loops=4, records=records
        )
        assert result.failed_requests == result.kv_used_after_drain == 0
        outputs.append([record.token_ids for record in records])
    assert outputs[0] == outputs[1]
    for row in trace.summary():
        assert row["total_rows"] == (36 if row["policy"] == "all" else 9)
    assert not trace.rows
