"""Physical stage allocation, native Huginn parity and cache lifecycle."""

from copy import deepcopy

import pytest
import torch

from tests.reference.huginn import dense_huginn_reference
from tests.test_huginn import tiny_huginn_config
from vllm_rlt import CacheConfig, SamplingParams, SchedulerConfig
from vllm_rlt.config import ExecutionConfig, ExitConfig
from vllm_rlt.core.kv_group import KVGroupSpec
from vllm_rlt.core.memory import make_cache_manager, plan_cache
from vllm_rlt.core.stage_kv_cache import StageKVCacheManager
from vllm_rlt.engine.llm_engine import LLMEngine
from vllm_rlt.models.huginn import HuginnForCausalLM


def cache(loops=3, bundles=4):
    return StageKVCacheManager(
        num_layers=4,
        num_kv_heads=1,
        head_dim=2,
        num_blocks=loops * bundles,
        block_size=2,
        max_loops=loops,
        recurrent_layers=(1, 2),
        groups=(
            KVGroupSpec("prelude", (0,), 1),
            KVGroupSpec("core", (1, 2), loops),
            KVGroupSpec("coda", (3,), 1),
        ),
    )


def test_huginn_r32_allocates_132_actual_planes():
    manager = StageKVCacheManager(
        num_layers=8,
        num_kv_heads=1,
        head_dim=2,
        num_blocks=96,
        block_size=2,
        max_loops=32,
        recurrent_layers=(2, 3, 4, 5),
        groups=(
            KVGroupSpec("prelude", (0, 1), 1),
            KVGroupSpec("core", (2, 3, 4, 5), 32),
            KVGroupSpec("coda", (6, 7), 1),
        ),
    )
    assert sum(group.planes for group in manager.groups) == 132
    assert manager.key_cache.shape == (96, 4, 2, 1, 2)
    assert manager.boundary_keys.shape == (3, 4, 2, 1, 2)
    dense_bytes = 2 * 96 * 8 * 2 * 1 * 2 * 4
    assert manager.allocated_bytes == dense_bytes * 132 // 256
    assert manager.num_free_blocks == 96


def test_atomic_growth_release_stale_batch_and_request_id_reuse():
    manager = cache(bundles=3)
    assert manager.allocate("first", 6, initial_tokens=2)
    assert manager.allocate("second", 4)
    before = manager._get_allocation("first").block_tables
    prepared = manager._prepare_batch(["first"], [0], [0])
    assert not manager.ensure_capacity("first", 4)
    assert manager._get_allocation("first").block_tables == before
    manager.free("second")
    assert manager.ensure_capacity("first", 6)
    tables = manager._get_allocation("first").block_tables
    assert all(all(block % 3 == depth for block in table) for depth, table in enumerate(tables))
    manager.pin_transfer("first", "reader")
    manager.free("first")
    assert manager.num_free_blocks == 0
    manager.unpin_transfer("first", "reader")
    assert manager.num_free_blocks == 9
    assert manager.allocate("first", 2)
    with pytest.raises(RuntimeError, match="stale"):
        manager._write_prepared(0, prepared, torch.ones(1, 1, 2), torch.ones(1, 1, 2))
    manager.free("first")
    assert manager.num_free_blocks == 9


def test_snapshot_restores_boundary_and_core_after_relocation():
    manager = cache(bundles=4)
    assert manager.allocate("first", 4)
    for layer in range(4):
        for depth in range(3) if layer in (1, 2) else (0,):
            values = torch.full((4, 1, 2), 10.0 * layer + depth)
            manager.write(layer, ["first"] * 4, [depth] * 4, range(4), values, values + 1)
    allocation = manager._get_allocation("first")
    old = [block for table in allocation.block_tables for block in table]
    written = deepcopy(allocation.written)
    snapshot = manager.snapshot_pages(old)
    manager.free("first")
    assert manager.allocate("blocker", 2)
    assert manager.allocate("restored", 4)
    allocation = manager._get_allocation("restored")
    new = [block for table in allocation.block_tables for block in table]
    assert old != new
    manager.restore_pages(new, snapshot)
    allocation.written = written
    for layer in range(4):
        for depth in range(3) if layer in (1, 2) else (0,):
            keys, values = manager.read(layer, "restored", depth)
            assert torch.equal(keys, torch.full_like(keys, 10.0 * layer + depth))
            assert torch.equal(values, keys + 1)
    manager.free("blocker")
    manager.free("restored")
    assert manager.num_free_blocks == 12


@pytest.mark.parametrize("loops", [1, 3, 32])
@pytest.mark.parametrize("parallel", [False, True])
def test_native_stages_match_independent_dense(loops, parallel):
    torch.manual_seed(42)
    model = HuginnForCausalLM(
        tiny_huginn_config(
            mean_recurrence=loops,
            n_layers=8,
            n_layers_in_prelude=2,
            n_layers_in_recurrent_block=4,
            n_layers_in_coda=2,
        )
    ).eval()
    manager = make_cache_manager(
        model, CacheConfig(stage_aware=True, block_size=2), max(24, loops * 6), "torch"
    )
    tokens = torch.tensor([4, 7, 9, 3, 12])
    state = torch.linspace(-0.2, 0.2, len(tokens) * model.config.n_embd).reshape(len(tokens), -1)
    depths, expected = dense_huginn_reference(model, tokens, state)
    assert manager.allocate("request", len(tokens))
    chunks = [list(range(len(tokens)))] if parallel else [[i] for i in range(len(tokens))]
    for positions in chunks:
        boundary = manager._prepare_batch(
            ["request"] * len(positions), [0] * len(positions), positions
        )
        hidden = model.prelude_prepared(
            tokens[positions], boundary, manager, initial_state=state[positions]
        )
        for depth, wanted in enumerate(depths):
            core = manager._prepare_batch(
                ["request"] * len(positions), [depth] * len(positions), positions
            )
            hidden, _ = model.recurrent_prepared(hidden, core, manager)
            torch.testing.assert_close(
                hidden[:, : model.config.n_embd], wanted[positions], atol=3e-6, rtol=3e-5
            )
        # Coda must still be writable after recurrent finalization.
        for position in positions:
            manager.finalize_token("request", position, loops - 1)
        logits = model.coda_prepared(hidden, boundary, manager)
        torch.testing.assert_close(logits, expected[positions], atol=3e-6, rtol=3e-5)
    manager.free("request")
    assert manager.num_used_blocks == 0


@pytest.mark.parametrize("incremental", [False, True])
def test_native_engine_outputs_and_reuse_match_rectangular(incremental):
    torch.manual_seed(42)
    model = HuginnForCausalLM(tiny_huginn_config()).eval()
    results = []
    for grouped in (False, True):
        engine = LLMEngine(
            model,
            cache_config=CacheConfig(
                num_blocks=96, block_size=2, stage_aware=grouped, incremental_allocation=incremental
            ),
            scheduler_config=SchedulerConfig(
                max_num_seqs=4, max_num_batched_tokens=4, prefill_chunk_size=2
            ),
        )
        rounds = []
        for _ in range(2):
            torch.manual_seed(123)
            for index, prompt in enumerate(([4, 7, 3], [9, 2], [1, 5, 8, 4])):
                engine.add_request(
                    str(index),
                    prompt,
                    SamplingParams(
                        max_tokens=4,
                        min_loops=3,
                        max_loops=3,
                        ignore_eos=True,
                        temperature=0.8,
                        seed=17 + index,
                        latent_seed=117 + index,
                        logprobs=0,
                        logprobs_mode="processed",
                    ),
                )
            finished = {}
            while engine.has_unfinished_requests():
                for output in engine.step(final_only=True):
                    if output.finished:
                        finished[output.request_id] = (
                            output.token_ids,
                            output.exit_depths,
                            output.log_probs,
                        )
            assert engine.cache_manager.num_used_blocks == 0
            rounds.append(finished)
        results.append(rounds)
    assert results[0] == results[1]


def test_byte_budget_rounds_by_complete_physical_bundles():
    model = HuginnForCausalLM(tiny_huginn_config()).eval()
    unit = 2 * (2 * 3 + 2) * 2 * 4 * 8 * 4
    settings = CacheConfig(
        stage_aware=True, block_size=2, kv_cache_memory_bytes=3 * unit + unit // 2
    )
    blocks, plan = plan_cache(model, settings, SchedulerConfig(), ExecutionConfig(), "torch")
    manager = make_cache_manager(model, settings, blocks, "torch")
    assert blocks == 9
    assert plan["bytes_per_page_bundle"] == unit
    assert manager.allocated_bytes == 3 * unit <= settings.kv_cache_memory_bytes


def test_mixed_exit_depths_preserve_boundary_planes_and_sampling():
    torch.manual_seed(42)
    model = HuginnForCausalLM(tiny_huginn_config(mean_recurrence=32)).eval()
    results = []
    for grouped in (False, True):
        engine = LLMEngine(
            model,
            cache_config=CacheConfig(num_blocks=512, block_size=2, stage_aware=grouped),
            exit_config=ExitConfig("trace", depths_by_request={"a": [32, 1, 17, 2]}),
        )
        engine.add_request(
            "a",
            [4, 7, 3],
            SamplingParams(
                max_tokens=4,
                min_loops=1,
                max_loops=32,
                ignore_eos=True,
                seed=3,
                latent_seed=4,
                logprobs=0,
            ),
        )
        outputs = []
        for _ in range(256):
            if not engine.has_unfinished_requests():
                break
            outputs.extend(engine.step(final_only=True))
        assert not engine.has_unfinished_requests()
        assert len(outputs) == 1 and outputs[0].exit_depths == [32, 1, 17, 2]
        results.append((outputs[0].token_ids, outputs[0].log_probs))
        assert engine.cache_manager.num_used_blocks == 0
    assert results[0] == results[1]


def test_preemption_restores_full_native_generation_and_scores():
    torch.manual_seed(41)
    model = HuginnForCausalLM(tiny_huginn_config()).eval()
    results = []
    params = SamplingParams(
        max_tokens=7,
        temperature=0.8,
        seed=45,
        latent_seed=29,
        min_loops=3,
        max_loops=3,
        ignore_eos=True,
        logprobs=0,
    )
    for suspend in (False, True):
        engine = LLMEngine(
            model,
            cache_config=CacheConfig(num_blocks=96, block_size=2, stage_aware=True),
            scheduler_config=SchedulerConfig(enable_preemption=True),
        )
        engine.add_request("a", [1, 2, 3], params)
        while len(engine.scheduler.requests["a"].generated_token_ids) < 3:
            engine.step()
        request = engine.scheduler.requests["a"]
        generator = request.generator
        rng = generator.get_state().clone()
        if suspend:
            engine.add_request("b", [5], SamplingParams(max_tokens=1, min_loops=3, max_loops=3))
            engine.scheduler.selected_request_ids.clear()
            assert engine.preemption.preempt(engine.scheduler.requests["b"])
            snapshot = engine.preemption.snapshots["a"]
            assert len(snapshot["pools"]) == 2
            assert request.generator is generator and torch.equal(generator.get_state(), rng)
        final = {}
        for _ in range(512):
            if not engine.has_unfinished_requests():
                break
            for output in engine.step(final_only=True):
                if output.finished:
                    final[output.request_id] = (
                        output.token_ids,
                        output.exit_depths,
                        output.log_probs,
                    )
        assert not engine.has_unfinished_requests()
        assert engine.cache_manager.num_used_blocks == 0
        assert not engine.preemption.snapshots
        if suspend:
            assert engine.preemption.resumptions == 1
        results.append(final["a"])
    assert results[0] == results[1]


def test_graph_device_proxy_addresses_use_core_layer_mapping():
    from vllm_rlt.worker.cuda_graph import _DeviceCache

    manager = cache()
    assert manager.allocate("first", 2)
    batch = manager._prepare_batch(["first", "first"], [2, 2], [0, 1])
    keys = torch.tensor([[[1.0, 2.0]], [[3.0, 4.0]]])
    values = keys + 10
    proxy = _DeviceCache(manager)
    proxy._write_prepared(2, batch, keys, values)
    output = proxy._attend_prepared(2, batch, keys)
    expected = torch.nn.functional.scaled_dot_product_attention(
        keys.transpose(0, 1)[None],
        keys.transpose(0, 1)[None],
        values.transpose(0, 1)[None],
        is_causal=True,
    )[0].transpose(0, 1)
    torch.testing.assert_close(output, expected)
    # Device proxy validation is CPU-only here; real CUDA capture is a separate test.
    with pytest.raises(ValueError, match="depth zero"):
        manager._write_prepared(0, batch, keys, values)
