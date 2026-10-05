"""Cold group fork: immutable partial pages, first-token state and request RNG."""

from dataclasses import replace

import pytest
import torch

from tests.helpers import tiny_nanbeige_config, tiny_ouro_config
from vllm_rlt import CacheConfig, ExecutionConfig, ExitConfig, SamplingParams, SchedulerConfig
from vllm_rlt.core.kv_cache_manager import KVCacheManager
from vllm_rlt.engine.llm_engine import LLMEngine
from vllm_rlt.models import NanbeigeForCausalLM, OuroForCausalLM
from vllm_rlt.request import Stage


def make_model(family="ouro", seed=31):
    torch.manual_seed(seed)
    config = tiny_ouro_config if family == "ouro" else tiny_nanbeige_config
    cls = OuroForCausalLM if family == "ouro" else NanbeigeForCausalLM
    return cls(
        config(
            vocab_size=17,
            hidden_size=16,
            intermediate_size=32,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
        )
    ).eval()


def engine(
    model,
    enabled=True,
    *,
    concurrency=8,
    incremental=False,
    mode="refill",
    asynchronous=False,
    blocks=256,
    prefix=False,
    preemption=False,
):
    return LLMEngine(
        model,
        cache_config=CacheConfig(
            blocks, 8, incremental_allocation=incremental, enable_prefix_caching=prefix
        ),
        scheduler_config=SchedulerConfig(
            max_num_seqs=concurrency,
            prefill_chunk_size=3,
            group_prefill_fork=enabled,
            mode=mode,
            enable_preemption=preemption,
        ),
        execution_config=ExecutionConfig(async_scheduling=asynchronous),
        exit_config=ExitConfig(
            "trace", depths_by_request={"full": [model.config.total_ut_steps] * 12}
        ),
    )


def add(e, count, prompt, outputs=4, start=0):
    for i in range(start, start + count):
        e.add_request(
            str(i),
            prompt,
            SamplingParams(
                max_tokens=outputs,
                temperature=0.8,
                seed=100 + i,
                ignore_eos=True,
                logprobs=0,
                logprobs_mode="processed",
            ),
            trace_id="full",
        )


def drain(e):
    outputs, prefill = {}, 0
    for _ in range(4000):
        if not e.has_unfinished_requests():
            if e.scheduler.group_forks is not None:
                assert not e.scheduler.group_forks.groups and not e.scheduler.group_forks.by_request
            assert not e.cache_manager._allocations
            return outputs, prefill
        for out in e.step(final_only=True):
            assert out.finished and out.request_id not in outputs
            outputs[out.request_id] = out
        batch = e.last_schedule
        if batch is not None and batch.stage == Stage.PREFILL:
            prefill += batch.num_tokens
    pytest.fail("group scheduler did not drain")


def equal(actual, expected):
    assert actual.keys() == expected.keys()
    for key, out in actual.items():
        ref = expected[key]
        assert out.token_ids == ref.token_ids
        assert out.exit_depths == ref.exit_depths
        assert out.finish_reason == ref.finish_reason
        assert out.weight_version == ref.weight_version
        assert out.sampling_params == ref.sampling_params
        torch.testing.assert_close(
            torch.tensor(out.log_probs), torch.tensor(ref.log_probs), atol=1e-5, rtol=3e-5
        )


@pytest.mark.parametrize("family", ["ouro", "nanbeige"])
@pytest.mark.parametrize("length", [1, 7, 8, 9, 17])
@pytest.mark.parametrize("concurrency", [1, 4, 8])
def test_cold_group_executes_one_complete_prefill(family, length, concurrency):
    model = make_model(family)
    expected = engine(model, False, concurrency=concurrency)
    candidate = engine(model, concurrency=concurrency)
    prompt = [2 + i % 13 for i in range(length)]
    add(expected, 8, prompt)
    add(candidate, 8, prompt)
    reference, full_work = drain(expected)
    actual, work = drain(candidate)
    equal(actual, reference)
    assert full_work == 8 * length and work == length
    assert candidate.scheduler.group_forks.forked_requests == 7
    assert candidate.scheduler.group_forks.reused_tokens == 7 * length
    assert not any(candidate.cache_manager._refs)
    # Distinct completion seeds remain separate RNG streams after first-token reuse.
    assert len({tuple(out.token_ids) for out in actual.values()}) > 1


@pytest.mark.parametrize("mode", ["refill", "no_refill"])
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("incremental", [False, True])
@pytest.mark.parametrize("outputs", [1, 4])
def test_first_token_and_async_host_lifecycle(mode, asynchronous, incremental, outputs):
    model = make_model()
    values = []
    for enabled in (False, True):
        e = engine(
            model,
            enabled,
            mode=mode,
            asynchronous=asynchronous,
            incremental=incremental,
            prefix=True,
        )
        add(e, 8, [2, 3, 4, 5, 6, 7, 8, 9, 10], outputs)
        values.append(drain(e)[0])
        e.reset_prefix_cache()
        assert not any(e.cache_manager._refs)
    equal(values[1], values[0])


@pytest.mark.parametrize("length", [7, 8, 9])
def test_partial_page_detaches_only_when_written(length):
    c = KVCacheManager(2, 1, 4, 64, 8, 4)
    assert c.allocate("parent", length + 4)
    for layer in range(2):
        for depth in range(4):
            values = (
                torch.arange(length * 4, dtype=torch.float32).reshape(length, 1, 4) + depth + layer
            )
            c.write(
                layer,
                ["parent"] * length,
                [depth] * length,
                list(range(length)),
                values,
                values + 1000,
            )
    assert c.fork_prefix("parent", "child", length, length + 4)
    parent, child = (c._get_allocation(name) for name in ("parent", "child"))
    n = (length + 7) // 8
    assert all(a[:n] == b[:n] for a, b in zip(parent.block_tables, child.block_tables, strict=True))
    originals = [
        [c.read(layer, "parent", depth, length) for depth in range(4)] for layer in range(2)
    ]
    with pytest.raises(ValueError, match="immutable shared prefix"):
        c.write(0, ["child"], [0], [length - 1], torch.zeros(1, 1, 4), torch.zeros(1, 1, 4))
    assert c.ensure_capacity("child", length + 1)
    if length % 8:
        assert all(
            a[n - 1] != b[n - 1]
            for a, b in zip(parent.block_tables, child.block_tables, strict=True)
        )
    assert all(
        a[: length // 8] == b[: length // 8]
        for a, b in zip(parent.block_tables, child.block_tables, strict=True)
    )
    c.write(0, ["child"], [0], [length], torch.zeros(1, 1, 4), torch.ones(1, 1, 4))
    for layer in range(2):
        for depth in range(4):
            for got, old in zip(
                c.read(layer, "parent", depth, length), originals[layer][depth], strict=True
            ):
                torch.testing.assert_close(got, old, atol=0, rtol=0)
    c.free("parent")
    c.free("child")
    assert c.num_free_blocks == 64 and not any(c._refs)


@pytest.mark.parametrize("incremental", [False, True])
@pytest.mark.parametrize("preemption", [False, True])
def test_memory_pressure_still_drains(incremental, preemption):
    model = make_model()
    outputs = []
    for enabled in (False, True):
        e = engine(model, enabled, blocks=16, incremental=incremental, preemption=preemption)
        add(e, 8, [2, 3, 4, 5, 6, 7, 8, 9, 10], 9)
        outputs.append(drain(e)[0])
        assert not any(e.cache_manager._refs)
    equal(outputs[1], outputs[0])


def test_abort_leader_or_waiter_and_reuse_ids():
    e = engine(make_model())
    prompt = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    add(e, 8, prompt)
    e.step()
    assert e.scheduler.requests["0"].stage == Stage.PREFILL
    assert e.abort_request("0").finish_reason == "abort"
    e.abort_request("3")
    first, _ = drain(e)
    assert set(first) == {"1", "2", "4", "5", "6", "7"}
    add(e, 8, prompt)
    second, work = drain(e)
    assert work == len(prompt) and len(second) == 8
    assert not any(e.cache_manager._refs)


def test_idle_resume_at_prelude_makes_progress_without_group_fork():
    e = engine(make_model(), False, preemption=True)
    add(e, 1, [2, 3, 4])
    while e.scheduler.requests["0"].stage != Stage.PRELUDE:
        e.step()
    add(e, 1, [5], start=1)
    e.scheduler.selected_request_ids.clear()
    assert e.preemption.preempt(e.scheduler.requests["1"])
    e.abort_request("1")
    assert e.scheduler.requests["0"].stage == Stage.WAITING
    outputs, _ = drain(e)
    assert len(outputs["0"].token_ids) == 4


def test_finished_group_does_not_survive_new_policy():
    model = make_model()
    e = engine(model, prefix=True)
    prompt = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    add(e, 8, prompt)
    drain(e)
    replacement = make_model(seed=99)
    parameters = list(replacement.named_parameters())
    e.start_weight_update(1)
    assert not any(e.cache_manager._refs)
    e.update_weights(parameters[:1])
    with pytest.raises(RuntimeError, match="paused"):
        add(e, 1, prompt)
    e.update_weights(parameters[1:])
    e.finish_weight_update()
    add(e, 8, prompt)
    result, work = drain(e)
    assert work == len(prompt)
    cold = engine(replacement, False)
    add(cold, 8, prompt)
    expected, _ = drain(cold)
    equal(result, {key: replace(value, weight_version=1) for key, value in expected.items()})


def test_huginn_full_latent_prefix_is_not_shared():
    from tests.test_huginn import tiny_huginn_config
    from vllm_rlt.models import HuginnForCausalLM

    with pytest.raises(ValueError, match="Ouro/Nanbeige"):
        LLMEngine(
            HuginnForCausalLM(tiny_huginn_config()),
            scheduler_config=SchedulerConfig(group_prefill_fork=True),
        )


def test_no_spare_partial_page_falls_back_without_changing_output():
    model = make_model()
    results = []
    for enabled in (False, True):
        e = engine(model, enabled, blocks=8)
        add(e, 8, [2, 3, 4, 5, 6, 7, 8, 9, 10], 2)
        result, work = drain(e)
        results.append(result)
        assert work == 72
        if enabled:
            assert e.scheduler.group_forks.capacity_fallbacks == 1
            assert e.scheduler.group_forks.forked_requests == 0
    equal(results[1], results[0])


def test_snapshot_survives_leader_abort_but_releases_after_last_waiter():
    model = make_model()
    prompt = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    e = engine(model)
    add(e, 8, prompt)
    while not any(g.snapshot_id for g in e.scheduler.group_forks.groups.values()):
        e.step()
    e.abort_request("0")
    actual, work = drain(e)
    assert work == 0
    baseline = engine(model, False)
    add(baseline, 7, prompt, start=1)
    equal(actual, drain(baseline)[0])
    assert not any(e.cache_manager._refs)
    add(e, 2, prompt)
    while not any(g.snapshot_id for g in e.scheduler.group_forks.groups.values()):
        e.step()
    e.abort_request("1")
    assert not e.scheduler.group_forks.groups
    drain(e)
    assert not any(e.cache_manager._refs)


def test_different_prompt_groups_and_mixed_output_budgets():
    model = make_model()
    values = []
    for enabled in (False, True):
        e = engine(model, enabled)
        for i in range(9):
            e.add_request(
                str(i),
                [2, 3, 4, 5, 6, 7, 8, 9, 10 + i % 3],
                SamplingParams(
                    max_tokens=1 + i % 5,
                    seed=33 + i,
                    temperature=0.8,
                    ignore_eos=True,
                    logprobs=0,
                    logprobs_mode="processed",
                ),
                trace_id="full",
            )
        result, work = drain(e)
        values.append(result)
        assert work == (27 if enabled else 81)
    equal(values[1], values[0])


def test_fork_counts_as_fresh_admission_for_bounded_bypass():
    e = engine(make_model())
    e.scheduler.config = replace(e.scheduler.config, max_admission_bypasses=2)
    add(e, 2, list(range(2, 15)))
    add(e, 3, [5], start=2)
    e.step()
    assert e.scheduler.requests["0"].stage == Stage.PREFILL
    assert e.scheduler.requests["2"].stage == Stage.CODA
    e.scheduler._admit()
    assert e.scheduler.requests["1"].admission_bypasses == 2
    assert e.scheduler.requests["3"].stage == Stage.CODA
    assert e.scheduler.requests["4"].stage == Stage.WAITING
    drain(e)


def test_pending_snapshot_can_be_reclaimed_for_large_priority_request():
    e = engine(make_model(), blocks=32)
    add(e, 4, [2 + i % 13 for i in range(17)], 8)
    while not any(g.snapshot_id for g in e.scheduler.group_forks.groups.values()):
        e.step()
    e.scheduler.config = replace(e.scheduler.config, policy="priority", max_admission_bypasses=1)
    e.add_request(
        "large",
        [15] * 49,
        SamplingParams(max_tokens=1, priority=-10, ignore_eos=True),
        trace_id="full",
    )
    outputs, _ = drain(e)
    assert len(outputs) == 5
    assert e.scheduler.group_forks.capacity_fallbacks >= 1
    assert not any(e.cache_manager._refs)


def test_larger_follower_falls_back_after_short_leader_finishes():
    model = make_model()
    outputs = []
    for enabled in (False, True):
        e = engine(model, enabled, blocks=8)
        prompt = [2, 3, 4, 5, 6, 7, 8, 9, 10]
        add(e, 1, prompt, 1)
        add(e, 1, prompt, 2, start=1)
        outputs.append(drain(e)[0])
    equal(outputs[1], outputs[0])


@pytest.mark.parametrize("temperature", [0.0, 0.8])
def test_speculative_rounds_preserve_independent_first_sample(temperature):
    from vllm_rlt import SpeculativeConfig

    model = make_model()
    values = []
    for enabled in (False, True):
        e = LLMEngine(
            model,
            speculative_config=SpeculativeConfig(3),
            cache_config=CacheConfig(256, 8),
            scheduler_config=SchedulerConfig(group_prefill_fork=enabled),
        )
        for i in range(8):
            e.add_request(
                str(i),
                [2, 3, 4, 5, 6, 7, 8, 9, 10],
                SamplingParams(
                    max_tokens=1 if i == 0 else 9,
                    seed=33 + i,
                    temperature=temperature,
                    ignore_eos=True,
                    logprobs=0,
                    logprobs_mode="processed",
                ),
            )
        values.append(drain(e)[0])
    equal(values[1], values[0])


@pytest.mark.gpu
@pytest.mark.parametrize(
    "asynchronous,graph", [(False, False), (True, False), (False, True), (True, True)]
)
def test_cuda_copy_on_write_orders_alternating_caller_streams(asynchronous, graph):
    model = make_model().to(device="cuda", dtype=torch.bfloat16)
    callers = [torch.cuda.Stream(), torch.cuda.Stream()]
    for caller in callers:
        caller.wait_stream(torch.cuda.current_stream())
    values = []
    with torch.cuda.stream(callers[0]):
        for enabled in (False, True):
            e = LLMEngine(
                model,
                attention_backend="triton",
                cache_config=CacheConfig(256, 8, incremental_allocation=True),
                scheduler_config=SchedulerConfig(group_prefill_fork=enabled, prefill_chunk_size=3),
                execution_config=ExecutionConfig(
                    async_scheduling=asynchronous,
                    cuda_graphs=graph,
                    static_buffers=graph,
                    pad_to_power_of_two=graph,
                ),
                exit_config=ExitConfig("trace", depths_by_request={"full": [4] * 12}),
            )
            for _ in range(2):
                add(e, 8, [2, 3, 4, 5, 6, 7, 8, 9, 10], 9)
                result = {}
                for step in range(4000):
                    if not e.has_unfinished_requests():
                        break
                    with torch.cuda.stream(callers[step % 2 if enabled else 0]):
                        for out in e.step(final_only=True):
                            assert out.finished and out.request_id not in result
                            result[out.request_id] = out
                assert len(result) == 8 and not e.has_unfinished_requests()
                values.append(result)
                assert not any(e.cache_manager._refs)
    for caller in callers:
        caller.synchronize()
    for actual in values[1:]:
        equal(actual, values[0])
