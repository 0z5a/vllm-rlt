"""Tiny CPU checks of the actual pinned cache, mask creation and request lifecycle."""

import importlib.metadata
import importlib.util
import socket
import time

import pytest
import torch

from vllm_lt.benchmarks.q2_external_driver import execute_case, pool_descriptor, require_empty
from vllm_lt.config import CacheConfig, SchedulerConfig
from vllm_lt.engine.llm_engine import LLMEngine
from vllm_lt.models.config import OuroConfig
from vllm_lt.models.ouro import OuroForCausalLM
from vllm_lt.validation.official import OfficialOuroReference
from vllm_lt.validation.official_cached import (
    OfficialOuroCachedReference,
    cached_official_provenance,
)


def _compatible():
    try:
        return (
            importlib.metadata.version("transformers") == "4.55.0"
            and importlib.util.find_spec("kernels") is None
        )
    except importlib.metadata.PackageNotFoundError:
        return False


requires_official = pytest.mark.skipif(
    not _compatible(), reason="requires prepared Transformers 4.55.0 reference env"
)


@pytest.fixture(autouse=True)
def no_device_or_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("cached reference CPU tests must not use CUDA or network")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    for name in ("is_available", "device_count", "current_device", "init", "_lazy_init"):
        monkeypatch.setattr(torch.cuda, name, forbidden)


@pytest.fixture
def model(request):
    torch.manual_seed(17)
    return OuroForCausalLM(
        OuroConfig.tiny(
            hidden_size=16, intermediate_size=32, num_attention_heads=2, max_position_embeddings=256
        )
    ).to(dtype=getattr(request, "param", torch.float32))


@pytest.fixture
def reference(model):
    ref = OfficialOuroCachedReference(model.config.to_dict(), dict(model.named_parameters()))
    yield ref
    ref.close(completion_confirmed=True)


def test_cached_provenance_is_distinct_from_q1():
    from vllm_lt.validation.official import official_provenance

    value = cached_official_provenance()
    assert value["cache_slots"] == 96
    assert value["use_cache"] is True and official_provenance()["use_cache"] is False
    assert value["dtype"] == "torch.float32"
    assert value["exit_at_step"] == 3 and value["total_ut_steps"] == 4
    assert value["logits_to_keep"] == 1 and value["attention_implementation"] == "eager"
    assert len(value["cache_shim_sha256"]) == 64


@requires_official
@pytest.mark.parametrize("model", [torch.float32, torch.bfloat16], indirect=True)
def test_cached_greedy_matches_uncached_prefixes_and_preserves_shared_weights(model, reference):
    caller = dict(model.named_parameters())
    original = {name: value.detach().clone() for name, value in caller.items()}
    flags = {name: value.requires_grad for name, value in caller.items()}
    uncached = OfficialOuroReference(model.config.to_dict(), caller)
    assert reference.config.use_cache is True and uncached.config.use_cache is False
    for name, parameter in reference.model.named_parameters():
        assert parameter.data_ptr() == caller[name].data_ptr()
        assert parameter.dtype == caller[name].dtype and not parameter.requires_grad
    prompt, generated = [1, 2, 3, 4], []
    for index in range(5):
        logits = reference.start(prompt, 5) if index == 0 else reference.advance(generated[-1])
        expected = uncached.predict(prompt + generated, [])[0]
        assert logits.shape == (model.config.vocab_size,)
        assert logits.dtype == next(model.parameters()).dtype
        torch.testing.assert_close(logits, expected, atol=1e-6, rtol=1e-5)
        assert logits.argmax().item() == expected.argmax().item()
        generated.append(logits.argmax().item())
    reference.complete(completion_confirmed=True)
    state = reference.snapshot(inspect_cache=True, check_finite=True)
    assert state["status"] == "finished"
    assert state["forward_calls"] == state["output_count"] == 5
    assert state["final_summary"]["lengths"] == [8] * 8
    assert state["final_summary"]["distinct_storage"] is True
    assert state["final_summary"]["all_finite"] is True
    reference.reset()
    assert reference.cache_slots == 0
    uncached.close()
    for name, tensor in caller.items():
        assert tensor.requires_grad == flags[name]
        torch.testing.assert_close(tensor, original[name], rtol=0, atol=0)


@requires_official
@pytest.mark.parametrize("layers", [2, 24])
def test_actual_depth_slot_updates_causal_masks_and_one_lm_head(layers, monkeypatch):
    config = OuroConfig.tiny(
        hidden_size=16, intermediate_size=32, num_attention_heads=2, num_hidden_layers=layers
    )
    model = OuroForCausalLM(config)
    ref = OfficialOuroCachedReference(config.to_dict(), model.state_dict())
    from vllm_lt.validation.reference_code.modeling_ouro import UniversalTransformerCache

    assert ref._cache_type.update is UniversalTransformerCache.update
    updates, masks, calls, head_rows, loops = [], [], [], [], []
    original_update = ref._cache_type.update
    original_mask = ref._cache_type.get_mask_sizes

    def update(cache, key, value, layer_idx, cache_kwargs=None):
        updates.append(layer_idx)
        return original_update(cache, key, value, layer_idx, cache_kwargs)

    def mask(cache, positions, layer_idx):
        result = original_mask(cache, positions, layer_idx)
        masks.append(result)
        return result

    monkeypatch.setattr(ref._cache_type, "update", update)
    monkeypatch.setattr(ref._cache_type, "get_mask_sizes", mask)

    def before(module, args, kwargs):
        calls.append(
            {
                "inputs": kwargs["input_ids"].tolist(),
                "positions": kwargs["position_ids"].tolist(),
                "cache_positions": kwargs["cache_position"].tolist(),
                "mask_shape": list(kwargs["attention_mask"].shape),
                "use_cache": kwargs["use_cache"],
                "exit": kwargs["exit_at_step"],
            }
        )

    handle = ref.model.register_forward_pre_hook(before, with_kwargs=True)
    head = ref.model.lm_head.register_forward_pre_hook(
        lambda module, args: head_rows.append(args[0].shape[:2])
    )
    layer = ref.model.model.layers[0].register_forward_hook(lambda *args: loops.append(1))
    ref.start([1, 2, 3], 3)
    ref.advance(4)
    ref.advance(5)
    for hook in (handle, head, layer):
        hook.remove()
    assert updates == list(range(4 * layers)) * 3
    assert masks == [(3, 0), (4, 0), (5, 0)]
    assert len(loops) == 12 and head_rows == [(1, 1)] * 3
    assert [value["inputs"] for value in calls] == [[[1, 2, 3]], [[4]], [[5]]]
    assert [value["positions"] for value in calls] == [[[0, 1, 2]], [[3]], [[4]]]
    assert [value["cache_positions"] for value in calls] == [[0, 1, 2], [3], [4]]
    assert [value["mask_shape"] for value in calls] == [[1, 3], [1, 4], [1, 5]]
    assert all(value["use_cache"] is True and value["exit"] == 3 for value in calls)
    summary = ref.snapshot(inspect_cache=True, check_finite=True)["final_summary"]
    assert summary["slot_count"] == summary["max_cache_size"] == 4 * layers
    assert summary["lengths"] == [5] * (4 * layers)
    assert summary["distinct_storage"] is True and summary["all_finite"] is True
    ref.complete(completion_confirmed=True)
    ref.close()


@requires_official
def test_w1_position_accounting_has_63_advances_and_no_final_token_forward(reference):
    prompt = [1, 2] * 64
    logits = reference.start(prompt, 64)
    for _ in range(63):
        logits = reference.advance(int(logits.argmax()))
    state = reference.snapshot(inspect_cache=True)
    assert state["output_count"] == state["forward_calls"] == 64
    assert state["expected_position"] == 191
    assert [item["input_count"] for item in state["calls"]] == [128] + [1] * 63
    assert [item["position"] for item in state["calls"]] == [0] + list(range(128, 191))
    assert [item["output_index"] for item in state["calls"]] == list(range(64))
    assert state["final_summary"]["lengths"] == [191] * 8
    with pytest.raises(RuntimeError, match="awaiting_completion"):
        reference.advance(int(logits.argmax()))
    assert reference.forward_calls == 64
    reference.complete(completion_confirmed=True)


@requires_official
@pytest.mark.parametrize(
    "prompt,outputs", [([], 1), ([True], 1), ([64], 1), ([1], True), ([1], 0), ([1] * 256, 2)]
)
def test_invalid_start_is_rejected_before_mutation(reference, prompt, outputs):
    before = reference.snapshot()
    with pytest.raises(ValueError):
        reference.start(prompt, outputs)
    assert reference.snapshot() == before and reference.cache_slots == 0


@requires_official
def test_reset_completion_and_close_boundaries(reference):
    with pytest.raises(RuntimeError, match="ready"):
        reference.advance(1)
    reference.start([1, 2], 2)
    old_cache = reference.cache
    before = reference.snapshot()
    with pytest.raises(ValueError):
        reference.advance(True)
    with pytest.raises(RuntimeError, match="active"):
        reference.start([3], 1)
    with pytest.raises(RuntimeError, match="confirmed"):
        reference.reset()
    with pytest.raises(RuntimeError, match="confirmed"):
        reference.complete(completion_confirmed=True)
    assert reference.snapshot() == before
    reference.advance(3)
    with pytest.raises(RuntimeError, match="confirmed"):
        reference.complete(completion_confirmed=False)
    reference.complete(completion_confirmed=True)
    reference.reset()
    assert old_cache.key_cache == old_cache.value_cache == []
    reference.start([4], 1)
    assert reference.cache is not old_cache
    assert reference.snapshot(inspect_cache=True)["final_summary"]["lengths"] == [1] * 8
    reference.complete(completion_confirmed=True)
    reference.close()
    reference.close()
    assert reference.cache_slots == 0 and reference.model is None
    for call in (lambda: reference.start([1], 1), reference.reset, lambda: reference.advance(1)):
        with pytest.raises(RuntimeError, match="closed"):
            call()


@requires_official
def test_partial_forward_failure_preserves_error_and_requires_confirmed_cleanup(
    reference, monkeypatch
):
    reference.start([1, 2], 2)
    error = RuntimeError("injected after first slot append")

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(reference.model.model.layers[1], "forward", fail)
    with pytest.raises(RuntimeError) as observed:
        reference.advance(3)
    assert observed.value is error
    assert reference.status == "failed" and reference.forward_calls == 2
    assert reference.output_count == 1 and reference.expected_position == 2
    assert reference.cache.get_seq_length(0) == 3 and reference.cache.get_seq_length(1) == 2
    with pytest.raises(RuntimeError, match="failed"):
        reference.advance(4)
    with pytest.raises(RuntimeError, match="confirmed"):
        reference.close()
    reference.reset(completion_confirmed=True)
    assert reference.cache_slots == 0 and reference.status == "ready"


@requires_official
def test_rejects_fp16_and_sliding_before_cached_model_use(model):
    with pytest.raises(ValueError, match="supported reference dtype"):
        OfficialOuroCachedReference(model.config.to_dict(), model.half().state_dict())
    config = model.config.to_dict()
    config["layer_types"] = ["sliding_attention"] * config["num_hidden_layers"]
    with pytest.raises(ValueError, match="full attention"):
        OfficialOuroCachedReference(config, model.state_dict())


@requires_official
def test_snapshots_are_detached_and_detailed_checks_are_opt_in(reference, monkeypatch):
    reference.start([1, 2], 1)

    def forbidden(*args, **kwargs):
        raise AssertionError("finite scans must be explicit and outside delivery")

    monkeypatch.setattr(torch, "isfinite", forbidden)
    value = reference.snapshot(inspect_cache=True)
    assert value["final_summary"]["all_finite"] is None
    value["calls"][0]["position"] = 999
    assert reference.snapshot()["calls"][0]["position"] == 0
    with pytest.raises(ValueError, match="inspection"):
        reference.snapshot(check_finite=True)
    with pytest.raises(AssertionError, match="explicit"):
        reference.snapshot(inspect_cache=True, check_finite=True)


@requires_official
def test_cache_replacement_is_rejected_before_forward(reference):
    reference.start([1], 2)
    reference.cache = reference._cache_type(max_cache_size=8)
    with pytest.raises(RuntimeError, match="identity or contiguous prefix"):
        reference.advance(2)
    assert reference.status == "failed" and reference.forward_calls == 1


@requires_official
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_real_cpu_native_and_official_execute_case_share_weights_and_release_requests(
    monkeypatch, dtype
):
    started = time.perf_counter_ns()
    deadline = started + 120 * 10**9

    # Only GPU timing/memory bookkeeping is replaced. Model, attention, cache,
    # scheduler, sampling, collector and adapter lifecycle remain real CPU code.
    for name in (
        "synchronize",
        "reset_peak_memory_stats",
        "memory_allocated",
        "memory_reserved",
        "max_memory_allocated",
        "max_memory_reserved",
    ):
        monkeypatch.setattr(torch.cuda, name, lambda *args, **kwargs: 0)

    torch.manual_seed(17)
    config = OuroConfig.tiny(
        vocab_size=8,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=24,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=256,
    )
    model = OuroForCausalLM(config).to(dtype=dtype).eval()
    model.requires_grad_(False)
    engine = LLMEngine(
        model,
        cache_config=CacheConfig(num_blocks=48, block_size=16),
        scheduler_config=SchedulerConfig(max_num_seqs=1, max_num_batched_tokens=128),
        attention_backend="torch",
    )
    weights = dict(model.named_parameters())
    official = OfficialOuroCachedReference(config.to_dict(), weights)
    assert all(
        value.data_ptr() == weights[name].data_ptr()
        for name, value in official.model.named_parameters()
    )
    assert all(value.device.type == "cpu" for value in model.parameters())
    pool = pool_descriptor(engine)
    prompt = [1 + index % 7 for index in range(128)]
    plan = {
        "plan_sha256": "cpu-integration-only-no-device-plan",
        "workload": {"requests": [{"prompt_token_ids": prompt, "max_output_tokens": 64}]},
        "contract": {
            "engine": {
                "sampling": {
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "top_k": -1,
                    "seed": 0,
                    "min_loops": 4,
                    "max_loops": 4,
                    "exit_threshold": 1.0,
                    "ignore_eos": True,
                }
            }
        },
    }
    results = []
    try:
        for implementation, bound in (("native", 380), ("official", 64)):
            result = execute_case(
                plan,
                {"implementation_id": implementation, "phase": "feasibility", "max_steps": bound},
                engine,
                official,
                started_ns=time.perf_counter_ns(),
                deadline_ns=deadline,
            )
            results.append(result)
            assert result["status"] == "complete" and not result["failures"]
            assert result["before_request"] == result["cleanup"] == require_empty(engine, official)
            assert result["cleanup"] == {
                "native_requests": 0,
                "native_used_blocks": 0,
                "official_cache_slots": 0,
            }
            assert pool_descriptor(engine) == pool
            assert official.status == "ready" and official.cache is None
            row = result["requests"][0]
            assert row["finished"] and row["finish_reason"] == "length"
            assert len(row["token_ids"]) == 64 and row["exit_depths"] == [4] * 64
            assert row["token_timestamps_ns"] == sorted(row["token_timestamps_ns"])
            assert result["arrival_ns"] <= row["token_timestamps_ns"][0]
            assert row["token_timestamps_ns"][-1] <= result["synchronized_ns"]
            assert result["metrics"]["generated_tokens"] == 64
            assert len([e for e in result["events"] if e["kind"] == "token_emitted"]) == 64

        native, external = results
        # FP32 retains the historical exact-generation gate. BF16 may change
        # argmax near ties; the v2 benchmark reports agreement separately.
        if dtype == torch.float32:
            assert native["requests"][0]["token_ids"] == external["requests"][0]["token_ids"]
        assert all(value.dtype == dtype for value in model.parameters())
        assert native["counts"]["steps"] == 380
        assert native["counts"]["stage_counts"] == {
            "prefill": 1,
            "prelude": 63,
            "recurrent": 252,
            "coda": 64,
        }
        # The hook also observes four recurrent traversals inside prefill.
        assert native["feasibility"]["finite_checks"] == {"recurrent": 256, "coda": 64}
        assert native["official_cache"] is None
        assert external["counts"]["steps"] == external["counts"]["official_logits_checked"] == 64
        cached = external["official_cache"]
        snapshot, summary = cached["snapshot"], cached["final_summary"]
        assert snapshot["final_summary"] == summary
        assert snapshot["status"] == "awaiting_completion"
        assert snapshot["output_count"] == snapshot["forward_calls"] == 64
        assert snapshot["cache_slots"] == summary["slot_count"] == summary["max_cache_size"] == 96
        assert summary["lengths"] == [191] * 96
        assert summary["key_shapes"] == summary["value_shapes"] == [[1, 1, 191, 8]] * 96
        assert summary["distinct_storage"] is True and summary["all_finite"] is True
        assert summary["device"] == "cpu" and summary["dtype"] == str(dtype)
        assert [r["position"] for r in snapshot["calls"]] == [0] + list(range(128, 191))
        assert [r["input_count"] for r in snapshot["calls"]] == [128] + [1] * 63
        assert [r["last_input_position"] for r in cached["calls"]] == list(range(127, 191))
        assert [r["token_id"] for r in cached["calls"]] == external["requests"][0]["token_ids"]
        assert time.perf_counter_ns() < deadline
    finally:
        official.close(completion_confirmed=True)
