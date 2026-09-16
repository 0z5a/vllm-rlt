"""Official paged attention: ragged prefixes, physical strides and async exits."""

from dataclasses import replace

import pytest
import torch

from vllm_lt import CacheConfig, ExecutionConfig, ExitConfig, SamplingParams, SchedulerConfig
from vllm_lt.engine.llm_engine import LLMEngine
from vllm_lt.kernels.flash_attention import FlashPagedAttention, select_version
from vllm_lt.kernels.paged_attention import torch_paged_attention
from vllm_lt.models import OuroConfig, OuroForCausalLM


@pytest.mark.parametrize(
    "sm,expected", [((8, 0), 2), ((8, 9), 2), ((9, 0), 3), ((10, 0), 4), ((10, 3), 4), ((12, 0), 4)]
)
def test_architecture_selection(sm, expected):
    assert select_version(sm, "flash_attn") == expected


def test_unsupported_architecture_and_dtype():
    with pytest.raises(ValueError, match="not supported"):
        select_version((10, 3), "flash_attn_3")
    with pytest.raises(ValueError, match="not supported"):
        select_version((7, 5), "flash_attn")
    with pytest.raises(ValueError, match="NVIDIA"):
        FlashPagedAttention(torch.device("cpu"), torch.bfloat16, 128, 16)


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("heads", [2, 4])
def test_ragged_strided_paged_attention(dtype, heads):
    torch.manual_seed(9)
    # Layer slicing leaves a non-contiguous block stride, as in the real cache.
    keys = torch.randn(12, 3, 16, 2, 128, device="cuda", dtype=dtype)[:, 1]
    values = torch.randn(12, 3, 16, 2, 128, device="cuda", dtype=dtype)[:, 1]
    q = torch.randn(4, heads, 128, device="cuda", dtype=dtype)
    tables = torch.tensor(
        [[7, 2, 9], [3, 0, 0], [1, 8, 6], [0, 0, 0]], device="cuda", dtype=torch.int32
    )
    lengths = torch.tensor([35, 1, 47, 0], device="cuda", dtype=torch.int32)
    attention = FlashPagedAttention(q.device, dtype, 128, 16)
    expected = torch_paged_attention(q, keys, values, tables, lengths)
    actual = attention(q, keys, values, tables, lengths)
    torch.testing.assert_close(
        actual, expected, atol=0.015 if dtype == torch.bfloat16 else 0.002, rtol=0.02
    )
    assert torch.isfinite(actual).all()
    assert torch.count_nonzero(actual[-1]) == 0


@pytest.mark.gpu
@pytest.mark.parametrize("layout", ["last_exited", "shared"])
@pytest.mark.parametrize("static", [False, True])
def test_flash_sync_async_mixed_depths(layout, static):
    torch.manual_seed(123)
    config = replace(OuroConfig.tiny(), head_dim=64)
    model = OuroForCausalLM(config).to(device="cuda", dtype=torch.bfloat16)
    traces = {str(i): [4, 2 + i % 3, 4, 2, 3, 4] for i in range(3)}
    outputs = []
    for backend, asynchronous, multi in [
        ("triton", False, False),
        ("flash_attn", False, False),
        ("flash_attn", True, False),
        ("flash_attn", True, True),
    ]:
        engine = LLMEngine(
            model,
            cache_config=CacheConfig(num_blocks=48, block_size=16, layout=layout),
            scheduler_config=SchedulerConfig(
                max_num_seqs=3, max_num_batched_tokens=8, prefill_chunk_size=4
            ),
            execution_config=ExecutionConfig(
                async_scheduling=asynchronous,
                multi_stream=multi,
                static_buffers=static,
                pad_to_power_of_two=static,
            ),
            exit_config=ExitConfig("trace", depths_by_request=traces),
            attention_backend=backend,
        )
        for i in range(3):
            engine.add_request(
                str(i),
                [2, 3, 4, 5] * (i + 1),
                SamplingParams(max_tokens=6, min_loops=1, ignore_eos=True),
            )
        finished = {}
        for _ in range(300):
            if not engine.has_unfinished_requests():
                break
            for out in engine.step():
                if out.finished:
                    finished[out.request_id] = (out.token_ids, out.exit_depths)
        assert len(finished) == 3
        assert engine.cache_manager.num_used_blocks == 0
        outputs.append(finished)
    assert all(value == outputs[0] for value in outputs)
