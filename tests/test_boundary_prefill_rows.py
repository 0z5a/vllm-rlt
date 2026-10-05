"""Boundary prefill keeps hidden rows, rotary positions and KV metadata aligned."""

import pytest
import torch

from tests.loopkv.test_parcae import tiny_config
from tests.test_huginn import tiny_huginn_config
from vllm_rlt import CacheConfig, ExecutionConfig, SchedulerConfig
from vllm_rlt.engine.llm_engine import LLMEngine
from vllm_rlt.models.huginn import HuginnForCausalLM
from vllm_rlt.models.parcae import ParcaeForCausalLM


@pytest.mark.parametrize("rows,requests", [(3, 1), (6, 3), (7, 1)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("family", ["huginn", "parcae"])
@torch.inference_mode()
def test_boundary_prefill_workspace_matches_eager(rows, requests, dtype, family):
    torch.manual_seed(61)
    model = (
        HuginnForCausalLM(tiny_huginn_config())
        if family == "huginn"
        else ParcaeForCausalLM(tiny_config())
    )
    model = model.to(dtype=dtype).eval()
    engines = [
        LLMEngine(
            model,
            cache_config=CacheConfig(num_blocks=128, block_size=2),
            scheduler_config=SchedulerConfig(max_num_seqs=3, max_num_batched_tokens=8),
            execution_config=ExecutionConfig(static_buffers=buffered, pad_to_power_of_two=buffered),
        )
        for buffered in (False, True)
    ]
    length = rows // requests
    ids = [str(i) for i in range(requests) for _ in range(length)]
    positions = list(range(length)) * requests
    tokens = list(range(3, 3 + rows))
    for _ in range(2):
        results = []
        for engine in engines:
            for i in range(requests):
                assert engine.cache_manager.allocate(str(i), length)
            torch.manual_seed(79)
            hidden = engine.model_runner._prefill_tokens(ids, positions, tokens)
            assert hidden.shape[0] == rows
            assert engine.model_runner.last_submitted_size == rows
            results.append(hidden.clone())
            for i in range(requests):
                engine.cache_manager.free(str(i))
            assert engine.cache_manager.num_used_blocks == 0
        torch.testing.assert_close(*results, atol=0, rtol=0)
