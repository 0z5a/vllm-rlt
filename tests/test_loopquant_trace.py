import io
import json

import torch

from loopquant.bench import run_cohort
from loopquant.trace import ActivationTrace, TracedOuroForCausalLM
from tests.helpers import tiny_ouro_config
from vllm_rlt import LLM, CacheConfig, SchedulerConfig
from vllm_rlt.core.kv_cache_manager import KVCacheManager
from vllm_rlt.models.ouro import OuroForCausalLM


def test_trace_reordered_actual_depths_and_padding():
    stream = io.StringIO()
    trace = ActivationTrace(stream, {"a": 2, "b": 1})
    config = tiny_ouro_config()
    model = TracedOuroForCausalLM(config, trace)
    cache = KVCacheManager(
        config.num_hidden_layers,
        config.num_key_value_heads,
        config.head_dim,
        num_blocks=16,
        block_size=2,
        max_loops=4,
    )
    assert cache.allocate("a", 2) and cache.allocate("b", 2)
    batch = cache._prepare_batch(["b", "a"], [3, 0], [0, 0])
    trace.prepare(batch, 3)
    module = model.model.layers[0].self_attn.q_proj
    values = torch.ones(3, config.hidden_size)
    values[-1] = float("nan")
    trace.record("model.layers.0.self_attn.q_proj", "input", module, values)
    row = json.loads(stream.getvalue())
    assert row["loop_id"] == [3, 0, -1]
    assert row["request_id"] == ["b", "a", None]
    assert row["valid_mask"] == [True, True, False]
    assert all(stats.rows == 1 and stats.amax == 1 for stats in trace.statistics.values())
    assert row["weight_storage_id"] == module.weight.untyped_storage().data_ptr()


def test_real_closed_loop_trace_preserves_outputs_and_token_accounting():
    config = tiny_ouro_config()
    prompts = [[5, 7], [3, 9, 6], [2]]
    stream = io.StringIO()
    trace = ActivationTrace(stream, {str(i): len(tokens) for i, tokens in enumerate(prompts)})
    traced = TracedOuroForCausalLM(config, trace)
    plain = OuroForCausalLM(config)
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
        assert result.failed_requests == 0 and result.kv_used_after_drain == 0
        outputs.append([record.token_ids for record in records])
    assert outputs[0] == outputs[1]
    name = "model.layers.0.self_attn.q_proj"
    for loop in range(4):
        assert trace.statistics[name, "input", loop, "prefill"].rows == 6
        assert trace.statistics[name, "input", loop, "decode"].rows == 6
    rows = [json.loads(line) for line in stream.getvalue().splitlines()]
    selected = [row for row in rows if row["physical_module_id"] == name]
    assert len({row["weight_storage_id"] for row in selected}) == 1
    assert not trace.rows
