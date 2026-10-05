import pytest

from experiments.loopkv.metrics import WorkCounters
from tests.helpers import tiny_ouro_config
from vllm_rlt import CacheConfig, ExecutionConfig, ExitConfig, SamplingParams, SchedulerConfig
from vllm_rlt.engine.llm_engine import LLMEngine
from vllm_rlt.models import OuroForCausalLM


@pytest.mark.parametrize("asynchronous", [False, True])
def test_submission_counters_include_every_executed_decode_depth(asynchronous):
    engine = LLMEngine(
        OuroForCausalLM(tiny_ouro_config()).eval(),
        cache_config=CacheConfig(128, 2, alias_last_exited=asynchronous),
        scheduler_config=SchedulerConfig(max_num_seqs=2, max_num_batched_tokens=4),
        exit_config=ExitConfig("trace", depths_by_request={"fixed": [4, 1, 2, 4]}),
        execution_config=ExecutionConfig(async_scheduling=asynchronous),
    )
    for i, prompt in enumerate(([2, 3], [4, 5, 6], [7])):
        engine.add_request(
            str(i),
            prompt,
            SamplingParams(max_tokens=4, min_loops=1, ignore_eos=True),
            trace_id="fixed",
        )
    counters = WorkCounters()
    for _ in range(200):
        if not engine.has_unfinished_requests():
            break
        engine.step()
        counters.observe(engine.last_schedule, len(engine.cache_manager._allocations))
    assert not engine.has_unfinished_requests()
    result = counters.summary()
    assert result["submitted_recurrent_rows"] == 21
    assert result["submitted_recurrent_depth_rows"] == {1: 9, 2: 6, 3: 3, 4: 3}
    assert result["submitted_prefill_tokens"] == 6
    assert result["observed_peak_residents_after_step"] == 2
    assert result["recurrent_calls"] == sum(result["recurrent_batch_histogram"].values())
    assert 1 <= result["effective_batch_mean"] <= 2


def test_effective_batch_percentiles_weight_calls_not_rows():
    counters = WorkCounters()
    counters.recurrent_rows.update({1: 7, 16: 2, 128: 1})
    result = counters.summary()
    assert result["effective_batch_nearest_rank_quantiles"] == {"p10": 1, "p50": 1, "p90": 16}
    assert result["effective_batch_mean"] == 16.7
