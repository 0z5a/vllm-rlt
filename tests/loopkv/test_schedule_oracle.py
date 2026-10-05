import pytest
import torch

from experiments.loopkv.schedule_oracle import compare
from tests.helpers import tiny_ouro_config
from vllm_rlt import CacheConfig, ExecutionConfig, ExitConfig, SamplingParams, SchedulerConfig
from vllm_rlt.engine.llm_engine import LLMEngine
from vllm_rlt.models import OuroForCausalLM
from vllm_rlt.request import Stage
from vllm_rlt.worker.model_runner import ModelRunner


@pytest.mark.parametrize("threshold", [0.0, 0.5, 1.0])
def test_serialized_resident_oracle_matches_engine_policy(threshold):
    torch.manual_seed(123)
    model = OuroForCausalLM(tiny_ouro_config()).eval()
    engines = [
        LLMEngine(
            model,
            cache_config=CacheConfig(128, 2, alias_last_exited=index == 1),
            scheduler_config=SchedulerConfig(max_num_seqs=3, max_num_batched_tokens=4),
            exit_config=ExitConfig("ouro_delayed"),
            execution_config=ExecutionConfig(async_scheduling=index < 2),
        )
        for index in range(3)
    ]
    params = SamplingParams(
        max_tokens=6, min_loops=2, max_loops=4, exit_threshold=threshold, ignore_eos=True
    )
    for engine in engines:
        for i in range(6):
            engine.add_request(str(i), [2 + i, 4, 5], params)
    report = compare(engines[:2], 300)
    assert report["complete"] and report["exact_request_objects"]
    baseline = {}
    while engines[2].has_unfinished_requests():
        for output in engines[2].step():
            if output.finished:
                baseline[output.request_id] = (output.token_ids, output.exit_depths)
    assert baseline == {
        key: (value["token_ids"], value["exit_depths"])
        for key, value in report["completed"][0].items()
    }
    assert all(e.cache_manager.num_free_blocks == 128 for e in engines)
    assert all(row["result_exact"] for row in report["rows"])


def test_oracle_retains_gate_divergence_and_rejects_unmatched_schedule(monkeypatch):
    torch.manual_seed(123)
    model = OuroForCausalLM(tiny_ouro_config()).eval()
    engines = [
        LLMEngine(
            model,
            cache_config=CacheConfig(128, 2, alias_last_exited=alias),
            exit_config=ExitConfig("ouro_delayed"),
            execution_config=ExecutionConfig(async_scheduling=True),
        )
        for alias in (False, True)
    ]
    original = ModelRunner.submit

    def disagree(runner, batch):
        ticket = original(runner, batch)
        if batch.stage == Stage.RECURRENT:
            ticket.cached = [float(runner is engines[0].model_runner)] * len(batch.items)
        return ticket

    monkeypatch.setattr(ModelRunner, "submit", disagree)
    for engine in engines:
        engine.add_request(
            "0", [2, 3], SamplingParams(max_tokens=3, exit_threshold=0.5, ignore_eos=True)
        )
    report = compare(engines, 30)
    assert not report["complete"]
    assert report["first_schedule_difference"][0] != report["first_schedule_difference"][1]
    divergent = [row for row in report["rows"] if not row["result_exact"]]
    assert divergent and divergent[0]["result_max_abs_error"] == 1.0
    assert divergent[0]["native_results"] == [1.0]
    assert divergent[0]["alias_results"] == [0.0]
