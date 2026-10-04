import pytest
import torch

from tests.helpers import tiny_ouro_config
from vllm_rlt import CacheConfig, ExecutionConfig, ExitConfig, SamplingParams, SchedulerConfig
from vllm_rlt.engine.llm_engine import LLMEngine
from vllm_rlt.models import OuroForCausalLM
from vllm_rlt.request import Stage
from vllm_rlt.worker.model_runner import Submission


@pytest.mark.parametrize("hold_coda", [False, True])
def test_async_alias_trace_abort_and_request_reuse(monkeypatch, hold_coda):
    torch.manual_seed(13)
    model = OuroForCausalLM(tiny_ouro_config()).eval()
    trace = ExitConfig("trace", depths_by_request={"fixed": [4, 1, 3, 2, 4, 1]})
    engines = [
        LLMEngine(
            model,
            cache_config=CacheConfig(128, 2, alias_last_exited=alias),
            scheduler_config=SchedulerConfig(max_num_seqs=3, max_num_batched_tokens=4),
            exit_config=trace,
            execution_config=ExecutionConfig(async_scheduling=alias),
        )
        for alias in (False, True)
    ]
    if hold_coda:
        original_ready = Submission.ready
        monkeypatch.setattr(
            Submission,
            "ready",
            lambda ticket: False if ticket.batch.stage == Stage.CODA else original_ready(ticket),
        )
    params = SamplingParams(max_tokens=6, min_loops=1, ignore_eos=True)
    for round_index in range(3):
        for engine in engines:
            for rid in ("0", "1", "2"):
                engine.add_request(rid, [int(rid) + 2, round_index + 5], params, trace_id="fixed")
        if round_index == 1:
            for _ in range(4):
                engines[1].step()
            for engine in engines:
                engine.abort_request("1")
                engine.add_request("1", [7, 8, 9], params, trace_id="fixed")
        rounds = []
        for engine in engines:
            completed = {}
            for _ in range(300):
                if not engine.has_unfinished_requests():
                    break
                for output in engine.step():
                    if output.finished:
                        completed[output.request_id] = output
            assert len(completed) == 3 and engine.cache_manager.num_free_blocks == 128
            rounds.append(completed)
        assert rounds[0] == rounds[1]
