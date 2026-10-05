import pytest
import torch

from experiments.loopkv.capture import drive
from tests.helpers import tiny_ouro_config
from vllm_rlt import CacheConfig, ExecutionConfig, ExitConfig, SamplingParams, SchedulerConfig
from vllm_rlt.engine.llm_engine import LLMEngine
from vllm_rlt.models import OuroForCausalLM
from vllm_rlt.request import Stage
from vllm_rlt.worker.model_runner import ModelRunner, Submission


@pytest.mark.parametrize("alias", [False, True])
@pytest.mark.parametrize("hold_coda", [False, True])
def test_text_stop_cancels_lookahead_and_keeps_reused_ids_clean(monkeypatch, alias, hold_coda):
    torch.manual_seed(23)
    engine = LLMEngine(
        OuroForCausalLM(tiny_ouro_config()).eval(),
        cache_config=CacheConfig(128, 2, alias_last_exited=alias),
        scheduler_config=SchedulerConfig(max_num_seqs=2, max_num_batched_tokens=4),
        exit_config=ExitConfig("ouro_delayed"),
        execution_config=ExecutionConfig(async_scheduling=True),
    )
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(
        ModelRunner,
        "_sample_tensor",
        lambda self, logits, request: torch.tensor(10 + request.num_scheduled_outputs),
    )
    original = Submission.ready
    monkeypatch.setattr(
        Submission,
        "ready",
        lambda self: False if hold_coda and self.batch.stage == Stage.CODA else original(self),
    )
    pieces = [b"x"] * 64
    pieces[10:14] = [b"a", b"Q", b":", b"later"]
    stops = {"token_bytes_hex": [piece.hex() for piece in pieces], "stops": ["Q:"]}
    params = SamplingParams(max_tokens=8, exit_threshold=0.5, ignore_eos=True)
    for _ in range(2):
        completed, _, _ = drive(
            engine, [[2, 3], [4] * 5, [6]], params, trace=False, stop_spec=stops
        )
        assert len(completed) == 3
        for output in completed.values():
            assert output["token_ids"] == [10, 11, 12]
            assert output["text_stop"] == {"text": "Q:", "byte_offset": 1}
            assert len(output["exit_depths"]) == 3
        assert engine.cache_manager.num_free_blocks == 128
        assert not engine.scheduler.requests
        assert not engine.model_runner.state_slots
