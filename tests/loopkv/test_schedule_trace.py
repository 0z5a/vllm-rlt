from dataclasses import asdict

import pytest
import torch

from experiments.loopkv.schedule_trace import Capture, Replay
from tests.helpers import tiny_ouro_config
from vllm_rlt import CacheConfig, ExecutionConfig, ExitConfig, SamplingParams, SchedulerConfig
from vllm_rlt.engine.llm_engine import LLMEngine
from vllm_rlt.models import OuroForCausalLM
from vllm_rlt.worker.model_runner import Submission


def drain(engine):
    completed = {}
    for _ in range(500):
        if not engine.has_unfinished_requests():
            return completed
        for output in engine.step():
            if output.finished:
                completed[output.request_id] = asdict(output)
    raise AssertionError("trace did not drain")


@pytest.mark.parametrize("held_capture", [False, True])
@pytest.mark.parametrize("alias", [False, True])
@pytest.mark.parametrize("policy", [(0.5, 2, 4), (1.0, 4, 4), (0.0, 1, 1)])
def test_replay_preserves_batches_despite_opposite_readback_readiness(
    monkeypatch, held_capture, alias, policy
):
    torch.manual_seed(71)
    model = OuroForCausalLM(tiny_ouro_config()).eval()

    def make(alias):
        return LLMEngine(
            model,
            cache_config=CacheConfig(128, 2, alias_last_exited=alias),
            scheduler_config=SchedulerConfig(max_num_seqs=3, max_num_batched_tokens=4),
            exit_config=ExitConfig("ouro_delayed"),
            execution_config=ExecutionConfig(async_scheduling=True),
        )

    def submit(engine):
        threshold, minimum, maximum = policy
        for i in range(7):
            engine.add_request(
                str(i),
                [2 + i] * (3 + i % 3),
                SamplingParams(
                    max_tokens=6,
                    min_loops=minimum,
                    max_loops=maximum,
                    exit_threshold=threshold,
                    ignore_eos=True,
                ),
            )

    monkeypatch.setattr(Submission, "ready", lambda self: not held_capture)
    native = make(False)
    capture = Capture(native)
    submit(native)
    expected = drain(native)
    # The replay must ignore readiness changes, including delayed EOS/coda delivery.
    monkeypatch.setattr(Submission, "ready", lambda self: held_capture)
    candidate = make(alias)
    replay = Replay(
        candidate, capture.events, {rid: item["exit_depths"] for rid, item in expected.items()}
    )
    submit(candidate)
    assert drain(candidate) == expected
    replay.assert_drained()
    assert candidate.model_runner.exit_config.mode == "ouro_delayed"
    assert native.cache_manager.num_free_blocks == 128
