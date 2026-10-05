import asyncio

import pytest
import torch

from experiments.loopkv.http_load import run_load
from tests.helpers import tiny_ouro_config
from vllm_rlt import CacheConfig, ExecutionConfig, ExitConfig, SamplingParams, SchedulerConfig
from vllm_rlt.engine.llm_engine import LLMEngine
from vllm_rlt.models import OuroForCausalLM


@pytest.mark.parametrize("concurrency", [None, 2])
@pytest.mark.parametrize("compact", [False, True])
def test_http_stream_matches_direct_engine_and_drains(concurrency, compact):
    torch.manual_seed(71)
    model = OuroForCausalLM(tiny_ouro_config()).eval()

    def make():
        return LLMEngine(
            model,
            cache_config=CacheConfig(128, 2, compact_last_exited=compact),
            scheduler_config=SchedulerConfig(max_num_seqs=3, max_num_batched_tokens=8),
            exit_config=ExitConfig("ouro_delayed"),
            execution_config=ExecutionConfig(async_scheduling=True),
        )

    params = SamplingParams(max_tokens=6, min_loops=4, max_loops=4, ignore_eos=True)
    workload = [
        {
            "request_id": str(i),
            "prompt_token_ids": [2 + i] * (i + 2),
            "max_tokens": 3 + i,
            "seed": i,
            "arrival_s": i * 0.002,
        }
        for i in range(5)
    ]
    engine = make()
    for _ in range(2):
        result = asyncio.run(run_load(engine, workload, params, concurrency=concurrency))
        assert result["seconds"] > 0
        assert result["peak_client_concurrency"] <= (concurrency or len(workload))
        expected = make()
        for row in workload:
            expected.add_request(
                row["request_id"],
                row["prompt_token_ids"],
                SamplingParams(
                    max_tokens=row["max_tokens"],
                    min_loops=4,
                    max_loops=4,
                    seed=row["seed"],
                    ignore_eos=True,
                ),
            )
        direct = {}
        while expected.has_unfinished_requests():
            for out in expected.step():
                if out.finished:
                    direct[out.request_id] = out
        for row in result["requests"]:
            output = direct[row["request_id"]]
            assert row["token_ids"] == output.token_ids
            assert row["exit_depths"] == output.exit_depths
            assert len(row["token_times_s"]) == len(row["token_ids"])
            assert row["sent_s"] <= row["token_times_s"][0] <= row["finished_s"]
            assert row["token_times_s"] == sorted(row["token_times_s"])
        expected.close()
    engine.close()


def test_http_goodput_uses_complete_request_slo_and_full_drain_denominator():
    from experiments.loopkv.http_bench import summarize

    result = {
        "seconds": 2.0,
        "arrival_mode": "closed_loop",
        "requests": [
            {
                "sent_s": 0.0,
                "planned_s": 0.0,
                "finished_s": 0.5,
                "token_times_s": [0.2, 0.4],
                "token_ids": [1, 2],
            },
            {
                "sent_s": 0.5,
                "planned_s": 0.4,
                "finished_s": 1.1,
                "token_times_s": [0.9, 0.9, 1.0],
                "token_ids": [3, 4, 5],
            },
        ],
    }
    row = summarize(result, {"ttft_s": 0.3, "request_s": 0.7, "request_p99_itl_s": 0.25})
    assert row["request_goodput_s"] == 0.5 and row["token_goodput_s"] == 1.0
    assert row["complete_requests"] == 2 and row["output_tokens"] == 5
    assert row["latency_seconds"]["token_itl"]["0.99"] == 0.2
    assert summarize(result, None)["request_goodput_s"] is None
    result["arrival_mode"] = "open_loop"
    result["requests"][0]["planned_s"] = -0.2
    assert (
        summarize(result, {"ttft_s": 0.3, "request_s": 0.7, "request_p99_itl_s": 0.25})[
            "request_goodput_s"
        ]
        == 0.0
    )
