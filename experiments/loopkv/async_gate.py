"""Finite CUDA routing/lifetime gate for the native and rectangular alias paths."""

import argparse
import json
from pathlib import Path

import torch

from vllm_rlt import CacheConfig, ExecutionConfig, ExitConfig, SamplingParams, SchedulerConfig
from vllm_rlt.engine.llm_engine import LLMEngine
from vllm_rlt.models import OuroConfig, OuroForCausalLM


def routing_lifetime_gate(model, use_uva):
    engine = LLMEngine(
        model,
        attention_backend="triton",
        cache_config=CacheConfig(128, 2, alias_last_exited=True),
        scheduler_config=SchedulerConfig(max_num_seqs=1, max_num_batched_tokens=4),
        exit_config=ExitConfig("trace", depths_by_request={"frozen": [4, 1]}),
        execution_config=ExecutionConfig(async_scheduling=True),
    )
    state, cache = engine.model_runner.async_state, engine.cache_manager
    state.use_uva = use_uva
    params = SamplingParams(max_tokens=2, min_loops=1, ignore_eos=True)
    engine.add_request("warm", [2], params, trace_id="frozen")
    engine.scheduler._admit()
    warm = engine.scheduler.requests["warm"]
    bank, batch = state.prepare([warm], [0], [0], 1, recurrent=True)
    bank.transfer(batch)
    torch.cuda.synchronize()
    engine.abort_request("warm")

    # Poison a reused allocation, then delay its reset on a separate stream.
    cache.source_depths.fill_(77)
    torch.cuda.synchronize()
    engine.add_request("reset", [3], params, trace_id="frozen")
    producer, consumer = torch.cuda.Stream(), torch.cuda.Stream()
    with torch.cuda.stream(producer):
        torch.cuda._sleep(500_000_000)
        engine.scheduler._admit()
    request = engine.scheduler.requests["reset"]
    allocation = cache._get_allocation("reset")
    ready = cache._allocation_ready[id(allocation)]
    bank, batch = state.prepare([request], [0], [0], 1, recurrent=True)
    assert bank.dependencies == [ready] and not ready.query()
    with torch.cuda.stream(consumer):
        bank.transfer(batch)
        observed = cache.source_depths[allocation.block_tables[0][0]].clone()
        bank.record_done()
    bank.done.synchronize()
    assert torch.equal(observed.cpu(), torch.full_like(observed.cpu(), -1))

    # The bank owns all depth tables and query depths until its delayed reader ends.
    expected_tables = bank.depth_tables.clone()
    expected_depths = bank.depths.clone()
    torch.cuda.synchronize()
    state.index = state.banks.index(bank)
    with torch.cuda.stream(consumer):
        torch.cuda._sleep(500_000_000)
        observed_tables = bank.depth_tables.clone()
        observed_depths = bank.depths.clone()
        bank.record_done()
    assert not bank.done.query()
    reused, next_batch = state.prepare([request], [3], [1], 1, recurrent=True)
    assert reused is bank and bank.done.query()
    reused.transfer(next_batch)
    torch.cuda.synchronize()
    assert torch.equal(observed_tables, expected_tables)
    assert torch.equal(observed_depths, expected_depths)
    assert bank.depths[0].item() == 3
    engine.abort_request("reset")
    assert cache.num_free_blocks == 128 and not state.owners
    engine.close()
    return {"use_uva": use_uva, "allocation_reset_ordered": True, "bank_reader_retired": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(19)
    model = (
        OuroForCausalLM(
            OuroConfig(
                vocab_size=64,
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=64,
                max_position_embeddings=128,
                total_ut_steps=4,
            )
        )
        .to(device="cuda", dtype=torch.bfloat16)
        .eval()
    )
    lifetime = [routing_lifetime_gate(model, use_uva) for use_uva in (False, True)]
    print(json.dumps({"lifetime": lifetime}), flush=True)
    cases = []
    for batch in (1, 4, 16, 32, 64, 128):
        for use_uva in (False, True):
            baseline = None
            for mode in ("sync", "native_async", "alias_async", "native_graph", "alias_graph"):
                engine = LLMEngine(
                    model,
                    attention_backend="triton",
                    cache_config=CacheConfig(
                        batch * 80, 2, alias_last_exited=mode in ("alias_async", "alias_graph")
                    ),
                    scheduler_config=SchedulerConfig(
                        max_num_seqs=batch, max_num_batched_tokens=max(4, batch)
                    ),
                    exit_config=ExitConfig(
                        "trace", depths_by_request={"frozen": [4, 1, 3, 2, 4, 1]}
                    ),
                    execution_config=ExecutionConfig(
                        async_scheduling=mode != "sync",
                        cuda_graphs=mode in ("native_graph", "alias_graph"),
                        static_buffers=mode in ("native_graph", "alias_graph"),
                        pad_to_power_of_two=mode in ("native_graph", "alias_graph"),
                    ),
                )
                if engine.model_runner.async_state is not None:
                    engine.model_runner.async_state.use_uva = use_uva
                rounds = []
                for round_index in range(2):
                    for i in range(2 * batch):
                        engine.add_request(
                            str(i),
                            [2 + i % 5, 9 + round_index] * 17,
                            SamplingParams(
                                max_tokens=6,
                                min_loops=1,
                                ignore_eos=True,
                            ),
                            trace_id="frozen",
                        )
                    if round_index:
                        engine.step()
                        engine.abort_request("0")
                        engine.add_request(
                            "0",
                            [7, 8, 9] * 11,
                            SamplingParams(
                                max_tokens=6,
                                min_loops=1,
                                ignore_eos=True,
                            ),
                            trace_id="frozen",
                        )
                    completed = {}
                    for _ in range(100_000):
                        if not engine.has_unfinished_requests():
                            break
                        for output in engine.step():
                            if output.finished:
                                completed[output.request_id] = (
                                    output.token_ids,
                                    output.exit_depths,
                                )
                    engine.model_runner.synchronize()
                    assert len(completed) == 2 * batch
                    assert engine.cache_manager.num_free_blocks == batch * 80
                    assert not engine.model_runner.state_slots
                    rounds.append(completed)
                if baseline is None:
                    baseline = rounds
                assert rounds == baseline, (batch, use_uva, mode)
                row = {
                    "batch": batch,
                    "requests_per_round": 2 * batch,
                    "use_uva": use_uva,
                    "mode": mode,
                    "exact_two_rounds": True,
                }
                graphs = engine.model_runner.graphs
                if graphs is not None:
                    assert graphs.captures > 0 and graphs.replays > graphs.captures
                    row["graphs"] = {
                        "captures": graphs.captures,
                        "replays": graphs.replays,
                        "fallbacks": graphs.fallbacks,
                    }
                cases.append(row)
                print(json.dumps(row), flush=True)
                engine.close()
    args.out.write_text(
        json.dumps(
            {
                "scope": "tiny_cuda_routing_lifetime_not_official_or_performance",
                "torch": torch.__version__,
                "gpu": torch.cuda.get_device_name(),
                "lifetime": lifetime,
                "cases": cases,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
