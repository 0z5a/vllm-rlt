"""Two spawned CUDA owners: NIXL KV, policy publication, and local-engine parity."""

import argparse
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import torch

from vllm_rlt import LLM, CacheConfig, ExecutionConfig, ExitConfig, SamplingParams, SchedulerConfig
from vllm_rlt.models import OuroConfig, OuroForCausalLM
from vllm_rlt.pd.config import PDConfig
from vllm_rlt.pd.engine import PDEngine


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--cuda-graphs", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--max-num-seqs", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(42)
    dtype = torch.bfloat16 if args.model else torch.float32
    source = args.model or OuroConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=128,
        total_ut_steps=4,
        bos_token_id=0,
        eos_token_id=0,
    )
    native = (
        OuroForCausalLM.from_pretrained(source, device="cuda:0", dtype=dtype)
        if args.model
        else OuroForCausalLM(source).to(device="cuda:0", dtype=dtype)
    )
    execution = ExecutionConfig(cuda_graphs=args.cuda_graphs, async_scheduling=True)
    cache = CacheConfig(num_blocks=512, enable_prefix_caching=True)
    scheduler = SchedulerConfig(
        max_num_seqs=args.max_num_seqs, max_num_batched_tokens=64, prefill_chunk_size=16
    )
    common = dict(
        execution_config=execution,
        exit_config=ExitConfig("ouro_delayed"),
        attention_backend="triton",
    )
    local = LLM(native, cache_config=cache, scheduler_config=scheduler, **common)
    pd = PDEngine(
        source,
        dtype="bfloat16" if args.model else "float32",
        pd_config=PDConfig(
            max_control_messages=1, startup_timeout=600, request_timeout=600, shutdown_timeout=120
        ),
        prefill_cache_config=cache,
        decode_cache_config=cache,
        prefill_scheduler_config=scheduler,
        decode_scheduler_config=scheduler,
        terminate_workers_on_failure=False,
        **common,
    )
    disaggregated = LLM(pd)
    prompts = [[3, 4, 5] * 9 + [6], [3, 4, 5] * 9 + [7]]
    report = dict(model=asdict(native.config), cuda_graphs=args.cuda_graphs, iterations=[])
    report["max_num_seqs"] = args.max_num_seqs
    try:
        for version, loops in enumerate((2, 3, 4), 1):
            with torch.no_grad():
                native.lm_head.weight.mul_(0.98)
            for arm in (local, disaggregated):
                arm.start_weight_update(version)
                weights = list(native.named_parameters())
                for offset in range(0, len(weights), 8):
                    arm.update_weights(weights[offset : offset + 8])
                arm.finish_weight_update()
                assert arm.get_weight_version() == version
            for mode in ("raw", "processed"):
                params = [
                    SamplingParams(
                        max_tokens=6,
                        max_loops=loops,
                        seed=11 + i,
                        temperature=args.temperature,
                        top_k=8,
                        top_p=0.8,
                        ignore_eos=True,
                        logprobs=0,
                        logprobs_mode=mode,
                    )
                    for i in range(len(prompts))
                ]
                outputs, timings = {}, {}
                for label, arm in (("local", local), ("pd", disaggregated)):
                    started = time.monotonic()
                    outputs[label] = arm.generate(prompts, params)
                    timings[label] = time.monotonic() - started
                errors = []
                for expected, result in zip(outputs["local"], outputs["pd"], strict=True):
                    assert expected.token_ids == result.token_ids
                    assert expected.exit_depths == result.exit_depths
                    assert result.weight_version == version
                    assert result.sampling_params == expected.sampling_params
                    delta = (
                        torch.tensor(expected.log_probs) - torch.tensor(result.log_probs)
                    ).abs()
                    errors.append(delta.max().item())
                    torch.testing.assert_close(
                        torch.tensor(expected.log_probs),
                        torch.tensor(result.log_probs),
                        atol=2e-5,
                        rtol=2e-5,
                    )
                row = dict(
                    version=version,
                    loops=loops,
                    mode=mode,
                    seconds=timings,
                    max_logprob_error=max(errors),
                    outputs=[asdict(o) for o in outputs["pd"]],
                )
                report["iterations"].append(row)
                print("PARITY", version, loops, mode, timings, max(errors), flush=True)
        # A failed chunk cannot expose a mixed policy; a newer full epoch recovers.
        disaggregated.start_weight_update(4)
        try:
            disaggregated.update_weights([("missing_parameter", torch.zeros(1))])
        except RuntimeError as error:
            report["rejected_chunk"] = str(error)
        else:
            raise AssertionError("Invalid publication unexpectedly succeeded")
        disaggregated.start_weight_update(5)
        disaggregated.update_weights(native.named_parameters())
        disaggregated.finish_weight_update()
        recovered = disaggregated.generate(prompts[:1], params[:1])[0]
        assert recovered.weight_version == 5
        stopped = disaggregated.generate(
            prompts[:1], [replace(params[0], stop_token_ids=(recovered.token_ids[0],))]
        )[0]
        assert len(stopped.token_ids) == len(stopped.log_probs) == 1
        assert stopped.finish_reason == "stop"
        pd.add_request("abort", prompts[0], params[0])
        while not pd.transfers[next(iter(pd.transfers))].latest_output:
            pd.step()
        aborted = pd.abort_request("abort")
        assert aborted.finish_reason == "abort"
        assert len(aborted.log_probs) == len(aborted.token_ids)
        while pd.has_unfinished_requests():
            pd.step()
        disaggregated.reset_prefix_cache()
        report.update(recovered_version=5, stop=asdict(stopped), abort=asdict(aborted))
    finally:
        disaggregated.close()
        local.close()
    report["workers"] = pd.worker_metrics
    assert all(
        m["used_blocks"] == 0 and m["weight_version"] == 5 for m in pd.worker_metrics.values()
    )
    assert sum(m["bytes_sent"] for m in pd.worker_metrics.values()) > 0
    args.output.write_text(json.dumps(report, indent=2))
    print("PD_CONTRACT_PASSED", args.output, flush=True)


if __name__ == "__main__":
    main()
