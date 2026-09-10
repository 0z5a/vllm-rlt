"""Bounded real-weight correctness smoke; run only inside a GPU reservation.

This is one validation pass, not a latency or model-quality benchmark. Downloads
must be completed separately. The dense oracle uses official eager BF16 attention
arithmetic; paged attention accumulates in float32, so tolerances are explicit.
"""

import argparse
import gc
import json
import os
from pathlib import Path

import torch

from vllm_lt.config import CacheConfig, SchedulerConfig
from vllm_lt.core.kv_cache_manager import KVCacheManager
from vllm_lt.engine.llm_engine import LLMEngine
from vllm_lt.models import OURO_MODEL_ID, OURO_REVISION, OuroForCausalLM
from vllm_lt.models.reference import dense_reference
from vllm_lt.sampling_params import SamplingParams


def comparison(actual, expected, *, atol, rtol):
    actual, expected = actual.float(), expected.float()
    difference = actual - expected
    return {
        "max_abs_error": difference.abs().max().item(),
        "rms_error": difference.square().mean().sqrt().item(),
        "reference_rms": expected.square().mean().sqrt().item(),
        "actual_top1": actual.argmax(dim=-1).tolist(),
        "reference_top1": expected.argmax(dim=-1).tolist(),
        "top1_equal": torch.equal(actual.argmax(dim=-1), expected.argmax(dim=-1)),
        "finite": bool(actual.isfinite().all() and expected.isfinite().all()),
        "allclose": torch.allclose(actual, expected, atol=atol, rtol=rtol),
    }


@torch.inference_mode()
def packed_prefill(model, prompts, backend):
    config = model.config
    parameter = next(model.parameters())
    cache = KVCacheManager(
        num_layers=config.num_hidden_layers,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        num_blocks=32,
        block_size=8,
        max_loops=config.total_ut_steps,
        device=parameter.device,
        dtype=parameter.dtype,
        backend=backend,
    )
    tokens, request_ids, positions = [], [], []
    try:
        for index, prompt in enumerate(prompts):
            request_id = str(index)
            assert cache.allocate(request_id, len(prompt))
            tokens.extend(prompt)
            request_ids.extend([request_id] * len(prompt))
            positions.extend(range(len(prompt)))
        hidden = model.prelude(torch.tensor(tokens, device=parameter.device, dtype=torch.long))
        logits = []
        for depth in range(config.total_ut_steps):
            hidden, _ = model.recurrent(
                hidden, request_ids, [depth] * len(tokens), positions, cache
            )
            logits.append(model.coda(hidden))
        return logits
    finally:
        for index in range(len(prompts)):
            cache.free(str(index))
        assert cache.num_used_blocks == 0, "prefill KV pages leaked"


def generate(model, tokenizer, prompts, params, *, backend, mode, serial=False):
    engine = LLMEngine(
        model,
        cache_config=CacheConfig(num_blocks=48, block_size=8),
        scheduler_config=SchedulerConfig(max_num_seqs=3, max_num_batched_tokens=6, mode=mode),
        attention_backend=backend,
    )
    outputs, schedule = {}, []

    def drain():
        for _ in range(1000):
            if not engine.has_unfinished_requests():
                return
            emitted = engine.step()
            batch = engine.last_schedule
            schedule.append(
                {
                    "stage": batch.stage.value,
                    "request_ids": [item.request.request_id for item in batch.items],
                    "positions_after_step": [item.request.position for item in batch.items],
                    "loops_done_after_step": [item.request.loops_done for item in batch.items],
                }
            )
            for output in emitted:
                if output.finished:
                    outputs[output.request_id] = {
                        "token_ids": output.token_ids,
                        "exit_depths": output.exit_depths,
                        "text": tokenizer.decode(output.token_ids, skip_special_tokens=True),
                        "finish_reason": output.finish_reason,
                    }
        raise RuntimeError("generation exceeded the 1000-step stop condition")

    try:
        for index, (prompt, param) in enumerate(zip(prompts, params)):
            engine.add_request(str(index), prompt, param)
            if serial:
                drain()
        if not serial:
            drain()
        assert len(outputs) == len(prompts), "generation did not return every request"
        assert engine.cache_manager.num_used_blocks == 0, "generation KV pages leaked"
        return {
            "outputs": [outputs[str(index)] for index in range(len(prompts))],
            "schedule": schedule,
            "used_blocks_after_completion": engine.cache_manager.num_used_blocks,
        }
    finally:
        for request_id in list(engine.scheduler.requests):
            engine.abort_request(request_id)
        assert engine.cache_manager.num_used_blocks == 0, "generation cleanup leaked KV pages"


def validate(args):
    from transformers import AutoTokenizer

    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("This real-checkpoint smoke requires a reserved CUDA/ROCm device")
    dtype = getattr(torch, args.dtype)
    # Keep arithmetic settings identical across the oracle and both backends.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.cuda.reset_peak_memory_stats(device)
    model = OuroForCausalLM.from_pretrained(args.model_path, device=device, dtype=dtype)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, local_files_only=True, trust_remote_code=False
    )
    prompt_texts = ["The capital of France is", "2 + 2 =", "The opposite of hot is"]
    prompts = [tokenizer.encode(text, add_special_tokens=False) for text in prompt_texts]
    if any(len(prompt) > 16 for prompt in prompts):
        raise ValueError("validation prompt exceeds the fixed 16-token smoke budget")
    report = {
        "validation_kind": "one correctness smoke, no speed or model-quality claim",
        "expected_model_id": OURO_MODEL_ID,
        "expected_revision": OURO_REVISION,
        "model_path": str(Path(args.model_path).resolve()),
        "device": str(next(model.parameters()).device),
        "device_name": torch.cuda.get_device_name(device),
        "visible_device_count": torch.cuda.device_count(),
        "visibility_environment": {
            name: os.environ.get(name)
            for name in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES")
        },
        "torch_version": torch.__version__,
        "dtype": args.dtype,
        "atol": args.atol,
        "rtol": args.rtol,
        "tf32": False,
        "bf16_reduced_precision_reduction": False,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "prompt_texts": prompt_texts,
        "prompt_token_ids": prompts,
        "max_tokens": [4, 8, 4],
        "exit_thresholds": [0.0, 0.7, 1.0],
        "min_loops": 2,
        "prefill_comparisons": [],
        "generation": {},
        "failures": [],
    }
    print(
        json.dumps(
            {
                "event": "loaded",
                "controls": {
                    key: value
                    for key, value in report.items()
                    if key not in ("prefill_comparisons", "generation", "failures")
                },
            }
        ),
        flush=True,
    )
    dense = [
        dense_reference(model, torch.tensor(prompt, device=device), model.config.total_ut_steps)
        for prompt in prompts
    ]
    packed_torch = packed_prefill(model, prompts, "torch")
    packed_triton = packed_prefill(model, prompts, "triton")
    for depth in range(model.config.total_ut_steps):
        dense_logits = torch.cat([reference[depth][2] for reference in dense])
        values = {
            "depth": depth + 1,
            "torch_vs_dense": comparison(
                packed_torch[depth], dense_logits, atol=args.atol, rtol=args.rtol
            ),
            "triton_vs_dense": comparison(
                packed_triton[depth], dense_logits, atol=args.atol, rtol=args.rtol
            ),
            "triton_vs_torch": comparison(
                packed_triton[depth], packed_torch[depth], atol=args.atol, rtol=args.rtol
            ),
        }
        report["prefill_comparisons"].append(values)
        for name in ("torch_vs_dense", "triton_vs_dense", "triton_vs_torch"):
            value = values[name]
            if not value["finite"] or not value["allclose"]:
                report["failures"].append(f"depth {depth + 1}: {name} exceeds declared tolerance")
            if depth == model.config.total_ut_steps - 1 and not value["top1_equal"]:
                report["failures"].append(f"final depth: {name} top-1 token IDs differ")
        print(json.dumps({"event": "prefill_comparison", **values}), flush=True)
    del dense, packed_torch, packed_triton, dense_logits
    params = [
        SamplingParams(max_tokens=count, exit_threshold=threshold, ignore_eos=True)
        for count, threshold in zip(report["max_tokens"], report["exit_thresholds"])
    ]
    for backend in ("torch", "triton"):
        for mode in ("serial", "refill", "no_refill"):
            name = f"{backend}_{mode}"
            result = generate(
                model,
                tokenizer,
                prompts,
                params,
                backend=backend,
                mode="refill" if mode == "serial" else mode,
                serial=mode == "serial",
            )
            report["generation"][name] = result
            print(json.dumps({"event": "generation", "configuration": name, **result}), flush=True)
    baseline = report["generation"]["torch_serial"]["outputs"]
    for name, result in report["generation"].items():
        same_tokens = [row["token_ids"] for row in result["outputs"]] == [
            row["token_ids"] for row in baseline
        ]
        same_exits = [row["exit_depths"] for row in result["outputs"]] == [
            row["exit_depths"] for row in baseline
        ]
        result["token_ids_match_torch_serial"] = same_tokens
        result["exit_depths_match_torch_serial"] = same_exits
        if not same_tokens or not same_exits:
            report["failures"].append(f"{name} token IDs or exit depths differ from torch_serial")
    report["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
    report["peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
    report["passed"] = not report["failures"]
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, help="Local pinned checkpoint and tokenizer")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--atol", type=float, default=0.25)
    parser.add_argument("--rtol", type=float, default=0.02)
    parser.add_argument("--output", type=Path, help="Optional complete JSON result artifact")
    args = parser.parse_args()
    if args.atol < 0 or args.rtol < 0:
        parser.error("tolerances must be nonnegative")
    report = validate(args)
    gc.collect()
    report["allocated_bytes_after_gc"] = torch.cuda.memory_allocated(args.device)
    report["reserved_bytes_after_gc"] = torch.cuda.memory_reserved(args.device)
    # Test-only cleanup of this process's persistent cuBLAS workspaces, matching
    # torch.testing._internal.common_utils.CudaMemoryLeakCheck. Python GC alone
    # leaves these framework-owned allocations live; they are not model/KV leaks.
    torch._C._cuda_clearCublasWorkspaces()
    torch.cuda.empty_cache()
    report["allocated_bytes_after_cleanup"] = torch.cuda.memory_allocated(args.device)
    report["reserved_bytes_after_cleanup"] = torch.cuda.memory_reserved(args.device)
    if report["allocated_bytes_after_cleanup"]:
        report["failures"].append("task-owned model tensors remain allocated after cleanup")
        report["passed"] = False
    serialized = json.dumps(report, indent=2)
    if args.output:
        args.output.write_text(serialized + "\n")
    print(serialized, flush=True)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
