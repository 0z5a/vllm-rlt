"""Launch one resident Ouro model behind an OpenAI completions endpoint."""

import argparse
import logging
import sys
from dataclasses import asdict
from functools import partial

import torch

from vllm_rlt.config import SchedulerConfig
from vllm_rlt.engine.llm_engine import LLMEngine
from vllm_rlt.entrypoints.runtime_args import (
    add_runtime_args,
    profile_config_from_args,
    runtime_configs,
)
from vllm_rlt.models import AutoModelForCausalLM, resolve_local_model_path
from vllm_rlt.recipes import OURO_MODEL_ID


def _is_ouro_model(model_name_or_path: str) -> bool:
    import json

    from vllm_rlt.models import resolve_local_model_path
    from vllm_rlt.recipes import get_recipe

    raw = str(model_name_or_path)
    recipe = get_recipe(raw)
    if recipe is not None:
        return recipe.served_model_name == "ouro"

    local_path = resolve_local_model_path(raw)
    if local_path is not None and (local_path / "config.json").is_file():
        try:
            cfg = json.loads((local_path / "config.json").read_text())
            return cfg.get("model_type", "").lower() == "ouro"
        except Exception:
            pass

    if "nanbeige" in raw.lower():
        return False

    return True


def load_engine(args):
    from tokenizers.decoders import ByteLevel
    from transformers import AutoTokenizer

    profiling = profile_config_from_args(args)
    if profiling is not None:
        profiling.resolve_activities(args.device)
    revision = args.revision
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer or args.model,
        revision=args.tokenizer_revision or revision,
        trust_remote_code=False,
    )
    if _is_ouro_model(args.model):
        decoder = getattr(getattr(tokenizer, "backend_tokenizer", None), "decoder", None)
        if not isinstance(decoder, ByteLevel):
            raise ValueError("serving requires the Ouro byte-level tokenizer")
    allow_download = getattr(args, "allow_download", getattr(args, "assume_yes", False))
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=revision,
        device=args.device,
        dtype=getattr(torch, args.dtype),
        allow_download=allow_download,
    )
    engine = LLMEngine(
        model,
        **runtime_configs(args),
        scheduler_config=SchedulerConfig(
            policy=getattr(args, "scheduling_policy", "fcfs"),
            enable_preemption=getattr(args, "enable_preemption", False),
            max_num_seqs=args.max_num_seqs,
            max_num_batched_tokens=args.max_num_batched_tokens,
            mode=args.mode,
            prefill_chunk_size=getattr(args, "prefill_chunk_size", 128),
            max_prefill_batches_before_decode=getattr(args, "max_prefill_batches_before_decode", 1),
            admission_scan_limit=getattr(args, "admission_scan_limit", 64),
            max_admission_bypasses=getattr(args, "max_admission_bypasses", 8),
            min_coda_batch_size=getattr(args, "min_coda_batch_size", 1),
        ),
        attention_backend=args.attention_backend,
    )
    if profiling is not None:
        engine.start_profile(profiling, scheduled=True)
    return engine, tokenizer


def main():
    from aiohttp import web

    from vllm_rlt.serving.protocol import ServingLimits
    from vllm_rlt.serving.server import create_app

    parser = argparse.ArgumentParser(description="Serve one Ouro model with OpenAI completions")
    parser.add_argument("--model", default=OURO_MODEL_ID)
    parser.add_argument("--revision")
    parser.add_argument("--tokenizer")
    parser.add_argument("--tokenizer-revision")
    parser.add_argument("--served-model-name", default=OURO_MODEL_ID)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    parser.add_argument(
        "--attention-backend",
        type=str.lower,
        choices=["torch", "triton", "flash_attn", "flash_attn_2", "flash_attn_3", "flash_attn_4"],
        default="triton",
    )
    parser.add_argument("--mode", choices=["refill", "no_refill"], default="refill")
    parser.add_argument("--num-blocks", type=int, help="Override automatic KV sizing")
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--max-num-batched-tokens", type=int, default=128)
    defaults = asdict(ServingLimits())
    for name, default in defaults.items():
        parser.add_argument("--" + name.replace("_", "-"), type=type(default), default=default)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        dest="assume_yes",
        help="Automatically approve model downloading from HuggingFace without interactive prompt",
    )
    add_runtime_args(parser)
    args = parser.parse_args()
    limits = ServingLimits(**{name: getattr(args, name) for name in defaults})
    if args.device == "cpu" and args.attention_backend != "torch":
        parser.error("CPU execution requires --attention-backend torch")
    local_path = resolve_local_model_path(args.model)
    args.allow_download = False
    if local_path is not None:
        args.model = str(local_path)
    else:
        if args.assume_yes:
            args.allow_download = True
        elif sys.stdin.isatty():
            prompt_msg = (
                f"Model '{args.model}' not found locally. Download from HuggingFace? (y/yes): "
            )
            response = input(prompt_msg).strip().lower()
            if response in ("y", "yes"):
                args.allow_download = True
            else:
                parser.error(f"Download of '{args.model}' was not approved.")
        else:
            parser.error(
                f"Model '{args.model}' is not available locally. In non-interactive "
                f"environments, pass -y/--yes to approve downloading from HuggingFace."
            )
    torch.set_num_threads(args.cpu_threads)
    logging.basicConfig(level=logging.INFO)
    app = create_app(
        partial(load_engine, args),
        model=args.served_model_name,
        limits=limits,
        allowed_hosts=("localhost", args.host),
    )
    web.run_app(
        app,
        host=args.host,
        port=args.port,
        handler_cancellation=True,
        shutdown_timeout=limits.shutdown_timeout,
    )


if __name__ == "__main__":
    main()
