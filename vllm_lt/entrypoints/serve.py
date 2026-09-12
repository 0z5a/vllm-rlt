"""Launch one resident Ouro model behind an OpenAI completions endpoint."""

import argparse
import json
import logging
from functools import partial

import torch

from vllm_lt.config import CacheConfig, SchedulerConfig
from vllm_lt.engine.llm_engine import LLMEngine
from vllm_lt.models.config import OURO_MODEL_ID, OURO_REVISION
from vllm_lt.models.ouro import OuroForCausalLM


def load_engine(args):
    from transformers import AutoTokenizer

    revision = args.revision or (OURO_REVISION if args.model == OURO_MODEL_ID else None)
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer or args.model,
        revision=args.tokenizer_revision or revision,
        trust_remote_code=False,
    )
    decoder = tokenizer.backend_tokenizer.decoder
    if decoder is None or json.loads(decoder.__getstate__()).get("type") != "ByteLevel":
        raise ValueError("serving requires the Ouro byte-level tokenizer")
    model = OuroForCausalLM.from_pretrained(
        args.model, revision=revision, device=args.device, dtype=getattr(torch, args.dtype)
    )
    engine = LLMEngine(
        model,
        cache_config=CacheConfig(num_blocks=args.num_blocks, block_size=args.block_size),
        scheduler_config=SchedulerConfig(
            max_num_seqs=args.max_num_seqs,
            max_num_batched_tokens=args.max_num_batched_tokens,
            mode=args.mode,
        ),
        attention_backend=args.attention_backend,
    )
    return engine, tokenizer


def main():
    from aiohttp import web

    from vllm_lt.serving.server import create_app

    parser = argparse.ArgumentParser(description="Serve one Ouro model with OpenAI completions")
    parser.add_argument("--model", default=OURO_MODEL_ID)
    parser.add_argument("--revision")
    parser.add_argument("--tokenizer")
    parser.add_argument("--tokenizer-revision")
    parser.add_argument("--served-model-name", default=OURO_MODEL_ID)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--attention-backend", choices=["torch", "triton"], default="triton")
    parser.add_argument("--mode", choices=["refill", "no_refill"], default="refill")
    parser.add_argument("--num-blocks", type=int, default=512)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--max-num-batched-tokens", type=int, default=128)
    parser.add_argument("--max-requests", type=int, default=64)
    parser.add_argument("--output-buffer", type=int, default=32)
    parser.add_argument("--max-body-bytes", type=int, default=1024 * 1024)
    parser.add_argument("--request-timeout", type=float, default=300)
    parser.add_argument("--write-timeout", type=float, default=10)
    parser.add_argument("--shutdown-timeout", type=float, default=30)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if args.device == "cpu" and args.attention_backend != "torch":
        parser.error("CPU execution requires --attention-backend torch")
    torch.set_num_threads(args.cpu_threads)
    logging.basicConfig(level=logging.INFO)
    app = create_app(
        partial(load_engine, args),
        model=args.served_model_name,
        max_requests=args.max_requests,
        output_buffer=args.output_buffer,
        max_body_bytes=args.max_body_bytes,
        request_timeout=args.request_timeout,
        write_timeout=args.write_timeout,
        shutdown_timeout=args.shutdown_timeout,
    )
    web.run_app(
        app,
        host=args.host,
        port=args.port,
        handler_cancellation=True,
        shutdown_timeout=args.shutdown_timeout,
    )


if __name__ == "__main__":
    main()
