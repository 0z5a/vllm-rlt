"""Warmed native engine capture; the finite controller owns GPU and IO locks."""

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import time

import torch

from loopquant.bench import TraceRequest, run_trace
from loopquant.checkpoint import load_weights
from loopquant.quantizers import ScaleLayout
from loopquant.serving import quantize_ouro_native
from vllm_rlt.config import CacheConfig, ExecutionConfig, ExitConfig, SchedulerConfig
from vllm_rlt.engine.llm_engine import LLMEngine
from vllm_rlt.models.ouro import OuroConfig, OuroForCausalLM

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--case", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
case = json.loads(args.case.read_text())
assert case["trial_kind"] == "cost_profile" and case["policy"] in ["BF16", "DYN", "SH1"]
folder = Path(case["model_path"])
manifest = json.loads((folder / "verified-manifest.json").read_text())
assert (
    manifest["independently_verified"]
    and manifest["revision"] == case["model_revision"]
)
trace = Path(case["trace_path"])
assert hashlib.sha256(trace.read_bytes()).hexdigest() == case["trace_sha256"]
requests = [
    TraceRequest(tuple(r["token_ids"]), r["output_tokens"], r["loops"], r["arrival_s"])
    for r in json.loads(trace.read_text())
]
assert len(requests) == 64
torch.manual_seed(case["seed"])
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
config = OuroConfig.from_dict(json.loads((folder / "config.json").read_text()))
assert config.total_ut_steps == case["prefill_loops"] == 4
with torch.device("meta"):
    model = OuroForCausalLM(config)
load_weights(model, folder, torch.device("cuda"), torch.bfloat16)
model.model.rotary_emb.inv_freq = model.model.rotary_emb.frequencies(device="cuda")
if case["policy"] != "BF16":
    scales = {}
    if case["policy"] == "SH1":
        path = Path(case["activation_scales"])
        assert (
            hashlib.sha256(path.read_bytes()).hexdigest()
            == case["activation_scales_sha256"]
        )
        phases = json.loads(path.read_text())
        scales = {
            name: torch.tensor(value, device="cuda").maximum(
                torch.tensor(phases["decode"]["SH1"][name], device="cuda")
            )
            for name, value in phases["prefill"]["SH1"].items()
        }
    quantize_ouro_native(
        model,
        scales,
        ScaleLayout(4),
        "dynamic" if case["policy"] == "DYN" else "static",
        max_rows=case["scheduler"]["max_num_batched_tokens"],
        fuse_norm=True,
    )
engine = LLMEngine(
    model,
    cache_config=CacheConfig(**case["cache"]),
    scheduler_config=SchedulerConfig(**case["scheduler"]),
    execution_config=ExecutionConfig(**case["execution"]),
    exit_config=ExitConfig(**case["exit"]),
    attention_backend="triton",
)
started = time.time()
with torch.inference_mode():
    run_trace(engine, requests[:32], concurrency=32, records=[], queue_samples=[])
    torch.cuda.synchronize()
    torch.cuda.profiler.start()
    with torch.cuda.nvtx.range("engine-" + case["policy"] + "-C32-B512"):
        result = run_trace(
            engine, requests, concurrency=32, records=[], queue_samples=[]
        )
        torch.cuda.synchronize()
    torch.cuda.profiler.stop()
assert result.completed_requests == 64 and result.output_tokens == 2048
assert (
    result.failed_requests == result.kv_used_after_drain == result.graph_captures == 0
)
args.output.write_text(
    json.dumps(
        dict(
            scope="Nsight-instrumented cost diagnostic; do not use as E2E speedup",
            policy=case["policy"],
            result=asdict(result),
            pid=os.getpid(),
            start_unix=started,
            end_unix=time.time(),
            case_sha256=hashlib.sha256(args.case.read_bytes()).hexdigest(),
            runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        ),
        indent=2,
    )
    + "\n"
)
