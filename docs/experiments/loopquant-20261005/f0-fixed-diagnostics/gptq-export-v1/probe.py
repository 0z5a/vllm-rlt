"""Fixed first/all/matched-row GPTQ exports from the native BF16 recurrent core."""

import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import time

import torch
from torch import nn

from loopquant.checkpoint import load_weights
from loopquant.gptq import gptq_int4
from loopquant.gptq_trace import HessianTrace
from loopquant.int4_export import export_int4_model
from loopquant.trace import TracedOuroForCausalLM
from vllm_rlt.core.kv_cache_manager import KVCacheManager
from vllm_rlt.models.ouro import OuroConfig


def collect(
    model: TracedOuroForCausalLM, windows: list[dict], device: torch.device
) -> list[dict]:
    config = model.config
    length = max(len(row["token_ids"]) for row in windows)
    cache = KVCacheManager(
        config.num_hidden_layers,
        config.num_key_value_heads,
        config.head_dim,
        4 * math.ceil(length / 16),
        16,
        4,
        device=device,
        dtype=torch.bfloat16,
        backend="triton" if device.type == "cuda" else "torch",
    )
    with torch.inference_mode():
        for row in windows:
            request = row["document_id"] + ":" + str(row["start"])
            tokens = torch.tensor(row["token_ids"], device=device, dtype=torch.long)
            assert cache.allocate(request, len(tokens))
            state = model.prelude(tokens)
            for loop in range(4):
                state, _ = model.recurrent(
                    state,
                    [request] * len(tokens),
                    [loop] * len(tokens),
                    list(range(len(tokens))),
                    cache,
                    compute_gate=False,
                )
            cache.free(request)
            assert not cache.num_used_blocks
    return model.trace.summary()


def export_baselines(
    model: TracedOuroForCausalLM, output: Path, revision: str, io_lock_path: Path
) -> list[dict]:
    projections = {
        name: layer
        for name, layer in model.named_modules()
        if name.startswith("model.layers.") and isinstance(layer, nn.Linear)
    }
    assert len(projections) == 7 * model.config.num_hidden_layers
    outputs = []
    for policy in ["first", "all", "matched"]:
        packed = {}
        for name, layer in projections.items():
            canonical = model.trace.aliases[name]
            hessian = model.trace.hessians[canonical, policy].matrix()
            codes, scales = gptq_int4(
                layer.weight, hessian, group_size=128, block_size=128, damping=0.01
            )
            packed[name] = codes.cpu(), scales.cpu()
            del hessian, codes, scales
        folder = output / ("gptq-" + policy)
        with io_lock_path.open("a") as io_lock:
            fcntl.flock(io_lock, fcntl.LOCK_EX)
            export_int4_model(
                model,
                folder,
                model_revision=revision,
                method="GPTQ-" + policy,
                packed=packed,
                group_size=128,
            )
        record = dict(
            policy=policy,
            physical_matrices=len(packed),
            manifest_sha256=hashlib.sha256(
                (folder / "manifest.json").read_bytes()
            ).hexdigest(),
        )
        outputs.append(record)
        print(json.dumps(record), flush=True)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--locks", type=Path, required=True)
    parser.add_argument("--gpu-index", type=int, choices=[0, 1], required=True)
    args = parser.parse_args()
    revision = "574fa66cb8bf5abdc979642d01cf2b79b16bfab1"
    calibration_hash = hashlib.sha256(args.calibration.read_bytes()).hexdigest()
    assert (
        calibration_hash
        == "b33e74f657049f725ca1fb025213110789594d2118d148f85fb2bf192e620d68"
    )
    windows = [json.loads(line) for line in args.calibration.read_text().splitlines()]
    assert len(windows) == 128 and all(row["split"] == "calibration" for row in windows)
    prompts = {
        row["document_id"] + ":" + str(row["start"]): len(row["token_ids"])
        for row in windows
    }
    assert len(prompts) == len(windows)
    manifest = json.loads((args.model / "verified-manifest.json").read_text())
    assert manifest["independently_verified"] and manifest["revision"] == revision
    config = OuroConfig.from_dict(json.loads((args.model / "config.json").read_text()))
    assert config.total_ut_steps == 4
    args.output.mkdir(exist_ok=False, parents=True)
    with (args.locks / f"gpu{args.gpu_index}-perf.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        started = time.time()
        torch.manual_seed(17)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
        trace = HessianTrace(prompts, 4, seed=17)
        with torch.device("meta"):
            model = TracedOuroForCausalLM(config, trace)
        with (args.locks / "heavy-io.lock").open("a") as io_lock:
            fcntl.flock(io_lock, fcntl.LOCK_EX)
            load_weights(model, args.model, torch.device("cuda"), torch.bfloat16)
        model.model.rotary_emb.inv_freq = model.model.rotary_emb.frequencies(
            device="cuda"
        )
        model.eval().requires_grad_(False)
        summary = collect(model, windows, torch.device("cuda"))
        assert len(summary) == 24 * 4 * 3
        for row in summary:
            assert row["total_rows"] == (251972 if row["policy"] == "all" else 62993)
        (args.output / "hessian-summary.json").write_text(
            json.dumps(summary, indent=2) + "\n"
        )
        outputs = export_baselines(
            model, args.output, revision, args.locks / "heavy-io.lock"
        )
        result = dict(
            status="complete",
            scope="fixed calibration PTQ exports; native quality not yet run",
            pid=os.getpid(),
            model_revision=revision,
            calibration_sha256=calibration_hash,
            group_size=128,
            block_size=128,
            damping=0.01,
            selection_seed=17,
            gram_bytes=sum(row["gram_bytes"] for row in summary),
            exports=outputs,
            start_unix=started,
            end_unix=time.time(),
            cuda_peak_bytes=torch.cuda.max_memory_allocated(),
            runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        )
        (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
