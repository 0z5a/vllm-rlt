"""Fixed native INT4 diagnostic at R1–R4; no calibration search or locked-test selection."""

import argparse
import fcntl
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import time

import torch

from loopquant.checkpoint import load_weights
from loopquant.int4_export import export_int4_model, load_int4_model
from loopquant.quality import native_window_nll
from vllm_rlt.core.kv_cache_manager import KVCacheManager
from vllm_rlt.models.ouro import OuroConfig, OuroForCausalLM

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--model", type=Path, required=True)
parser.add_argument("--windows", type=Path, required=True)
parser.add_argument(
    "--policy",
    choices=["BF16", "RTN", "GPTQ-first", "GPTQ-all", "GPTQ-matched"],
    required=True,
)
parser.add_argument("--native-export", type=Path)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--locks", type=Path, required=True)
parser.add_argument("--gpu-index", type=int, choices=[0, 1], required=True)
args = parser.parse_args()
if args.policy.startswith("GPTQ-") and args.native_export is None:
    parser.error("GPTQ requires its calibrated native export")
if args.native_export is not None and args.policy in {"BF16", "RTN"}:
    parser.error("this packet only reloads supplied GPTQ exports")
revision = "574fa66cb8bf5abdc979642d01cf2b79b16bfab1"
windows_hash = hashlib.sha256(args.windows.read_bytes()).hexdigest()
assert (
    windows_hash == "325f23166ecec45f22a8a8254dc5edc7bf465b1d456beff801b58c7a07770c1c"
)
windows = [json.loads(line) for line in args.windows.read_text().splitlines()]
assert len(windows) == 128 and all(row["split"] == "dev" for row in windows)
manifest = json.loads((args.model / "verified-manifest.json").read_text())
assert manifest["independently_verified"] and manifest["revision"] == revision
config = OuroConfig.from_dict(json.loads((args.model / "config.json").read_text()))
assert config.total_ut_steps == 4
length = max(len(row["token_ids"]) for row in windows)
args.output.mkdir(exist_ok=False, parents=True)
with (args.locks / f"gpu{args.gpu_index}-perf.lock").open("a") as gpu_lock:
    fcntl.flock(gpu_lock, fcntl.LOCK_EX)
    started = time.time()
    torch.manual_seed(17)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    export_manifest_hash = None
    if args.native_export is None:
        with torch.device("meta"):
            model = OuroForCausalLM(config)
        with (args.locks / "heavy-io.lock").open("a") as io_lock:
            fcntl.flock(io_lock, fcntl.LOCK_EX)
            load_weights(model, args.model, torch.device("cuda"), torch.bfloat16)
        model.model.rotary_emb.inv_freq = model.model.rotary_emb.frequencies(
            device="cuda"
        )
        model.eval().requires_grad_(False)
    if args.policy != "BF16":
        export = args.native_export or args.output / "export"
        if args.policy == "RTN":
            with (args.locks / "heavy-io.lock").open("a") as io_lock:
                fcntl.flock(io_lock, fcntl.LOCK_EX)
                export_int4_model(
                    model, export, model_revision=revision, group_size=128
                )
            del model
            gc.collect()
            torch.cuda.empty_cache()
        exported = json.loads((export / "manifest.json").read_text())
        assert (
            exported["model_revision"] == revision and exported["method"] == args.policy
        )
        assert exported["group_size"] == 128 and exported["packed_weight_copies"] == 1
        with (args.locks / "heavy-io.lock").open("a") as io_lock:
            fcntl.flock(io_lock, fcntl.LOCK_EX)
            model = load_int4_model(export, torch.device("cuda"))
            duplicate = load_int4_model(export, torch.device("cuda"))
        state, restored = model.state_dict(), duplicate.state_dict()
        assert state.keys() == restored.keys()
        for name in state:
            assert torch.equal(
                state[name].view(torch.uint8), restored[name].view(torch.uint8)
            ), name
        del duplicate, state, restored
        gc.collect()
        torch.cuda.empty_cache()
        export_manifest_hash = hashlib.sha256(
            (export / "manifest.json").read_bytes()
        ).hexdigest()
    cache = KVCacheManager(
        config.num_hidden_layers,
        config.num_key_value_heads,
        config.head_dim,
        4 * math.ceil(length / 16),
        16,
        4,
        device="cuda",
        dtype=torch.bfloat16,
        backend="triton",
    )
    arms = []
    for loops in [1, 2, 3, 4]:
        documents = {}
        with (args.output / f"windows-r{loops}.jsonl").open("x") as stream:
            for row in windows:
                tokens = torch.tensor(row["token_ids"], device="cuda", dtype=torch.long)
                nll, count = native_window_nll(model, tokens, loops, cache)
                stream.write(
                    json.dumps(
                        dict(
                            document_id=row["document_id"],
                            start=row["start"],
                            content_sha256=row["content_sha256"],
                            nll=nll,
                            tokens=count,
                        )
                    )
                    + "\n"
                )
                stream.flush()
                document = documents.setdefault(
                    row["document_id"], dict(nll=0.0, tokens=0)
                )
                document["nll"] += nll
                document["tokens"] += count
        total_nll = sum(row["nll"] for row in documents.values())
        total_tokens = sum(row["tokens"] for row in documents.values())
        assert total_tokens == 62361 and cache.num_used_blocks == 0
        arm = dict(
            loops=loops,
            windows=128,
            documents=documents,
            nll=total_nll,
            tokens=total_tokens,
            perplexity=math.exp(total_nll / total_tokens),
            kv_used_after_drain=cache.num_used_blocks,
        )
        arms.append(arm)
        print(
            json.dumps(
                {key: value for key, value in arm.items() if key != "documents"}
            ),
            flush=True,
        )
    result = dict(
        status="complete",
        policy=args.policy,
        group_size=128 if args.policy != "BF16" else None,
        scope="fixed diagnostic dev curves; no hyperparameter search, locked G2 or performance claim",
        model_revision=revision,
        windows_sha256=windows_hash,
        arms=arms,
        native_export_manifest_sha256=export_manifest_hash,
        encoded_reload_exact=args.policy != "BF16",
        pid=os.getpid(),
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        start_unix=started,
        end_unix=time.time(),
    )
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
