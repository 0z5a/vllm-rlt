"""Finite Ouro Q0 runs over a registered pool, with native export and resumption."""

import fcntl
import hashlib
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

import torch

from vllm_rlt.models.ouro import OuroConfig, OuroForCausalLM

from .checkpoint import load_weights
from .data import Window, training_order, training_windows
from .native_export import _digest, export_native_ouro
from .qat import deploy_ouro_q0, prepare_ouro_q0
from .quantizers import ScaleLayout
from .train import Q0Config, Q0Trainer, TrainingBatch


def run_ouro_q0(config: dict, output: Path, resume: Path | None = None) -> None:
    if config["format"] != "fp8_e4m3fn":
        raise ValueError("the Q0 runner implements E4M3 FP8 only")
    model_path, windows_path, scales_path = (
        Path(config[key]) for key in ("model", "windows", "scales")
    )
    for key, path in (("windows", windows_path), ("scales", scales_path)):
        if _digest(path) != config[key + "_sha256"]:
            raise ValueError(f"registered {key} hash mismatch")
    manifest = json.loads((model_path / "verified-manifest.json").read_text())
    if not manifest["independently_verified"] or manifest["revision"] != config["model_revision"]:
        raise ValueError("Q0 requires the verified pinned model")
    windows = [
        Window(**{**row, "token_ids": tuple(row["token_ids"])})
        for row in map(json.loads, windows_path.read_text().splitlines())
    ]
    length, seed = config["sequence_length"], config["seed"]
    pool = training_windows(windows, length)
    q0 = Q0Config(**config["training"])
    if q0.tokens_per_update % length or config["updates"] < 1:
        raise ValueError("the finite update budget must contain full microbatches")
    batch_count = q0.tokens_per_update // length
    architecture = OuroConfig.from_dict(json.loads((model_path / "config.json").read_text()))
    if q0.loops != architecture.total_ut_steps:
        raise ValueError("the initial Q0 schedule requires the model's native loop depth")
    boundaries = {
        "SH1": (),
        "ST2": (q0.loops // 2,),
        "LOOP": tuple(range(1, q0.loops)),
    }[config["policy"]]
    layout = ScaleLayout(q0.loops, boundaries)
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    locks = Path(config["locks"])
    with (locks / f"gpu{config['gpu_index']}-perf.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        started = time.time()
        torch.manual_seed(seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
        with torch.device("meta"):
            model = OuroForCausalLM(architecture)
        with (locks / "heavy-io.lock").open("a") as io_lock:
            fcntl.flock(io_lock, fcntl.LOCK_EX)
            load_weights(model, model_path, torch.device("cuda"), torch.bfloat16)
        model.model.rotary_emb.inv_freq = model.model.rotary_emb.frequencies(device="cuda")
        phases = json.loads(scales_path.read_text())
        scales = {
            name: torch.tensor(value, device="cuda").maximum(
                torch.tensor(phases["decode"][config["policy"]][name], device="cuda")
            )
            for name, value in phases["prefill"][config["policy"]].items()
        }
        student = prepare_ouro_q0(model, scales, layout)
        trainer = Q0Trainer(student, q0)
        if resume is not None:
            previous = json.loads((resume.parent / "config.json").read_text())
            for key in (
                "model_revision",
                "runtime_revision",
                "windows_sha256",
                "scales_sha256",
                "policy",
                "sequence_length",
                "seed",
                "training",
            ):
                if previous[key] != config[key]:
                    raise ValueError(f"resumption changed the registered {key}")
            with (locks / "heavy-io.lock").open("a") as io_lock:
                fcntl.flock(io_lock, fcntl.LOCK_EX)
                trainer.resume(resume)
        if trainer.updates >= config["updates"]:
            raise ValueError("the requested update budget is already complete")
        frozen = {name: (p.data_ptr(), p._version) for name, p in model.named_parameters()}
        initial_updates, initial_tokens = trainer.updates, trainer.cumulative_tokens
        with (output / "updates.jsonl").open("x") as stream:
            while trainer.updates < config["updates"]:
                indices = training_order(len(pool), seed, trainer.data_position, batch_count)
                batches = [
                    TrainingBatch(
                        torch.tensor([pool[index].token_ids], device="cuda", dtype=torch.long),
                        torch.ones(1, length, device="cuda", dtype=torch.bool),
                    )
                    for index in indices
                ]
                torch.cuda.synchronize()
                update_started = time.perf_counter()
                record = trainer.step(batches)
                torch.cuda.synchronize()
                record.update(seconds=time.perf_counter() - update_started, pool_indices=indices)
                if frozen != {
                    name: (p.data_ptr(), p._version) for name, p in model.named_parameters()
                } or any(p.grad is not None for p in model.parameters()):
                    raise ValueError("Q0 modified a protected model parameter")
                stream.write(json.dumps(record) + "\n")
                stream.flush()
                print(
                    json.dumps(
                        {
                            key: record[key]
                            for key in (
                                "update",
                                "seconds",
                                "losses",
                                "gradient_norm",
                                "cumulative_tokens",
                            )
                        }
                    ),
                    flush=True,
                )
                if trainer.updates % 5 == 0 or trainer.updates == config["updates"]:
                    with (locks / "heavy-io.lock").open("a") as io_lock:
                        fcntl.flock(io_lock, fcntl.LOCK_EX)
                        trainer.save(output / "checkpoint.partial")
                        (output / "checkpoint.partial").replace(output / "checkpoint.pt")
        trainable = {name: list(value.shape) for name, value in trainer.parameters.items()}
        deploy_ouro_q0(student, max_rows=length)
        with (locks / "heavy-io.lock").open("a") as io_lock:
            fcntl.flock(io_lock, fcntl.LOCK_EX)
            export_native_ouro(model, output / "export", model_revision=config["model_revision"])
            checkpoint_sha = _digest(output / "checkpoint.pt")
        result = dict(
            status="complete",
            scope="Q0 training and encoded export; native reload/quality remain separate gates",
            training=asdict(q0),
            policy=config["policy"],
            initial_updates=initial_updates,
            completed_updates=trainer.updates,
            run_tokens=trainer.cumulative_tokens - initial_tokens,
            cumulative_tokens=trainer.cumulative_tokens,
            pool_windows=len(pool),
            pool_unique_tokens=len(pool) * length,
            pool_documents=len({row.document_id for row in pool}),
            omitted_tail_tokens=sum(len(row.token_ids) for row in windows) - len(pool) * length,
            pool_epochs=trainer.data_position / len(pool),
            calibration_document_overlap=len(
                {row.document_id for row in windows if row.split == "calibration"}
            ),
            trainable_parameters=trainable,
            packed_weight_copies=1,
            frozen_weights="grad=None; storage/version unchanged",
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            checkpoint_sha256=checkpoint_sha,
            config_sha256=_digest(output / "config.json"),
            runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            start_unix=started,
            end_unix=time.time(),
            pid=os.getpid(),
            torch_version=torch.__version__,
            cuda_version=torch.version.cuda,
            gpu_name=torch.cuda.get_device_name(),
        )
        (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
