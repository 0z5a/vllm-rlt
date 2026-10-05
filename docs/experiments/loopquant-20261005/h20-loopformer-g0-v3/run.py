"""Receive verified peer assets then audit one checkpoint while retaining the original H20 locks."""

import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from materialize_verified_tar import TransferFile, materialize

root = Path("/root/autodl-tmp/0z5a/loopquant-h20-20261006")
label = "h20-loopformer-native-g0-v3"
window = root / label
source = root / ("source-" + label)
model = root / "models/loopformer-3block-8iterations"
locks = Path("/root/autodl-tmp/0z5a-coordination/h20")
manifest = json.loads((window / "manifest.json").read_text())
assert (
    Path("/proc/sys/kernel/random/boot_id").read_text().strip() == manifest["boot_id"]
)
for name, expected in manifest["original_locks"].items():
    stat = (locks / name).stat()
    assert (stat.st_ino, stat.st_dev) == (expected["inode"], expected["device"])


def start_ticks(pid: int) -> int:
    return int(Path(f"/proc/{pid}/stat").read_text().split(") ", 1)[1].split()[19])


with (locks / "gpu0.lock").open("r") as gpu, (locks / "io.lock").open("r") as io:
    fcntl.flock(gpu, fcntl.LOCK_EX | fcntl.LOCK_NB)
    fcntl.flock(io, fcntl.LOCK_EX | fcntl.LOCK_NB)
    uuid = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True
    ).strip()
    assert uuid == manifest["target_uuid"]
    assert not subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
    ).strip()
    for name, digest in manifest["files"].items():
        assert hashlib.sha256((window / name).read_bytes()).hexdigest() == digest
    environment = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES="0",
        OMP_NUM_THREADS="4",
        MKL_NUM_THREADS="4",
        PYTHONPATH=str(source),
        TRITON_CACHE_DIR=str(root / "triton-cache"),
        CUDA_CACHE_PATH=str(root / "cuda-cache"),
    )
    commands = {
        "probe": [
            sys.executable,
            str(window / "probe.py"),
            "--model",
            str(model),
            "--requests",
            str(window / "requests.json"),
            "--output",
            str(root / f"evidence/{label}.json"),
        ],
    }
    receipt = dict(
        controller_pid=os.getpid(),
        controller_start_ticks=start_ticks(os.getpid()),
        start_unix=time.time(),
        boot_id=manifest["boot_id"],
        gpu_uuid=uuid,
        original_locks=manifest["original_locks"],
        children=[],
        manifest_sha256=hashlib.sha256(
            (window / "manifest.json").read_bytes()
        ).hexdigest(),
        model_hold="retained for remaining registered quantization and E2E work",
    )
    receipt["phase"] = "transfer"
    (root / f"evidence/{label}-running.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    plan = json.loads((window / "model-plan.json").read_text())
    files = [
        TransferFile(row["name"], row["size"], row["sha256"]) for row in plan["files"]
    ]
    transferred = materialize(sys.stdin.buffer, model, files)
    verified = dict(
        model=plan["model"],
        revision=plan["revision"],
        files=transferred,
        independently_verified=True,
        transport="read-only peer tar stream",
    )
    (model / "verified-manifest.json").write_text(json.dumps(verified, indent=2) + "\n")
    receipt["transfer"] = verified
    receipt["phase"] = "probe"
    print(
        json.dumps(
            dict(
                event="CHECKPOINT_TRANSFER_VERIFIED",
                model=str(model),
                files=transferred,
            )
        ),
        flush=True,
    )
    source.mkdir(exist_ok=False)
    subprocess.run(
        ["tar", "-xzf", str(window / "source.tar.gz"), "-C", str(source)], check=True
    )
    for phase, command in commands.items():
        with (root / f"evidence/{label}-{phase}.log").open("x") as log:
            child = subprocess.Popen(
                command,
                cwd=source,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            current = dict(
                phase=phase,
                child_pid=child.pid,
                child_start_ticks=start_ticks(child.pid),
                start_unix=time.time(),
                command=command,
            )
            receipt["children"].append(current)
            (root / f"evidence/{label}-running.json").write_text(
                json.dumps(receipt, indent=2) + "\n"
            )
            print(json.dumps(dict(event="started", **current)), flush=True)
            code = child.wait()
        current.update(returncode=code, end_unix=time.time())
        print(json.dumps(dict(event="completed", **current)), flush=True)
        if code != 0:
            break
    receipt.update(returncode=code, end_unix=time.time())
    (root / f"evidence/{label}-receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    print(json.dumps(receipt), flush=True)
raise SystemExit(code)
