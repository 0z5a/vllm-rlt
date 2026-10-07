"""Download then audit one checkpoint while retaining the original H20 locks."""

import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root = Path("/root/autodl-tmp/0z5a/loopquant-h20-20261006")
label = "h20-ouro-2.6b-native-g0-v1"
window = root / label
source = root / ("source-" + label)
model = root / "models/ouro-2.6b"
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
    source.mkdir(exist_ok=False)
    subprocess.run(
        ["tar", "-xzf", str(window / "source.tar.gz"), "-C", str(source)], check=True
    )
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
        "download": [
            sys.executable,
            str(window / "download.py"),
            "--metadata",
            str(window / "model-metadata.json"),
            "--endpoint",
            "https://hf-mirror.com",
            "--destination",
            str(model),
        ],
    }
    for backend in ("torch", "triton"):
        for reference in ("auto", "math"):
            commands[backend + "_" + reference] = [
                sys.executable,
                str(window / "probe.py"),
                "--model",
                str(model),
                "--model-plan",
                str(window / "model-plan.json"),
                "--family",
                "ouro",
                "--requests",
                str(window / "requests.json"),
                "--backend",
                backend,
                "--reference-sdpa",
                reference,
                "--output",
                str(root / f"evidence/{label}-{backend}-{reference}.json"),
            ]
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
        if phase == "download" and code != 0:
            break
    code = 0 if all(row["returncode"] == 0 for row in receipt["children"]) else 1
    receipt.update(returncode=code, end_unix=time.time())
    (root / f"evidence/{label}-receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    print(json.dumps(receipt), flush=True)
raise SystemExit(code)
