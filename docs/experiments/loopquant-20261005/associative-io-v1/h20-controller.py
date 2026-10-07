"""Finite H20 qualification holding the original GPU0 and IO locks until child exit."""

import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root = Path("PRIVATE_PATH")
window = root / "h20-fp8-primitives-v1"
source = root / "source-h20-fp8-primitives-v1"
locks = Path("PRIVATE_PATH")
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
    fcntl.flock(gpu, fcntl.LOCK_EX)
    fcntl.flock(io, fcntl.LOCK_EX)
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
    command = [
        sys.executable,
        str(window / "probe.py"),
        "--output",
        str(root / "evidence/h20-fp8-primitives-v1.json"),
    ]
    receipt = dict(
        controller_pid=os.getpid(),
        controller_start_ticks=start_ticks(os.getpid()),
        start_unix=time.time(),
        boot_id=manifest["boot_id"],
        gpu_uuid=uuid,
        original_locks=manifest["original_locks"],
        command=command,
        manifest_sha256=hashlib.sha256(
            (window / "manifest.json").read_bytes()
        ).hexdigest(),
    )
    with (root / "evidence/h20-fp8-primitives-v1.log").open("x") as log:
        child = subprocess.Popen(
            command, cwd=source, env=environment, stdout=log, stderr=subprocess.STDOUT
        )
        receipt.update(child_pid=child.pid, child_start_ticks=start_ticks(child.pid))
        (root / "evidence/h20-fp8-primitives-v1-running.json").write_text(
            json.dumps(receipt, indent=2) + "\n"
        )
        print(json.dumps(receipt), flush=True)
        code = child.wait()
    receipt.update(returncode=code, end_unix=time.time())
    (root / "evidence/h20-fp8-primitives-v1-receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    print(json.dumps(receipt), flush=True)
raise SystemExit(code)
