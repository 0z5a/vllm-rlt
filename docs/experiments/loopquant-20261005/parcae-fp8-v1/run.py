"""One finite child, immutable source and original coordinator locks."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--gpu-index", type=int, choices=[0, 1], required=True)
gpu = parser.parse_args().gpu_index
root = Path("/root/autodl-tmp/0z5a/loopquant-20261005")
window, source = root / "parcae-fp8-v1", root / "source-parcae-fp8-v1"
with (root.parent / "heavy-io.lock").open("r") as io:
    fcntl.flock(io, fcntl.LOCK_EX)
    for name, digest in json.loads((window / "manifest.json").read_text())[
        "files"
    ].items():
        assert hashlib.sha256((window / name).read_bytes()).hexdigest() == digest
    source.mkdir(exist_ok=False)
    subprocess.run(
        ["tar", "-xzf", str(window / "source.tar.gz"), "-C", str(source)], check=True
    )
environment = dict(
    os.environ,
    CUDA_VISIBLE_DEVICES=str(gpu),
    OMP_NUM_THREADS="4",
    MKL_NUM_THREADS="4",
    PYTHONPATH=str(source),
    TRITON_CACHE_DIR=str(root / "triton-cache"),
    CUDA_CACHE_PATH=str(root / "cuda-cache"),
)
receipt = dict(controller_pid=os.getpid(), start_unix=time.time())
command = [
    sys.executable,
    str(window / "probe.py"),
    "--gpu-index",
    str(gpu),
    "--locks",
    str(root.parent),
    "--output",
    str(root / "evidence/parcae-fp8-v1.json"),
]
with (root / "evidence/parcae-fp8-v1.log").open("x") as log:
    child = subprocess.Popen(
        command, cwd=source, env=environment, stdout=log, stderr=subprocess.STDOUT
    )
    receipt.update(child_pid=child.pid, command=command)
    print(json.dumps(receipt), flush=True)
    (root / "evidence/parcae-fp8-v1-running.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    code = child.wait()
receipt.update(returncode=code, end_unix=time.time())
with (root.parent / "heavy-io.lock").open("r") as io:
    fcntl.flock(io, fcntl.LOCK_EX)
    (root / "evidence/parcae-fp8-v1-receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
print(json.dumps(receipt), flush=True)
sys.exit(code)
