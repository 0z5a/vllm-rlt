"""Run one frozen HRM CPU comparison and retain the natural child wait status."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root = Path(__file__).parent
packet = root / "snapshots/hrm-official-cpu-sdpa-v1"
manifest = json.loads((packet / "manifest.json").read_text())
for name, digest in manifest["files"].items():
    assert hashlib.sha256((packet / name).read_bytes()).hexdigest() == digest
source = root / "snapshots/hrm-adapter-cpu-v1/source"
command = [
    sys.executable,
    str(packet / "run.py"),
    "--root",
    str(root),
    "--requests",
    str(packet / "requests.json"),
    "--author-root",
    str(packet / "author"),
]
environment = dict(
    os.environ,
    CUDA_VISIBLE_DEVICES="",
    OMP_NUM_THREADS="2",
    MKL_NUM_THREADS="2",
    TOKENIZERS_PARALLELISM="false",
    PYTHONPATH=str(source),
)
receipt = dict(
    controller_pid=os.getpid(),
    start_unix=time.time(),
    command=command,
    manifest_sha256=hashlib.sha256((packet / "manifest.json").read_bytes()).hexdigest(),
)
with (root / "evidence/official-hrm-adapter-cpu-sdpa-attempt1.log").open("x") as log:
    child = subprocess.Popen(
        command, cwd=source, env=environment, stdout=log, stderr=subprocess.STDOUT
    )
    receipt["child_pid"] = child.pid
    (root / "evidence/official-hrm-adapter-cpu-sdpa-running.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    print(json.dumps(receipt), flush=True)
    code = child.wait()
receipt.update(returncode=code, end_unix=time.time())
(root / "evidence/official-hrm-adapter-cpu-sdpa-receipt.json").write_text(
    json.dumps(receipt, indent=2) + "\n"
)
print(json.dumps(receipt), flush=True)
raise SystemExit(code)
