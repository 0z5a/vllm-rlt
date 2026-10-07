"""Run four finite children in order while retaining both original H20 resource locks."""

import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root = Path("/root/autodl-tmp/0z5a/loopquant-h20-20261006")
window = root / "h20-boundary-fix-v2"
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
    sources = {}
    for name in manifest["sources"]:
        source = root / ("source-h20-boundary-fix-v2-" + name)
        source.mkdir(exist_ok=False)
        subprocess.run(
            ["tar", "-xzf", str(window / (name + ".tar.gz")), "-C", str(source)],
            check=True,
        )
        sources[name] = source
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
    )
    for family, source_key in manifest["families"].items():
        source = sources[source_key]
        command = [
            sys.executable,
            str(window / "probe.py"),
            "--family",
            family,
            "--output",
            str(root / f"evidence/h20-boundary-fix-v2-{family}.json"),
        ]
        environment = dict(
            os.environ,
            CUDA_VISIBLE_DEVICES="0",
            OMP_NUM_THREADS="4",
            MKL_NUM_THREADS="4",
            PYTHONPATH=str(source),
            TRITON_CACHE_DIR=str(root / "triton-cache"),
            CUDA_CACHE_PATH=str(root / "cuda-cache"),
        )
        with (root / f"evidence/h20-boundary-fix-v2-{family}.log").open("x") as log:
            child = subprocess.Popen(
                command,
                cwd=source,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            current = dict(
                family=family,
                child_pid=child.pid,
                child_start_ticks=start_ticks(child.pid),
                start_unix=time.time(),
                command=command,
            )
            receipt["children"].append(current)
            (root / "evidence/h20-boundary-fix-v2-running.json").write_text(
                json.dumps(receipt, indent=2) + "\n"
            )
            print(json.dumps(dict(event="started", **current)), flush=True)
            code = child.wait()
        current.update(returncode=code, end_unix=time.time())
        (root / "evidence/h20-boundary-fix-v2-progress.json").write_text(
            json.dumps(receipt, indent=2) + "\n"
        )
        print(json.dumps(dict(event="completed", **current)), flush=True)
    code = 0 if all(row["returncode"] == 0 for row in receipt["children"]) else 1
    receipt.update(returncode=code, end_unix=time.time())
    (root / "evidence/h20-boundary-fix-v2-receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    print(json.dumps(receipt), flush=True)
raise SystemExit(code)
