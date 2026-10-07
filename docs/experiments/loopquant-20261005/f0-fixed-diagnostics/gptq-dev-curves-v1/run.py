"""Three finite fixed GPTQ dev diagnostics; original locks and natural child completion."""

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
root = Path("PRIVATE_PATH")
packet, source = root / "gptq-dev-curves-v1", root / "source-gptq-dev-curves-v1"
with (root.parent / "heavy-io.lock").open("a") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    for name, digest in json.loads((packet / "manifest.json").read_text())[
        "files"
    ].items():
        assert hashlib.sha256((packet / name).read_bytes()).hexdigest() == digest
    parent = json.loads((root / "evidence/gptq-export-v1/result.json").read_text())
    assert parent["status"] == "complete"
    assert (
        parent["runner_sha256"]
        == "5512f931defe3fb41bb91f4e89c0a6a8c1ca0552e7ff3ba3c67d45519fe2dfce"
    )
    assert {row["policy"] for row in parent["exports"]} == {"first", "all", "matched"}
    for row in parent["exports"]:
        artifact = (
            root
            / "evidence/gptq-export-v1"
            / ("gptq-" + row["policy"])
            / "manifest.json"
        )
        assert (
            hashlib.sha256(artifact.read_bytes()).hexdigest() == row["manifest_sha256"]
        )
    source.mkdir(exist_ok=False)
    subprocess.run(
        ["tar", "-xzf", str(packet / "source.tar.gz"), "-C", str(source)], check=True
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
receipt = dict(controller_pid=os.getpid(), start_unix=time.time(), records=[])
for policy in ["GPTQ-first", "GPTQ-all", "GPTQ-matched"]:
    name = "gptq-dev-curves-v1-" + policy.lower()
    command = [
        sys.executable,
        str(packet / "probe.py"),
        "--gpu-index",
        str(gpu),
        "--locks",
        str(root.parent),
        "--policy",
        policy,
        "--native-export",
        str(root / "evidence/gptq-export-v1" / policy.lower()),
        "--model",
        str(root / "models/ouro-1.4b"),
        "--windows",
        str(packet / "dev.jsonl"),
        "--output",
        str(root / "evidence" / name),
    ]
    record = dict(policy=policy, start_unix=time.time(), command=command)
    with (root / "evidence" / (name + ".log")).open("x") as log:
        child = subprocess.Popen(
            command, cwd=source, env=environment, stdout=log, stderr=subprocess.STDOUT
        )
        record["child_pid"] = child.pid
        print(json.dumps(record), flush=True)
        code = child.wait()
    record.update(returncode=code, end_unix=time.time())
    receipt["records"].append(record)
    receipt["end_unix"] = time.time()
    with (root.parent / "heavy-io.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        (root / "evidence/gptq-dev-curves-v1-receipt.json").write_text(
            json.dumps(receipt, indent=2) + "\n"
        )
    print(json.dumps(record), flush=True)
    if code:
        sys.exit(code)
