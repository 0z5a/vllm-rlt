"""Three finite Nsight captures. Children never acquire a second copy of the locks."""

import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root = Path("/root/autodl-tmp/0z5a/loopquant-20261005")
window, source = root / "cost-profile-v1", root / "source-cost-profile-v1"
manifest = json.loads((window / "manifest.json").read_text())
output = root / "evidence/cost-profile-v1"
with (
    (root.parent / "gpu0-perf.lock").open("a") as gpu0,
    (root.parent / "gpu1-perf.lock").open("a") as gpu1,
    (root.parent / "heavy-io.lock").open("a") as io,
):
    for lock in [gpu0, gpu1, io]:
        fcntl.flock(lock, fcntl.LOCK_EX)
    for name, digest in manifest["files"].items():
        assert hashlib.sha256((window / name).read_bytes()).hexdigest() == digest
    source.mkdir(exist_ok=False)
    output.mkdir(exist_ok=False)
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
    receipt = dict(controller_pid=os.getpid(), start_unix=time.time(), records=[])
    print(json.dumps(dict(event="start", controller_pid=os.getpid())), flush=True)
    for policy in ["BF16", "DYN", "SH1"]:
        prefix = output / policy.lower()
        command = [
            "/usr/local/bin/nsys",
            "profile",
            "--sample=none",
            "--cpuctxsw=none",
            "--trace=cuda,nvtx,cublas",
            "--cuda-graph-trace=node",
            "--capture-range=cudaProfilerApi",
            "--capture-range-end=stop",
            "--kill=none",
            "--wait=all",
            "--output",
            str(prefix),
            sys.executable,
            str(window / "probe.py"),
            "--case",
            str(window / (policy.lower() + ".json")),
            "--output",
            str(prefix.with_suffix(".json")),
        ]
        with prefix.with_suffix(".log").open("x") as log:
            result = subprocess.run(
                command,
                cwd=source,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        row = dict(
            policy=policy, command=command, returncode=result.returncode, reports=[]
        )
        receipt["records"].append(row)
        if result.returncode == 0:
            for report in [
                "cuda_gpu_kern_sum",
                "cuda_api_sum",
                "cuda_gpu_mem_time_sum",
                "cuda_kern_exec_sum",
                "cublas_api_sum",
            ]:
                command = [
                    "/usr/local/bin/nsys",
                    "stats",
                    "-r",
                    report,
                    "--format",
                    "csv",
                    str(prefix.with_suffix(".nsys-rep")),
                ]
                with (output / (policy.lower() + "-" + report + ".csv")).open(
                    "x"
                ) as log:
                    stats = subprocess.run(
                        command, stdout=log, stderr=subprocess.STDOUT
                    )
                row["reports"].append(
                    dict(name=report, command=command, returncode=stats.returncode)
                )
        receipt["end_unix"] = time.time()
        (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        print(json.dumps(dict(policy=policy, returncode=result.returncode)), flush=True)
        if result.returncode:
            break
sys.exit(
    int(
        any(
            row["returncode"] or any(report["returncode"] for report in row["reports"])
            for row in receipt["records"]
        )
    )
)
