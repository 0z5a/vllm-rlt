"""Finite Huginn native INT4 Graph-off/on preflight; no full-weight quality claim."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time

import torch

from tests.loopquant_huginn_int4_checks import check_huginn_int4

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--locks", type=Path, required=True)
parser.add_argument("--gpu-index", type=int, choices=[0, 1], required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
records = []
with (args.locks / f"gpu{args.gpu_index}-perf.lock").open("r") as gpu:
    fcntl.flock(gpu, fcntl.LOCK_EX)
    started = time.time()
    for graphs in [False, True]:
        folder = args.output.parent / "huginn-int4-v1-exports" / f"graphs-{int(graphs)}"
        with (args.locks / "heavy-io.lock").open("r") as io:
            fcntl.flock(io, fcntl.LOCK_EX)
            row = check_huginn_int4(folder, graphs)
            row["export_manifest"] = json.loads((folder / "manifest.json").read_text())
            (folder / "tensors.pt").unlink()
        row["temporary_weight_payload_removed"] = True
        records.append(row)
        print(json.dumps(dict(event="pass", graphs=graphs)), flush=True)
    torch.cuda.synchronize()
    receipt = dict(
        status="pass",
        scope="two tiny native cases; no full-weight quality or performance claim",
        pid=os.getpid(),
        start_unix=started,
        end_unix=time.time(),
        records=records,
        torch_version=torch.__version__,
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    with (args.locks / "heavy-io.lock").open("r") as io:
        fcntl.flock(io, fcntl.LOCK_EX)
        args.output.write_text(json.dumps(receipt, indent=2) + "\n")
