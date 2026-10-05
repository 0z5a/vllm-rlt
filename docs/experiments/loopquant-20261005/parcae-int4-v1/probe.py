"""Finite Parcae INT4 checks; no full-checkpoint quality or timing claim."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time

import torch

from tests.loopquant_int4_checks import check_int4_native_model

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
        for compact in [False, True]:
            label = f"graphs-{int(graphs)}-compact-{int(compact)}"
            folder = args.output.parent / "parcae-int4-v1-exports" / label
            with (args.locks / "heavy-io.lock").open("r") as io:
                fcntl.flock(io, fcntl.LOCK_EX)
                row = check_int4_native_model("parcae", graphs, folder, compact=compact)
                row["export_manifest"] = json.loads(
                    (folder / "manifest.json").read_text()
                )
                (folder / "tensors.pt").unlink()
            row["temporary_weight_payload_removed"] = True
            records.append(row)
            print(json.dumps(dict(event="pass", case=label)), flush=True)
    torch.cuda.synchronize()
    receipt = dict(
        status="pass",
        scope="four tiny native cases; no full-checkpoint quality or performance claim",
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
