"""Finite Huginn FP8 and explicit loop-metadata qualification."""

import argparse
import fcntl
import hashlib
import itertools
import json
import os
from pathlib import Path
import time

import torch

from tests.loopquant_huginn_fp8_checks import (
    check_graph_loop_metadata,
    check_huginn_fp8,
)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--locks", type=Path, required=True)
parser.add_argument("--gpu-index", type=int, choices=[0, 1], required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
records = []
with (args.locks / f"gpu{args.gpu_index}-perf.lock").open("r") as gpu:
    fcntl.flock(gpu, fcntl.LOCK_EX)
    started = time.time()
    for asynchronous in [False, True]:
        row = check_graph_loop_metadata(asynchronous)
        records.append(dict(kind="row_loop_metadata", **row))
        print(
            json.dumps(dict(event="metadata_pass", asynchronous=asynchronous)),
            flush=True,
        )
    cases = itertools.product(
        ["DYN", "SH1", "ST2", "LOOP"], [False, True], [False, True]
    )
    for index, (policy, graphs, fuse_norm) in enumerate(cases):
        folder = args.output.parent / "huginn-fp8-v1-exports" / f"case-{index:02d}"
        with (args.locks / "heavy-io.lock").open("r") as io:
            fcntl.flock(io, fcntl.LOCK_EX)
            row = check_huginn_fp8(folder, policy, graphs, fuse_norm)
            row["export_manifest"] = json.loads((folder / "manifest.json").read_text())
            (folder / "tensors.pt").unlink()
        records.append(
            dict(kind="huginn_fp8", temporary_weight_payload_removed=True, **row)
        )
        print(
            json.dumps(
                dict(
                    event="case_pass",
                    index=index,
                    policy=policy,
                    graphs=graphs,
                    fuse_norm=fuse_norm,
                )
            ),
            flush=True,
        )
    torch.cuda.synchronize()
    receipt = dict(
        status="pass",
        scope="16 tiny Huginn FP8 and2 metadata cases; no full-weight quality or performance claim",
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
