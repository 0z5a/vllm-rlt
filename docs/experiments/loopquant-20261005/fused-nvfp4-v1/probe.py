"""Finite byte-exact fixed-global NVFP4 producer and native GEMM qualification."""

import argparse
import fcntl
import hashlib
import itertools
import json
import os
from pathlib import Path
import time

import torch

from tests.loopquant_nvfp4_checks import check_fused_nvfp4

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--gpu-index", type=int, choices=[0, 1], required=True)
parser.add_argument("--locks", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
records = []
with (args.locks / f"gpu{args.gpu_index}-perf.lock").open("a") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    started = time.time()
    torch.backends.cuda.matmul.allow_tf32 = False
    for shape, rows, norm, fixed in itertools.product(
        [(64, 32), (2048, 2048), (2048, 5632), (5632, 2048)],
        [1, 17, 128, 129],
        [False, True],
        [False, True],
    ):
        print(
            json.dumps(
                dict(
                    event="case_start",
                    shape=shape,
                    rows=rows,
                    norm=norm,
                    fixed_blocks=fixed,
                )
            ),
            flush=True,
        )
        records.append(check_fused_nvfp4(rows, *shape, norm, fixed))
        print(json.dumps(records[-1]), flush=True)
    assert len(records) == 64
    torch.cuda.synchronize()
    result = dict(
        status="pass",
        scope="64 fixed-global fused producer/native GEMM cases and192 producer Graph replays; no full model, quality or throughput claim",
        records=records,
        pid=os.getpid(),
        start_unix=started,
        end_unix=time.time(),
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        torch_version=torch.__version__,
        cuda_version=torch.version.cuda,
    )
    with (args.locks / "heavy-io.lock").open("a") as io_lock:
        fcntl.flock(io_lock, fcntl.LOCK_EX)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
