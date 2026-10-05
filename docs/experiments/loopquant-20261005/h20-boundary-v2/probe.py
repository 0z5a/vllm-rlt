"""Finite rerun of existing FP8/INT4 checks after boundary prefill row repair."""

import argparse
import itertools
import json
import os
from pathlib import Path
import time

import torch

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--family",
    choices=["huginn_fp8", "parcae_fp8", "huginn_int4", "parcae_int4"],
    required=True,
)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
assert torch.cuda.get_device_capability() == (9, 0)
torch.backends.cuda.matmul.allow_tf32 = False
started = time.time()
records = []
if args.family.endswith("fp8"):
    if args.family == "huginn_fp8":
        from tests.loopquant_huginn_fp8_checks import check_huginn_fp8 as check
    else:
        from tests.loopquant_parcae_fp8_checks import check_parcae_fp8 as check
    cases = list(
        itertools.product(["DYN", "SH1", "ST2", "LOOP"], [False, True], [False, True])
    )
else:
    cases = [
        (graphs, compact)
        for graphs in (False, True)
        for compact in ((False,) if args.family == "huginn_int4" else (False, True))
    ]
for index, case in enumerate(cases):
    folder = (
        args.output.parent
        / "h20-boundary-fix-v2-exports"
        / args.family
        / f"case-{index:02d}"
    )
    if args.family.endswith("fp8"):
        row = check(folder, *case)
    elif args.family == "huginn_int4":
        from tests.loopquant_huginn_int4_checks import check_huginn_int4

        row = check_huginn_int4(folder, case[0])
    else:
        from tests.loopquant_int4_checks import check_int4_native_model

        row = check_int4_native_model("parcae", case[0], folder, compact=case[1])
    torch.cuda.synchronize()
    row["export_manifest"] = json.loads((folder / "manifest.json").read_text())
    (folder / "tensors.pt").unlink()
    records.append(
        dict(kind="native_model", temporary_weight_payload_removed=True, **row)
    )
    print(
        json.dumps(dict(event="case_pass", family=args.family, index=index, case=case)),
        flush=True,
    )
if args.family == "huginn_fp8":
    from tests.loopquant_huginn_fp8_checks import check_graph_loop_metadata

    for asynchronous in (False, True):
        row = check_graph_loop_metadata(asynchronous)
        torch.cuda.synchronize()
        records.append(dict(kind="row_depth_control", **row))
        print(
            json.dumps(dict(event="metadata_pass", asynchronous=asynchronous)),
            flush=True,
        )
args.output.write_text(
    json.dumps(
        dict(
            status="pass",
            family=args.family,
            device=torch.cuda.get_device_name(),
            torch_version=torch.__version__,
            cuda_version=torch.version.cuda,
            pid=os.getpid(),
            start_unix=started,
            end_unix=time.time(),
            records=records,
            scope="tiny native correctness and encoded reload at unchanged criteria; no full-checkpoint quality or performance",
        ),
        indent=2,
    )
    + "\n"
)
