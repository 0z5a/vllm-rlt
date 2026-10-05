"""Run the frozen native FP8 family checks on H20 without a test-runner dependency."""

import argparse
import itertools
import json
import os
from pathlib import Path
import time

import torch

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--family", choices=["hrm", "loopformer", "huginn", "parcae"], required=True
)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
if args.family == "hrm":
    from tests.loopquant_hrm_fp8_checks import check_hrm_fp8 as check
elif args.family == "loopformer":
    from tests.loopquant_loopformer_fp8_checks import check_loopformer_fp8 as check
elif args.family == "huginn":
    from tests.loopquant_huginn_fp8_checks import check_huginn_fp8 as check
else:
    from tests.loopquant_parcae_fp8_checks import check_parcae_fp8 as check

assert torch.cuda.get_device_capability() == (9, 0)
torch.backends.cuda.matmul.allow_tf32 = False
started = time.time()
records = []
for index, (policy, graphs, option) in enumerate(
    itertools.product(["DYN", "SH1", "ST2", "LOOP"], [False, True], [False, True])
):
    folder = (
        args.output.parent
        / "h20-native-families-v1-exports"
        / args.family
        / f"case-{index:02d}"
    )
    row = check(folder, policy, graphs, option)
    torch.cuda.synchronize()
    row["export_manifest"] = json.loads((folder / "manifest.json").read_text())
    (folder / "tensors.pt").unlink()
    records.append(
        dict(kind="native_model", temporary_weight_payload_removed=True, **row)
    )
    print(
        json.dumps(
            dict(
                event="case_pass",
                family=args.family,
                index=index,
                policy=policy,
                graphs=graphs,
                option=option,
            )
        ),
        flush=True,
    )
if args.family == "huginn":
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
            scope="tiny native FP8 correctness and encoded reload; no full-checkpoint quality or performance",
        ),
        indent=2,
    )
    + "\n"
)
