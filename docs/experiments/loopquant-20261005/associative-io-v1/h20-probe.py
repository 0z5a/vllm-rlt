"""Finite fused-producer/rowwise-GEMM qualification, not a throughput benchmark."""

import argparse
import json
import os
from pathlib import Path
import time

import torch

from loopquant.fused_fp8 import FP8InputQuantizer, FP8Workspace, RowwiseFP8Linear
from loopquant.quantizers import ScaleLayout, fp8_encode
from vllm_rlt.layers import RMSNorm


def reference_input(values, producer, gain, loops):
    if gain is not None:
        x = values.float()
        values = gain * (x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)).to(
            values.dtype
        )
    padded = torch.nn.functional.pad(values, (0, 0, 0, (-values.shape[0]) % 16))
    if producer.mode == "dynamic":
        scales = padded.float().abs().amax(-1, keepdim=True).clamp_min(1e-12) / 448
    else:
        ids = torch.nn.functional.pad(loops, (0, (-values.shape[0]) % 16))
        scales = producer.scales[producer.layout.indices(ids)].unsqueeze(-1)
    return fp8_encode(padded, scales), scales


def reference_gemm(packed, scales, linear, rows):
    return (
        (packed.float() @ linear.packed_weight.float()) * scales * linear.weight_scales
    )[:rows].to(torch.bfloat16)


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
assert torch.cuda.get_device_capability() == (9, 0), "this packet targets H20 SM90"
started = time.time()
torch.set_grad_enabled(False)
torch.manual_seed(17)
torch.backends.cuda.matmul.allow_tf32 = False
records = []
worst_error = -1.0
worst_case = {}
for columns, outputs in [(64, 32), (2048, 2048), (2048, 5632), (5632, 2048)]:
    original = torch.nn.Linear(
        columns, outputs, bias=False, device="cuda", dtype=torch.bfloat16
    )
    linear = RowwiseFP8Linear(original)
    pointer = linear.packed_weight.data_ptr()
    norm = RMSNorm(columns).to(device="cuda", dtype=torch.bfloat16)
    norm.weight.data.uniform_(0.75, 1.25)
    for rows in [1, 17, 128]:
        values = torch.randn(rows, columns, device="cuda", dtype=torch.bfloat16)
        if rows > 1:
            values[0].zero_()
        values[-1, 0], values[-1, 1] = 1000, -1000
        loops = torch.arange(rows, device="cuda") % 4
        scratch = FP8Workspace.allocate(rows, columns, torch.device("cuda"))
        for label, mode, layout, scales in [
            ("DYN", "dynamic", ScaleLayout(4), [1.0]),
            ("SH1", "static", ScaleLayout(4), [0.01]),
            ("ST2", "static", ScaleLayout(4, (2,)), [0.01, 0.02]),
        ]:
            producer = FP8InputQuantizer(
                torch.tensor(scales, device="cuda"), layout, mode
            )
            for fused_norm in [False, True]:
                gain = norm.weight if fused_norm else None
                scratch.packed.fill_(float("nan"))
                scratch.row_scales.fill_(float("nan"))
                producer(values, scratch, norm_weight=gain, loop_ids=loops)
                expected_packed, expected_scales = reference_input(
                    values, producer, gain, loops
                )
                case = dict(
                    inputs=values.detach().clone(),
                    norm_gain=gain.detach().clone() if gain is not None else None,
                    loop_ids=loops.clone(),
                    static_scales=producer.scales.clone(),
                    packed_weight=linear.packed_weight.clone(),
                    weight_scales=linear.weight_scales.clone(),
                    packed_input=scratch.packed.clone(),
                    row_scales=scratch.row_scales.clone(),
                    expected_packed=expected_packed,
                    expected_scales=expected_scales,
                )
                if not fused_norm and (
                    not torch.equal(scratch.packed.float(), expected_packed.float())
                    or not torch.equal(scratch.row_scales, expected_scales)
                ):
                    torch.save(case, args.output.with_suffix(".failed-case.pt"))
                if not fused_norm:
                    torch.testing.assert_close(
                        scratch.packed.float(),
                        expected_packed.float(),
                        atol=0,
                        rtol=0,
                    )
                    torch.testing.assert_close(
                        scratch.row_scales, expected_scales, atol=0, rtol=0
                    )
                assert torch.count_nonzero(scratch.packed[rows:].float()) == 0
                assert torch.isfinite(scratch.row_scales).all()
                actual = linear(scratch, rows)
                encoded_reference = reference_gemm(
                    scratch.packed, scratch.row_scales, linear, rows
                )
                independent = reference_gemm(
                    expected_packed, expected_scales, linear, rows
                )
                case.update(
                    actual=actual,
                    reference=independent,
                    encoded_reference=encoded_reference,
                )
                error = float((actual.float() - independent.float()).abs().max())
                if error > worst_error:
                    worst_error, worst_case = error, case
                if not torch.allclose(
                    actual, independent, atol=0.02, rtol=0.02
                ) or not torch.allclose(
                    actual, encoded_reference, atol=0.02, rtol=0.02
                ):
                    torch.save(case, args.output.with_suffix(".failed-case.pt"))
                torch.testing.assert_close(
                    actual, encoded_reference, atol=0.02, rtol=0.02
                )
                torch.testing.assert_close(actual, independent, atol=0.02, rtol=0.02)
                assert linear.packed_weight.data_ptr() == pointer
                record = dict(
                    rows=rows,
                    columns=columns,
                    outputs=outputs,
                    policy=label,
                    fused_norm=fused_norm,
                    packed_equal=bool(
                        torch.equal(scratch.packed.float(), expected_packed.float())
                    ),
                    max_abs=float((actual.float() - independent.float()).abs().max()),
                )
                records.append(record)
                print(json.dumps(record), flush=True)
torch.save(worst_case, args.output.with_suffix(".worst-case.pt"))
# Stable scratch/scale/loop-ID addresses must observe updates after graph capture.
rows, columns = 17, 64
values = torch.randn(rows, columns, device="cuda", dtype=torch.bfloat16)
loops = torch.arange(rows, device="cuda") % 4
producer = FP8InputQuantizer(
    torch.tensor([0.01, 0.02], device="cuda"), ScaleLayout(4, (2,)), "static"
)
linear = RowwiseFP8Linear(
    torch.nn.Linear(columns, 32, bias=False, device="cuda", dtype=torch.bfloat16)
)
scratch = FP8Workspace.allocate(rows, columns, torch.device("cuda"))
stream = torch.cuda.Stream()
stream.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(stream):
    for _ in range(3):
        producer(values, scratch, loop_ids=loops)
        linear(scratch, rows)
torch.cuda.current_stream().wait_stream(stream)
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph, stream=stream):
    producer(values, scratch, loop_ids=loops)
    captured = linear(scratch, rows)
identities = [
    tensor.data_ptr()
    for tensor in [
        values,
        loops,
        producer.scales,
        scratch.packed,
        scratch.row_scales,
        linear.packed_weight,
    ]
]
for factor in [1.0, 1.5, 0.5]:
    values.mul_(factor)
    loops.copy_(3 - loops)
    producer.scales.mul_(factor)
    graph.replay()
    packed, scales = reference_input(values, producer, None, loops)
    torch.testing.assert_close(
        captured, reference_gemm(packed, scales, linear, rows), atol=0.02, rtol=0.02
    )
assert identities == [
    tensor.data_ptr()
    for tensor in [
        values,
        loops,
        producer.scales,
        scratch.packed,
        scratch.row_scales,
        linear.packed_weight,
    ]
]
torch.cuda.synchronize()
args.output.write_text(
    json.dumps(
        dict(
            status="pass",
            scope="fused FP8 correctness; no performance measurement",
            records=records,
            graph_replays=3,
            pid=os.getpid(),
            start_unix=started,
            end_unix=time.time(),
            torch_version=torch.__version__,
            cuda_version=torch.version.cuda,
            device=torch.cuda.get_device_name(),
        ),
        indent=2,
    )
    + "\n"
)
