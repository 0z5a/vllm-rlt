"""Publish one executed record per row before its layer writes, on the same stream."""

import triton
import triton.language as tl


@triton.jit
def _publish(POINTERS, WIDTHS, DEPTHS, POSITIONS, RECORDS, N: tl.constexpr, BLOCK: tl.constexpr):
    rows = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = rows < N
    pointers = tl.load(POINTERS + rows, valid, other=0).to(tl.pointer_type(tl.int32))
    width = tl.load(WIDTHS + rows, valid, other=0).to(tl.int64)
    depth = tl.load(DEPTHS + rows, valid, other=0)
    position = tl.load(POSITIONS + rows, valid, other=0)
    record = tl.load(RECORDS + rows, valid, other=-1).to(tl.int32)
    tl.store(pointers + depth * width + position, record, valid)


def publish_records(pointers, widths, depths, positions, records):
    _publish[(triton.cdiv(pointers.numel(), 128),)](
        pointers, widths, depths, positions, records, pointers.numel(), 128
    )
