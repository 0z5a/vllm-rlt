"""Ouro projection shapes: BF16 operands and FP32 TMEM accumulation."""

import triton
import triton.language as tl


@triton.jit
def project(A, W, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
            BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    columns = tl.program_id(1) * BN + tl.arange(0, BN)
    inner = tl.arange(0, BK)
    accumulator = tl.zeros((BM, BN), tl.float32)
    for block in range(tl.cdiv(K, BK)):
        kk = block * BK + inner
        activation = tl.load(A + rows[:, None] * K + kk[None, :],
                             (rows[:, None] < M) & (kk[None, :] < K), 0)
        weight = tl.load(W + columns[None, :] * K + kk[:, None],
                         (columns[None, :] < N) & (kk[:, None] < K), 0)
        accumulator = tl.dot(activation, weight, accumulator)
    tl.store(C + rows[:, None] * N + columns[None, :], accumulator,
             (rows[:, None] < M) & (columns[None, :] < N))
