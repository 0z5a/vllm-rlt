# Fixed-global NVFP4 fused producer preflight

The new producer combines optional protected RMSNorm, block16 scale calculation,
E2M1 packing and direct SWIZZLE_32_4_4 scale writes in one launch. Its global FP32
scale is fixed; local E4M3 block scales are either dynamically computed or supplied
as one shared vector. All scratch and fixed scale tensors are allocated during
setup. GEMM remains the installed Torch native block-scaled operation.

This is the limited F2 kernel preflight allowed alongside F1, not a full training
grid or model deployment claim. The earlier60-case unfused NVFP4 ABI check passed;
that result does not validate this new fused producer.

The frozen GPU qualification contains64 cases: four K/N geometries, rows1/17/128/129,
optional RMSNorm and dynamic/fixed block scales. It requires exact packed bytes,
all padded scale slots, native GEMM against decoded operands at the unchanged
combined atol/rtol0.02, plus192 producer Graph replays with stable scratch pointers.
The cases include zero rows, outliers and reuse after input changes. The producer
supports K divisible by64. Mixed-depth scale tables and model export are not yet
implemented for this format.

| Check | Result |
|---|---|
| Existing reference E2M1 rounding/swizzle tests | 3 passed |
| Complete CPU regression | 494 passed,29 skipped,264 GPU cases deselected |
| New64 fused GPU cases /192 Graph replays | Not run |
| Full model native export/quality | Not run |
| E2E throughput/speedup | Not measured |

Source `54d34826da92d6cf7e21fc1a69c246f378c12ce5`; frozen packet manifest
`74ff732c92efec9f469cf04221a77eb735fec4dbd43c59bc7effa1e69b11228f`.
The packet is local and awaits admission. The CPU suite does not compile or
execute the new CUDA kernel. [Full CPU receipt](full-cpu-fused-nvfp4-attempt1.xml).
