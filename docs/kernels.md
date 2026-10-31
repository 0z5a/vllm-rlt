# Native kernel development

Native kernels are optional and live alongside the existing Triton kernels.
Importing `vllm_rlt.kernels` loads the built library once and registers the Torch
fallbacks. It neither compiles code nor creates a CUDA context.

## Layout

| Location | Responsibility |
| --- | --- |
| `vllm_rlt/kernels/activation.py` | Python activation entry and Torch reference |
| `vllm_rlt/kernels/_extension.py` | One-time loading and missing-library detection |
| `csrc/ops.h` | Native operator declarations |
| `csrc/torch_bindings.cpp` | Operator schemas and CUDA dispatch registration |
| `csrc/activation/silu_and_mul.cu` | SiLU-and-Mul CUDA validation, launch and computation |
| `setup.py` | Optional build of `vllm_rlt._C` |
| `tests/test_activation.py` | Reference, CUDA numerics, layouts, tails and graph replay |
| `benchmarks/kernels/silu_and_mul.py` | Matched Torch/native/vLLM accuracy and latency comparison |

Add future kernels to the same extension and group sources by operation family.
No backend registry or base class is required. Put future kernel performance
benchmarks under `benchmarks/kernels/`.

## SiLU-and-Mul contract and current status

`silu_and_mul_reference(gate, up)` computes `F.silu(gate) * up`. The inputs have
identical shape, dtype and device; broadcasting is rejected. Supported dtypes
are FP32, FP16 and BF16. The output owns new storage and preserves input shape,
dtype and device. Neither input is modified. The Torch reference preserves
the intermediate SiLU rounding of the existing MLP expression.

Separate inputs avoid concatenating Ouro and Nanbeige's separate projection
outputs. The reference also accepts Huginn's noncontiguous split-projection
views. The native entry requires both inputs to be contiguous; it neither pads
nor copies incompatible inputs.

`silu_and_mul(gate, up)` uses PyTorch dispatch: CUDA inputs use the native kernel
when the extension is installed; CPU inputs and installations without the
extension use the Torch reference. Import it from `vllm_rlt.kernels`. The model
call site does not check devices or extension availability. The native kernel
flattens contiguous inputs, using 16-byte
loads/stores for the aligned vector body and scalar accesses for the remaining
elements in the same kernel. Contiguous views with misaligned storage offsets
use scalar accesses. Empty inputs return an empty output without launching.

SiLU uses FP32 arithmetic and rounds to the input dtype before multiplying by
`up`, preserving the reference's intermediate rounding. FP16/BF16 vector paths
use `__hmul2`; FP32 uses separate round-to-nearest products. On SM103, forcing
`__fmul2_rn` increased the 16-byte vector kernel's registers/thread from 28 to
34 and reduced resident warps, hurting memory throughput. The scalar path uses
`__hmul` for FP16/BF16 and `__fmul_rn` for FP32, retaining the SiLU rounding
boundary without explicitly converting the rounded activation back to FP32.
The device guard follows the input device, and launches use that device's current
stream.

Blocks have 256 threads. The grid is capped at 4096 blocks, except for FP32
workloads with at least `16384 * 256` work items: these use up to 16384 blocks.
Each aligned work item processes four FP32 elements; an unaligned work item
processes one scalar. The threshold requires enough work to fill the larger
grid. This keeps small and medium shapes' launch geometry unchanged. Grid sizes,
16/32-byte transfers and scalar/packed FP32 products were compared on B300;
the selected configuration is not a universal hardware optimum.

Ouro's inference MLP calls this registered operator directly. Availability is
resolved once during module initialization; extension ABI or dependency errors
propagate rather than silently selecting another path. The native CUDA dispatch
reuses its C++ input checks; a missing extension does not mask native execution
errors when an installed kernel is selected.
Nanbeige and Huginn do not use this operator yet.

Autograd is unsupported: gradient-enabled inputs are rejected unless gradients
are disabled, consistent with Ouro's inference-only model. `torch.compile` has
no fake kernel registration. Tests cover the operator and Ouro MLP, including
CUDA Graph capture/replay.

## Build and reference checks

Ordinary `pip install .` remains Python-only. To build the optional extension,
use an environment with a compatible CUDA PyTorch and the CUDA toolkit (including
`nvcc`) already installed. A driver alone is insufficient. Run these commands
with the Python interpreter used for inference:

```bash
python -m pip install setuptools wheel ninja
VLLM_RLT_BUILD_CUDA=1 python -m pip install --no-build-isolation -e .
python -m pytest tests/test_activation.py -q
CUDA_VISIBLE_DEVICES=<allocated_gpu> python -m pytest \
  tests/test_activation.py --run-gpu -q
```

`pip` invokes setuptools, whose PyTorch `CUDAExtension`/`BuildExtension` compiles
the C++ bindings and CUDA sources with the host compiler and `nvcc`, then installs
`vllm_rlt._C`. `--no-build-isolation` uses the environment's existing PyTorch
headers and libraries; it avoids a separate build environment with a potentially
different Torch version. No runtime JIT compilation or CMake is required.

For example, to build for Ampere SM80 and B300 SM103 without a visible GPU:

```bash
TORCH_CUDA_ARCH_LIST="8.0;10.3" VLLM_RLT_BUILD_CUDA=1 MAX_JOBS=2 \
  python -m pip wheel --no-build-isolation --no-deps . -w dist
python -m pip install dist/vllm_rlt-*.whl
```

Choose architectures supported by the target hardware and toolkit; the example
wheel contains SM80 and SM103 code. A binary without code compatible with the
device raises an error; there is no fallback after a native launch failure. A build without
`VLLM_RLT_BUILD_CUDA=1` produces a Python-only distribution. GPU kernel tests and
the benchmark require the native extension and fail if it is absent.

Restart Python after rebuilding the library. A successful build or passing CPU
reference tests establishes no CUDA compute correctness or speedup.

## Accuracy and performance comparison

With vLLM installed in the same environment:

```bash
CUDA_VISIBLE_DEVICES=<allocated_gpu> .venv/bin/python -m benchmarks.kernels.silu_and_mul \
  --output artifacts/silu-and-mul/results.json \
  --profile-dir artifacts/silu-and-mul/profiles
```

The benchmark checks matched random inputs against eager Torch and a staged
FP64 oracle that rounds SiLU before multiplication. It retains maximum absolute
and ULP errors, RMSE, mismatch fractions, and nonfinite mismatches. The default
matrix covers FP32/FP16/BF16, 1/16/128/1024/4096 rows, and widths 5632/5633.

Latency is measured without profiling, with three repetitions and shuffled
backend order. CUDA Graph timing amortizes replay over 32 calls; eager timing
includes Python, allocation, launch and device completion. `vllm_prepacked`
excludes concatenation and preallocates its output; `vllm_with_cat` includes
concatenation and output allocation from the same separate inputs. These are
distinct comparisons. None establishes full-model throughput or task quality.

For a source-built standalone vLLM baseline, pass `--vllm-library <so>` and
`--vllm-namespace <namespace>`, plus `--vllm-source <checkout>` to record its
commit. The report retains library hashes, the local diff and raw measurements.
Optional ops-only/full profiles are captured separately after timing; full
capture enables shapes, memory and stacks. Large traces stay under ignored
`artifacts/`.
