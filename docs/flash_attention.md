# FlashAttention backends

`flash_attn` chooses FA2 on SM8, FA3 on SM9 (Hopper), and FA4 on SM10.
Use `flash_attn_2`, `flash_attn_3` or `flash_attn_4` to select a generation
explicitly. Missing packages raise an error; an explicit selection does not
silently fall back to Triton. `auto` is a separate policy that checks compatible
installed implementations. Triton remains available without FlashAttention.

## Install FA3 on Hopper

FA3 is the official Dao-AILab implementation in the `hopper` directory. Its
package is `flash-attn-3`, and the project imports
`flash_attn_3.flash_attn_interface.flash_attn_with_kvcache`. The existing
`.[flash-attn]` extra installs FA2 and does not provide this module.
The PyPI `flash-attn-3` 0.0.0 placeholder is an empty, withdrawn release; use
[the official source installation](https://github.com/Dao-AILab/flash-attention#flashattention-3-beta-release).

The official requirements include Hopper and CUDA toolkit >=12.3, with CUDA 12.8
recommended. Install the CUDA-enabled PyTorch required by the project first.
Build against that PyTorch and a compatible CUDA toolkit, not an arbitrary
prebuilt wheel from a different PyTorch/CUDA ABI. A working CUDA runtime alone
does not provide `nvcc`; locate the toolkit separately. No driver upgrade is
needed when the existing driver already runs the selected PyTorch CUDA build.

Use Linux and activate a dedicated virtual environment before installing. The requirements
file pins the official source commit, including its NVIDIA CUTLASS submodule:

```bash
# Start from an installed vllm-rlt environment with CUDA-enabled PyTorch.
python -c 'import torch; print(torch.__version__, torch.version.cuda)'
# Point CUDA_HOME at a compatible toolkit; do not change a shared toolkit.
"$CUDA_HOME/bin/nvcc" --version
uv pip install --index-url https://pypi.org/simple ninja einops wheel packaging
MAX_JOBS=4 NVCC_THREADS=1 uv pip install --no-build-isolation --no-deps \
  -r requirements/flash-attn3.txt
python -c 'from importlib.metadata import version; from flash_attn_3 import flash_attn_interface; print(version("flash-attn-3"), flash_attn_interface.__file__)'
```

`--no-build-isolation` uses the already installed PyTorch headers/ABI;
`--no-deps` keeps its version intact. The prerequisite command supplies the
remaining build/runtime packages. Set `MAX_JOBS` to the CPU and memory resources
reserved for compilation. Build on a scheduler CPU allocation when required by
the cluster; compiling FA3 does not require allocating a GPU. Official source
builds may take substantial time. The pinned source reports FA3 version 3.0.0.

An existing CUDA 13 toolkit with a PyTorch cu128 build is not a reason to upgrade
the driver or shared packages. Use a task-local CUDA 12.8 toolkit instead. NVIDIA
provides [official redistributable components](https://developer.download.nvidia.com/compute/cuda/redist/redistrib_12.8.1.json)
with SHA256 checksums for `cuda_nvcc`, `cuda_cudart` and `cuda_cccl`. Retain the
component versions/checksums and build flags with the resulting wheel.

## Run GPU regression

Use a scheduler-reserved device. Confirm that the selected implementation is
FA3 before testing; importing the Python module alone does not execute a kernel:

```bash
python - <<'PY'
import torch
from vllm_rlt.kernels.flash_attention import FlashPagedAttention
assert torch.cuda.is_available()
attention = FlashPagedAttention(torch.device('cuda'), torch.bfloat16, 64, 16)
print(torch.cuda.get_device_name(), torch.cuda.get_device_capability(), attention.info)
assert attention.generation == 3
PY
python -m pytest tests/test_cuda_graph.py --run-gpu -k flash_attn -q
python -m pytest tests/test_flash_attention.py --run-gpu \
  -k 'ragged_strided_paged_attention or flash_sync_async_mixed_depths' -q
python -m pytest tests/test_speculative.py tests/test_async_pipeline.py \
  tests/test_async_state.py tests/test_serving.py tests/test_cuda_graph.py \
  --run-gpu -k 'not flash_attn_4' -q
```

The first pytest command runs all 12 FA3 graph cases: eager/graph equivalence,
request-ID reuse, both KV layouts, static/dynamic buffers, and sync/async/multiple
stream scheduling. The second exercises eight FP16/BF16 paged/GQA and mixed-depth
cases. The final command includes the FA3 graph cases and the previous Triton,
serving and state/lifetime regression; only explicit FA4 cases are deselected.
Do not use `-k 'not flash_attn'` to qualify FA3.

## H20 validation (2026-10-10)

Runtime/test source was PR66 commit `0464988766f2f07001b47d63a71d516b76b85a3e`.
Validation used one reserved H20-3e (SM90), Python 3.10.20,
PyTorch 2.11.0+cu128, Triton 3.6.0 and the task-local FA3 native extension
`3.0.0+pr66.94e22c9`. The build used official source `94e22c9`, NVIDIA
CUTLASS `7127592069c2fe01b041e174ba4345ef9b279671` and hash-verified NVIDIA
CUDA nvcc 12.8.93, cudart 12.8.90 and CCCL 12.8.90 components. Shared PyTorch
packages and the driver were not upgraded; FA3 remains absent from the shared
environment.

All original **12 graph node IDs passed, with zero skips**, and their XML names
were checked against the preserved earlier failure list. The previous related
suite passed **160 tests, with three explicit FA4 cases deselected**. Additional
FA3 paged/GQA and mixed-depth tests passed **eight tests**; 14 unrelated cases
were deselected by that focused selector. Async Triton tests passed **26 tests**
(22 GPU cases and four CPU helpers). No FA3 cases were deselected from the related
suite. These run real FA3 kernels on FP16/BF16 tensors and tiny model fixtures;
they do not measure real-checkpoint FA3 accuracy, throughput or production load.

The bounded inference build retains paged KV, SplitKV, GQA, varlen, FP16/BF16
and all equal Q/K/V head widths. It disables backward, FP8, SM80, softcap and
unequal head widths; those features are not qualified. The GPU fixtures cover
head widths 64 and 128. To reproduce this inference build, set these official flags before
the pinned install command above (use the CPU count reserved by your scheduler):

```bash
export FLASH_ATTENTION_FORCE_BUILD=TRUE
export FLASH_ATTENTION_DISABLE_BACKWARD=TRUE FLASH_ATTENTION_DISABLE_FP8=TRUE
export FLASH_ATTENTION_DISABLE_SM80=TRUE FLASH_ATTENTION_DISABLE_SOFTCAP=TRUE
export FLASH_ATTENTION_DISABLE_HDIMDIFF64=TRUE FLASH_ATTENTION_DISABLE_HDIMDIFF192=TRUE
MAX_JOBS=4 NVCC_THREADS=1 uv pip install --no-build-isolation --no-deps \
  -r requirements/flash-attn3.txt
```

The task added the metadata tag `FLASH_ATTN_LOCAL_VERSION=pr66.94e22c9` to record
its source revision. Without that tag the same source reports 3.0.0. The general
source installation uses upstream defaults unless flags are set explicitly.

Run post-install checks from the project directory rather than from `hopper`:
its source compatibility package can shadow the installed native extension.
On a filesystem that cannot copy Python metadata/xattrs, place the uv build cache
on a local scratch filesystem using `UV_CACHE_DIR`; this affects temporary build
files and does not require reinstalling shared packages or changing permissions.

The earlier 12 failures were from manual H20 regression, not GitHub CI. This
repository has no GitHub workflow configuration; the inspected PR head had no
workflow runs/checks/status contexts. The pinned dependency and commands provide
a reproducible runner setup without creating a hosted GPU CI service or changing
permissions.

## Other generations

FA2 uses `uv pip install -e '.[flash-attn]'` and requires paged KV block sizes
that are multiples of 256. FA4 uses `uv pip install -e '.[flash-attn4]'`; its
pinned release is 4.0.0b30. The adapter supports paged FA4 on SM9/SM10 and rejects
SM12 before importing or allocating KV, because that pinned release does not
support paged KV there. FA3 installation and tests do not qualify FA2 or FA4.
