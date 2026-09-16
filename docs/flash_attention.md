# Official FlashAttention backend

Select `--attention-backend FLASH_ATTN` (case-insensitive in the CLI/server), or
`attention_backend="flash_attn"` in Python. This directly calls the official
FlashAttention package; it does not use FlashInfer or a TensorRT engine.
CUDA graphs are not implemented or enabled by this change.

## Hardware selection and installation

| Hardware | Automatic implementation | Package | Validation here |
| --- | --- | --- | --- |
| Ampere/Ada, SM 8.x | FA2 | `flash-attn>=2.8.3` | Selection logic only; hardware unavailable |
| Hopper, SM 9.x | FA3 | Official `flash-attn-3` from `hopper/` | Selection logic only; hardware unavailable |
| Blackwell, SM 10.x/12.x | FA4 CuTe DSL | `flash-attn-4==4.0.0b30` | Real B300 SM103 GPU tests |

Explicit `flash_attn_2`, `flash_attn_3`, and `flash_attn_4` choices are also
available. Unsupported architecture/dtype and missing packages fail explicitly;
there is no silent fallback to another generation or to Triton. Both FP16 and
BF16 are supported; FP32 is rejected. Head dimension must be divisible by 8
and at most 256, with additional restrictions enforced by the upstream kernel.
This adapter currently targets NVIDIA CUDA, not ROCm.

FA2's paged-cache interface requires a page size divisible by **256**. On
Ampere/Ada set `--block-size 256`; the default 16 is deliberately not silently
changed, because page size affects KV capacity and allocation. FA4 on this B300
was validated with 16-token pages. Other configurations need device validation.

The B300 environment uses Python 3.12, PyTorch 2.13.0+cu130 and:

```bash
uv pip install --python /home/zjy/code/david/b_workspace/.b_rdma/bin/python \
  'flash-attn-4[cu13]==4.0.0b30'
```

Installation upgraded `apache-tvm-ffi` from 0.1.11 to 0.1.13.post3; PyTorch was
not changed. FA4 is an upstream beta release. For Ampere/Ada install the project's
`flash-attn` extra with build isolation disabled. For Hopper follow the official
FA3 installation instructions; its package exposes
`flash_attn_3.flash_attn_interface`. FA4 and FA2 extras are hardware-specific,
not mandatory dependencies for CPU/Triton users.

Upstream references:
[installation and supported architectures](https://github.com/Dao-AILab/flash-attention),
[FA3 paged interface](https://github.com/Dao-AILab/flash-attention/blob/main/hopper/flash_attn_interface.py),
[FA4 interface](https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/cute/interface.py).

## KV semantics and asynchronous execution

Implementation: [flash_attention.py](../vllm_lt/kernels/flash_attention.py).
The cache manager still writes the current token's KV and selects the physical
page table for each query's recurrence depth. The adapter passes existing K/V
layer views directly, without gathering or copying the full cache into a
contiguous tensor. Each query is represented as a length-one sequence; its
visible KV length is its position plus one. This preserves causality even when
multiple chunked-prefill rows share a page table. Prefix lengths remain on GPU.

FA2 receives `block_table` and `cache_seqlens`; FA3 receives `page_table` and
`cache_seqlens`; FA4 receives `page_table` and `seqused_k`. FA4's maximum KV
length is bounded by the supplied table width rather than the entire physical
pool size, and `num_splits=0` enables upstream split selection. Zero-length
padding rows produce zero attention output. No second K/V append is requested.

Kernels run on the caller's current CUDA stream, preserving the existing
S/AS/AM dependencies and buffer lifetime rules. Import and architecture
selection happen at cache construction, including automatic memory profiling,
rather than inside each recurrent layer. Startup logs record generation and
installed package version; `cache_manager.attention_info` exposes these fields.

## Start and request

Example on an available B300 GPU (GPU 4 was used for validation):

```bash
cd /home/zjy/code/david/b_workspace/vllm-lt
CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  ../.b_rdma/bin/python -m vllm_lt.entrypoints.serve \
  --model ../models/Ouro-1.4B --served-model-name ouro \
  --device cuda --dtype bfloat16 --attention-backend FLASH_ATTN \
  --max-num-seqs 8 --max-num-batched-tokens 2048 --prefill-chunk-size 2048 \
  --exit-mode ouro_delayed --host 127.0.0.1 --port 8000 \
  > /home/zjy/code/david/tmp/ouro-flash-attn-20260915/server.log 2>&1
```

This command is synchronous. Add `--async-scheduling --single-stream` for AS,
or `--async-scheduling` for AM. Static buffers and padding remain off unless
explicitly requested. `ouro_delayed` reuses the published early-exit gate; it is
not a distilled lookahead predictor. Changing attention does not change that.

```bash
curl -sS http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"ouro","prompt":"Explain matrix multiplication briefly.","max_tokens":32,"temperature":0}'
```

For the decode context benchmark, add `--attention-backend flash_attn` to
`python -m benchmarks.context_sweep`. It also accepts `--block-size` and records
the resolved generation/package/version in `manifest.json`. Existing benchmark
commands retain their original Triton default so historical runs are identifiable.

## Validation on B300

Logs: `/home/zjy/code/david/tmp/ouro-flash-attn-20260915/`.

- Full regression: **272 passed, 11 optional-dependency skips** (`full-tests.log`).
- New tests: FP16/BF16 MHA/GQA paged output against the Torch reference;
  non-contiguous per-layer KV views; nonsequential physical page mappings;
  ragged lengths including zero; mixed exit traces with both KV layouts and
  S/AS/AM, with static/padding off and on (`tests/test_flash_attention.py`).
- Real Ouro BF16: 128-token prompt, concurrency 3, 16 generated tokens;
  fixed4 and controlled mixed234 exits, S/AS/AM, dynamic and static+padding.
  **All 12 measured configurations agreed on output IDs and exit depths**
  within each policy (`real-smoke/results.json`). This is a smoke test, not
  a statistically repeated performance baseline or a quality evaluation.
- Real-model CLI single request with automatic KV capacity and uppercase
  `FLASH_ATTN`: completed successfully (`cli-auto-kv.log`).

The CDB paper's Appendix A uses H100 and paged FA3. B300 results from this
adapter must be labeled FA4, not a reproduction of that exact hardware/kernel
configuration. A larger performance sweep is still required to quantify gains.

Paired Triton/FA4 decode A/B protocol and results: [attention A/B](flash_attention_ab.md).
