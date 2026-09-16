"""Official FlashAttention paged kernels; no FlashInfer or KV gathering.

Every query row is a length-one sequence with its own depth-specific page
mapping and visible prefix length, including rows from chunked prefill.
"""

import importlib
import logging
from importlib.metadata import version

import torch

FLASH_BACKENDS = ("flash_attn", "flash_attn_2", "flash_attn_3", "flash_attn_4")


def select_version(capability, backend):
    major, minor = capability
    automatic = 4 if major in (10, 12) else 3 if major == 9 else 2
    selected = automatic if backend == "flash_attn" else int(backend.rsplit("_", 1)[1])
    supported = {2: major in (8, 9), 3: major == 9, 4: major in (9, 10, 12)}
    if not supported[selected]:
        raise ValueError(f"FlashAttention-{selected} is not supported on SM{major}{minor}")
    return selected


class FlashPagedAttention:
    def __init__(self, device, dtype, head_dim, block_size, backend="flash_attn"):
        if device.type != "cuda" or torch.version.hip:
            raise ValueError("flash_attn currently requires an NVIDIA CUDA device")
        if dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("flash_attn requires float16 or bfloat16")
        if head_dim > 256 or head_dim % 8:
            raise ValueError("flash_attn requires head_dim divisible by 8 and <= 256")
        self.generation = select_version(torch.cuda.get_device_capability(device), backend)
        if self.generation == 2 and block_size % 256:
            raise ValueError("FlashAttention-2 paged KV requires --block-size a multiple of 256")
        module, package, function = {
            2: ("flash_attn", "flash-attn", "flash_attn_with_kvcache"),
            3: ("flash_attn_3.flash_attn_interface", "flash-attn-3", "flash_attn_with_kvcache"),
            4: ("flash_attn.cute", "flash-attn-4", "flash_attn_varlen_func"),
        }[self.generation]
        try:
            self.kernel = getattr(importlib.import_module(module), function)
            package_version = version(package)
        except (ImportError, AttributeError) as error:
            raise ImportError(
                f"{backend} selected FA{self.generation}; install the official {package} package. "
                "See docs/flash_attention.md. No backend fallback was performed."
            ) from error
        self.info = dict(
            backend="flash_attn",
            generation=self.generation,
            package=package,
            version=package_version,
        )
        logging.getLogger(__name__).warning("Attention implementation: %s", self.info)

    def __call__(self, q, key_cache, value_cache, block_tables, context_lengths):
        if q.shape[0] == 0:
            return torch.empty_like(q)
        # The visible prefix already includes exactly the current causal position.
        # Do not append K/V here: the cache manager has written it at this depth.
        query = q.unsqueeze(1)
        if self.generation == 4:
            out = self.kernel(
                query,
                key_cache,
                value_cache,
                seqused_k=context_lengths,
                page_table=block_tables,
                max_seqlen_q=1,
                max_seqlen_k=block_tables.shape[1] * key_cache.shape[1],
                causal=True,
                num_splits=0,
            )
        else:
            page_argument = "block_table" if self.generation == 2 else "page_table"
            out = self.kernel(
                query,
                key_cache,
                value_cache,
                cache_seqlens=context_lengths,
                causal=True,
                **{page_argument: block_tables},
            )
        if isinstance(out, tuple):
            out = out[0]  # FA4 beta returns (output, optional LSE).
        return out.squeeze(1)
