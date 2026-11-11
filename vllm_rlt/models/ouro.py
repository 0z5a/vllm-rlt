# SPDX-License-Identifier: Apache-2.0
# Architecture adapted from ByteDance/Ouro-1.4B modeling_ouro.py, revision
# 574fa66cb8bf5abdc979642d01cf2b79b16bfab1 (Apache-2.0).
# Original architecture includes code Copyright 2024 The Qwen team, Alibaba
# Group and the HuggingFace Inc. team. All rights reserved.
"""Native Ouro execution split into embedding, recurrent core, and LM head.

The scheduler owns loop depth and halting. Each row can belong to a different
request or loop depth; the cache supplies the corresponding causal KV history.
"""

import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch import nn
from torch.nn import functional as F

from vllm_rlt.kernels import silu_and_mul, single_launch_ops
from vllm_rlt.layers import (
    RMSNorm,
    RotaryEmbedding,
    apply_rotary_pos_emb,
    rotate_half,
)

if TYPE_CHECKING:
    from vllm_rlt.core.kv_cache_manager import KVCacheManager
    from vllm_rlt.worker.metadata import DeviceKVMetadata, KVMetadata


@dataclass(frozen=True)
class OuroConfig:
    vocab_size: int = 49152
    hidden_size: int = 2048
    intermediate_size: int = 5632
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    num_key_value_heads: int = 16
    head_dim: int = 128
    hidden_act: str = "silu"
    max_position_embeddings: int = 65536
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0
    total_ut_steps: int = 4
    early_exit_threshold: float = 1.0
    bos_token_id: int | None = 0
    eos_token_id: int | None = 0
    pad_token_id: int | None = None
    tie_word_embeddings: bool = False
    attention_dropout: float = 0.0
    rope_scaling: dict[str, Any] | None = None
    use_sliding_window: bool = False
    sliding_window: int | None = None
    layer_types: list[str] | None = None

    def __post_init__(self) -> None:
        for name in (
            "vocab_size",
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "max_position_embeddings",
            "total_ut_steps",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.head_dim % 2:
            raise ValueError("head_dim must be even for split-half RoPE")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        for name in ("rope_theta", "rms_norm_eps", "initializer_range"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.early_exit_threshold) or not 0 <= self.early_exit_threshold <= 1:
            raise ValueError("early_exit_threshold must be in [0, 1]")
        if self.hidden_act != "silu":
            raise ValueError("Only Ouro's silu activation is supported")
        if self.rope_scaling is not None:
            raise ValueError("RoPE scaling is not supported")
        if self.use_sliding_window or self.sliding_window is not None:
            raise ValueError("Sliding-window attention is not supported")
        if self.layer_types is not None and (
            len(self.layer_types) != self.num_hidden_layers
            or any(layer != "full_attention" for layer in self.layer_types)
        ):
            raise ValueError("Every Ouro layer must use full_attention")
        if self.tie_word_embeddings:
            raise ValueError("Tied embeddings are not supported by the Ouro-1.4B loader")
        if self.attention_dropout != 0:
            raise ValueError("Inference requires attention_dropout=0")
        for name in ("bos_token_id", "eos_token_id", "pad_token_id"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 0 <= value < self.vocab_size
            ):
                raise ValueError(f"{name} must be a vocabulary index or None")

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "OuroConfig":
        if values.get("model_type", "ouro") != "ouro":
            raise ValueError("Only model_type='ouro' is supported")
        if values.get("architectures", ["OuroForCausalLM"]) != ["OuroForCausalLM"]:
            raise ValueError("Only the OuroForCausalLM architecture is supported")
        names = {field.name for field in fields(cls)}
        metadata = {
            "architectures",
            "auto_map",
            "model_type",
            "torch_dtype",
            "dtype",
            "transformers_version",
            "max_window_layers",
            "use_cache",
            "_name_or_path",
        }
        unknown = values.keys() - names - metadata
        if unknown:
            raise ValueError(f"Unsupported Ouro configuration fields: {sorted(unknown)}")
        config_values = {key: value for key, value in values.items() if key in names}
        if "head_dim" not in config_values and "hidden_size" in config_values:
            heads = config_values.get("num_attention_heads", cls.num_attention_heads)
            if config_values["hidden_size"] % heads:
                raise ValueError("hidden_size must be divisible by num_attention_heads")
            config_values["head_dim"] = config_values["hidden_size"] // heads
        if config_values.get("num_key_value_heads", 1) is None:
            config_values["num_key_value_heads"] = config_values.get(
                "num_attention_heads", cls.num_attention_heads
            )
        return cls(**config_values)

    def to_dict(self) -> dict[str, Any]:
        return {"model_type": "ouro", "architectures": ["OuroForCausalLM"], **asdict(self)}


# Backward-compatible aliases for shared layers
OuroRMSNorm = RMSNorm
OuroRotaryEmbedding = RotaryEmbedding
_rotate_half = rotate_half


class _StackedProjections:
    """Bias-free projections of one input, stored as the rows of one matrix.

    The projections' parameters become views of the stacked weight, so one
    matrix product computes all of them without a second copy of the weights.
    Stacking moves the weights and therefore happens once, when a checkpoint
    is loaded; ``weight`` is ``None`` as soon as a projection's storage is no
    longer the stacked rows (``Module.to``, a replaced parameter), and callers
    then use the projections one by one.
    """

    __slots__ = ("_pointers", "_weight", "modules")

    def __init__(self, *modules: nn.Linear) -> None:
        self.modules = modules
        self._weight = None
        self._pointers = ()

    def stack(self) -> None:
        parameters = [module._parameters["weight"] for module in self.modules]
        weight = torch.cat([parameter.data for parameter in parameters])
        start = 0
        for parameter in parameters:
            stop = start + parameter.shape[0]
            parameter.data = weight[start:stop]
            start = stop
        self._weight = weight
        self._pointers = tuple(parameter.data_ptr() for parameter in parameters)

    @property
    def weight(self) -> torch.Tensor | None:
        if self._weight is not None and self._pointers == tuple(
            module._parameters["weight"].data_ptr() for module in self.modules
        ):
            return self._weight
        return None


class OuroAttention(nn.Module):
    def __init__(self, config: OuroConfig, layer_idx: int) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * config.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * config.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * config.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * config.head_dim, config.hidden_size, bias=False
        )
        self._qkv = _StackedProjections(self.q_proj, self.k_proj, self.v_proj)

    def _single_launch(self, hidden, cos, sin, batch, cache, ops):
        """``forward`` with one matrix product for q/k/v and one launch for RoPE."""
        config = self.config
        rows, dim = hidden.shape[0], config.head_dim
        q_heads, k_heads = config.num_attention_heads, config.num_key_value_heads
        weight = self._qkv.weight
        if weight is None:
            q = F.linear(hidden, self.q_proj.weight)
            k = F.linear(hidden, self.k_proj.weight)
            v = F.linear(hidden, self.v_proj.weight).view(rows, k_heads, dim)
            q, k = ops.rope_qk(q, k, cos, sin, q_heads=q_heads, k_heads=k_heads)
        else:
            qkv = F.linear(hidden, weight)
            q, k = ops.rope_qk(
                qkv, qkv, cos, sin, q_heads=q_heads, k_heads=k_heads, k_offset=q_heads * dim
            )
            v = qkv.view(rows, -1, dim)[:, q_heads + k_heads :]
        cache._write_prepared(self.layer_idx, batch, k, v)
        output = cache._attend_prepared(self.layer_idx, batch, q)
        return F.linear(output.reshape(rows, -1), self.o_proj.weight)

    def forward(
        self,
        hidden: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        batch: "KVMetadata | DeviceKVMetadata",
        cache: "KVCacheManager",
    ) -> torch.Tensor:
        shape = (hidden.shape[0], -1, self.config.head_dim)
        q = self.q_proj(hidden).view(shape)
        k = self.k_proj(hidden).view(shape)
        v = self.v_proj(hidden).view(shape)
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        cache._write_prepared(self.layer_idx, batch, k, v)
        output = cache._attend_prepared(self.layer_idx, batch, q)
        return self.o_proj(output.reshape(hidden.shape[0], -1))


class OuroMLP(nn.Module):
    def __init__(self, config: OuroConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self._gate_up = _StackedProjections(self.gate_proj, self.up_proj)

    def _single_launch(self, hidden, ops):
        """``forward`` with one matrix product for gate/up and one launch for the gating."""
        weight = self._gate_up.weight
        if weight is None:
            return self.forward(hidden)
        return F.linear(ops.silu_mul(F.linear(hidden, weight)), self.down_proj.weight)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.down_proj(silu_and_mul(self.gate_proj(hidden), self.up_proj(hidden)))


class OuroDecoderLayer(nn.Module):
    def __init__(self, config: OuroConfig, layer_idx: int) -> None:
        super().__init__()
        self.self_attn = OuroAttention(config, layer_idx)
        self.mlp = OuroMLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.input_layernorm_2 = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm_2 = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def _single_launch(self, hidden, normed, following, cos, sin, batch, cache, ops):
        """``forward`` given ``normed = input_layernorm(hidden)``.

        Each residual sum also normalizes itself for its reader, so the block
        launches two normalizations instead of four and two additions. Returns
        the block's output and ``following(output)``, where ``following`` is
        the RMSNorm that reads this block: the next block's input norm.
        """
        norm, after = self.input_layernorm_2, self.post_attention_layernorm
        attention = self.self_attn._single_launch(normed, cos, sin, batch, cache, ops)
        hidden, normed = ops.rms_norm_add(
            attention,
            norm.weight,
            norm.variance_epsilon,
            hidden,
            after.weight,
            after.variance_epsilon,
        )
        norm = self.post_attention_layernorm_2
        return ops.rms_norm_add(
            self.mlp._single_launch(normed, ops),
            norm.weight,
            norm.variance_epsilon,
            hidden,
            following.weight,
            following.variance_epsilon,
        )

    def forward(
        self,
        hidden: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        batch: "KVMetadata | DeviceKVMetadata",
        cache: "KVCacheManager",
    ) -> torch.Tensor:
        attention = self.self_attn(self.input_layernorm(hidden), position_embeddings, batch, cache)
        hidden = hidden + self.input_layernorm_2(attention)
        return hidden + self.post_attention_layernorm_2(
            self.mlp(self.post_attention_layernorm(hidden))
        )


class OuroModel(nn.Module):
    def __init__(self, config: OuroConfig) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList(
            OuroDecoderLayer(config, layer) for layer in range(config.num_hidden_layers)
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(head_dim=config.head_dim, rope_theta=config.rope_theta)
        self.early_exit_gate = nn.Linear(config.hidden_size, 1, bias=True)


class OuroForCausalLM(nn.Module):
    """Inference-only native Ouro-1.4B with checkpoint-compatible parameter names."""

    def __init__(self, config: OuroConfig) -> None:
        super().__init__()
        self.config = config
        self.model = OuroModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.apply(self._initialize)
        self.requires_grad_(False)
        self.eval()

    def _initialize(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0, std=self.config.initializer_range)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
            if isinstance(module, nn.Embedding) and module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].zero_()

    @property
    def _readers(self) -> list[RMSNorm]:
        """The RMSNorm that reads each block's output: the next block's, then the final one."""
        layers = self.model.layers
        return [layer.input_layernorm for layer in list(layers)[1:]] + [self.model.norm]

    def _single_launch_ops(self, hidden: torch.Tensor):
        """``kernels.layer_ops`` when this traversal can run on it, else a false value."""
        if (
            not hidden.is_cuda
            or hidden.ndim != 2
            or torch.is_grad_enabled()
            or not hidden.is_contiguous()
        ):
            return None
        ops = single_launch_ops()
        config = self.config
        weight = self.model.norm.weight
        if (
            ops
            and hidden.dtype in ops.ROUNDED
            and weight.dtype == hidden.dtype
            and weight.device == hidden.device
            and hidden.shape[1] <= ops.MAX_WIDTH
            and ops.rope_supported(
                config.num_attention_heads, config.num_key_value_heads, config.head_dim
            )
        ):
            return ops
        return None

    def stack_projections(self) -> None:
        """Store each block's q/k/v and gate/up weights as one matrix each.

        This moves those weights, so it must precede everything that keeps
        their addresses, CUDA graph capture in particular; ``from_pretrained``
        calls it once after loading. Parameter names, shapes and values are
        unchanged.
        """
        for layer in self.model.layers:
            layer.self_attn._qkv.stack()
            layer.mlp._gate_up.stack()

    def prelude(self, token_ids: torch.Tensor) -> torch.Tensor:
        if token_ids.ndim != 1:
            raise ValueError("prelude expects packed token_ids with shape [N]")
        return self.model.embed_tokens(token_ids)

    def recurrent(
        self,
        hidden: torch.Tensor,
        request_ids: Sequence[str],
        depths: Sequence[int],
        positions: Sequence[int] | torch.Tensor,
        cache: "KVCacheManager",
        *,
        compute_gate: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Execute one full shared core; depths are zero-based cache namespaces."""
        if hidden.ndim != 2 or hidden.shape[1] != self.config.hidden_size:
            raise ValueError("recurrent expects hidden with shape [N, hidden_size]")
        count = hidden.shape[0]
        if count == 0 or not (len(request_ids) == len(depths) == len(positions) == count):
            raise ValueError("recurrent requires matching, nonempty packed metadata")
        # Internal model/cache traversal contract: descriptor ownership and
        # allocation identity are checked again by every prepared layer call.
        batch = cache._prepare_batch(request_ids, depths, positions)
        return self.recurrent_prepared(hidden, batch, cache, compute_gate=compute_gate)

    def recurrent_prepared(self, hidden, batch, cache, *, compute_gate=True):
        """Run core with runner-owned metadata, including inactive padding rows."""
        position_embeddings = self.model.rotary_emb(hidden, batch.position_ids)
        ops = self._single_launch_ops(hidden)
        if ops:
            # The same traversal with each block's operators fused into single
            # launches; the modules' forward methods are its reference.
            cos, sin = position_embeddings
            layers = self.model.layers
            first = layers[0].input_layernorm
            normed = ops.rms_norm(hidden, first.weight, first.variance_epsilon)
            for layer, following in zip(layers, self._readers, strict=True):
                hidden, normed = layer._single_launch(
                    hidden, normed, following, cos, sin, batch, cache, ops
                )
            hidden = normed
        else:
            for layer in self.model.layers:
                hidden = layer(hidden, position_embeddings, batch, cache)
            # Norm is inside the recurrence in Ouro; this normalized state is the
            # next loop's input as well as the gate and LM head input.
            hidden = self.model.norm(hidden)
        return hidden, self.model.early_exit_gate(hidden).squeeze(-1) if compute_gate else None

    def coda(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden)

    @classmethod
    def from_pretrained(
        cls,
        path_or_repo: str | Path,
        *,
        revision: str | None = None,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.bfloat16,
        config: OuroConfig | None = None,
    ) -> "OuroForCausalLM":
        """Stream strictly checked safetensors into a meta model; never execute Hub code.

        Local directories may contain a single model.safetensors or an indexed sharded checkpoint.
        """
        from safetensors import safe_open

        if dtype not in (torch.float32, torch.float16, torch.bfloat16):
            raise ValueError("dtype must be float32, float16, or bfloat16")
        folder = Path(path_or_repo).expanduser()
        if not folder.is_dir():
            from vllm_rlt.models import resolve_model_config

            source, revision, _ = resolve_model_config(path_or_repo, revision=revision)
            folder = Path(source)
        if config is None:
            config = OuroConfig.from_dict(json.loads((folder / "config.json").read_text()))
        elif not isinstance(config, OuroConfig):
            raise TypeError("config must be an OuroConfig")
        index_path = folder / "model.safetensors.index.json"
        index = json.loads(index_path.read_text())["weight_map"] if index_path.is_file() else None
        if index is None:
            shards = [folder / "model.safetensors"]
        else:
            filenames = set(index.values())
            if any(
                Path(name).name != name or not name.endswith(".safetensors") for name in filenames
            ):
                raise ValueError("Checkpoint index must reference local safetensors basenames")
            shards = [folder / name for name in sorted(filenames)]
        if any(not shard.is_file() for shard in shards):
            raise FileNotFoundError(
                "A complete safetensors checkpoint is required (pickle .bin is unsupported)"
            )
        with torch.device("meta"):
            model = cls(config)
        expected = {name: tuple(parameter.shape) for name, parameter in model.named_parameters()}
        discovered: dict[str, Path] = {}
        # Validate every key and shape before allocating model weight memory.
        for shard in shards:
            with safe_open(shard, framework="pt", device="cpu") as checkpoint:
                for name in checkpoint.keys():
                    if name in discovered:
                        raise ValueError(f"Duplicate checkpoint tensor: {name}")
                    if name not in expected:
                        raise ValueError(f"Unexpected checkpoint tensor: {name}")
                    if tuple(checkpoint.get_slice(name).get_shape()) != expected[name]:
                        raise ValueError(f"Checkpoint shape mismatch for {name}")
                    if index is not None and index.get(name) != shard.name:
                        raise ValueError(f"Checkpoint index mismatch for {name}")
                    discovered[name] = shard
        missing = expected.keys() - discovered.keys()
        if missing:
            raise ValueError(f"Missing checkpoint tensors: {sorted(missing)}")
        if index is not None and set(index) != set(expected):
            raise ValueError("Checkpoint index does not match Ouro parameter names")
        for shard in shards:
            with safe_open(shard, framework="pt", device="cpu") as checkpoint:
                for name in checkpoint.keys():
                    tensor = checkpoint.get_tensor(name)
                    if not tensor.is_floating_point():
                        raise ValueError(f"Checkpoint tensor must be floating point: {name}")
                    module_path, parameter_name = name.rsplit(".", 1)
                    module = model.get_submodule(module_path)
                    setattr(
                        module,
                        parameter_name,
                        nn.Parameter(tensor.to(device=device, dtype=dtype), requires_grad=False),
                    )
        model.model.rotary_emb.inv_freq = model.model.rotary_emb.frequencies(device=device)
        if torch.device(device).type == "cuda":
            model.stack_projections()
        return model.eval()
