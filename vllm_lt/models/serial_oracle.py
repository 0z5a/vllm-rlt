# SPDX-License-Identifier: Apache-2.0
"""Independent functional Ouro reference with dense LAST-EXITED token history.

Equations follow the pinned official Ouro model, including eager native-dtype
attention and FP32 RMSNorm/RoPE arithmetic. No native execution/cache helpers are
used. The engine's declared gate contract uses FP32 sigmoid and a host scalar CDF.
"""

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

import torch
from torch.nn import functional as F

from .config import OuroConfig


@dataclass(frozen=True)
class ExitPolicy:
    min_loops: int = 2
    max_loops: int | None = None
    exit_threshold: float = 1.0

    def __post_init__(self):
        for name in ("min_loops", "max_loops"):
            value = getattr(self, name)
            if name == "max_loops" and value is None:
                continue
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.max_loops is not None and self.min_loops > self.max_loops:
            raise ValueError("minimum depth exceeds maximum depth")
        threshold = self.exit_threshold
        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
            or not math.isfinite(threshold)
            or not 0 <= threshold <= 1
        ):
            raise ValueError("exit_threshold must be finite and in [0, 1]")


@dataclass(frozen=True)
class Boundary:
    operation: str
    positions: tuple[int, ...]
    depth: int  # One-based, matching serialized traces.
    layer: int | None  # Physical layers are zero-based.


Observer = Callable[[Boundary, torch.Tensor], None]


@dataclass(frozen=True)
class TokenTrace:
    position: int
    output_index: int
    exit_depth: int
    gate_logits: tuple[float, ...]
    gate_probabilities: tuple[float, ...]
    cumulative_probabilities: tuple[float, ...]
    logits: torch.Tensor  # [vocab], independent of oracle cache/state buffers.
    hidden: torch.Tensor  # [hidden_size], normalized at the final executed loop.


@dataclass(frozen=True)
class KVSnapshot:
    positions: tuple[int, ...]
    keys: torch.Tensor  # [depth, layer, selected position, KV head, dimension]
    values: torch.Tensor
    initialized: torch.Tensor  # CPU bool [depth, layer, selected position]


class SerialOuroOracle:
    """One request's diagnostic state; callers supply every subsequent input.

    Weights/configuration are shared read-only. Observer values are borrowed,
    detached tensors: callbacks must not mutate them and must copy planned
    snapshots they retain. TokenTrace and KVSnapshot own their tensor copies.
    """

    def __init__(self, config: OuroConfig, weights: Mapping[str, torch.Tensor], *, capacity: int):
        if type(capacity) is not int or not 0 < capacity <= config.max_position_embeddings:
            raise ValueError("capacity must be a positive integer within the model context limit")
        self.config = config
        self.capacity = capacity
        self.length = 0
        self.output_index = -1
        self._closed = False
        self._failed = False
        expected = self._weight_shapes(config)
        selected = {}
        for name, shape in expected.items():
            tensor = weights.get(name)
            if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != shape:
                raise ValueError(f"missing or incorrectly shaped oracle weight {name!r}")
            selected[name] = tensor
        parameter = selected["model.embed_tokens.weight"]
        self.device, self.dtype = parameter.device, parameter.dtype
        if self.dtype not in (torch.float32, torch.float16, torch.bfloat16):
            raise ValueError("oracle weights must be FP32, FP16, or BF16")
        if any(t.device != self.device or t.dtype != self.dtype for t in selected.values()):
            raise ValueError("oracle weights must share one device and dtype")
        self.weights = MappingProxyType(selected)
        shape = (
            config.total_ut_steps,
            config.num_hidden_layers,
            capacity,
            config.num_key_value_heads,
            config.head_dim,
        )
        self.key_cache = torch.empty(shape, device=self.device, dtype=self.dtype)
        self.value_cache = torch.empty_like(self.key_cache)
        self.initialized = torch.zeros(shape[:3], device="cpu", dtype=torch.bool)
        self._inv_freq = 1.0 / (
            config.rope_theta
            ** (
                torch.arange(0, config.head_dim, 2, dtype=torch.float32, device=self.device)
                / config.head_dim
            )
        )

    @staticmethod
    def _weight_shapes(config):
        hidden, intermediate = config.hidden_size, config.intermediate_size
        query = config.num_attention_heads * config.head_dim
        kv = config.num_key_value_heads * config.head_dim
        shapes = {
            "model.embed_tokens.weight": (config.vocab_size, hidden),
            "model.norm.weight": (hidden,),
            "model.early_exit_gate.weight": (1, hidden),
            "model.early_exit_gate.bias": (1,),
            "lm_head.weight": (config.vocab_size, hidden),
        }
        for layer in range(config.num_hidden_layers):
            prefix = f"model.layers.{layer}"
            for norm in (
                "input_layernorm",
                "input_layernorm_2",
                "post_attention_layernorm",
                "post_attention_layernorm_2",
            ):
                shapes[f"{prefix}.{norm}.weight"] = (hidden,)
            for name, shape in {
                "self_attn.q_proj": (query, hidden),
                "self_attn.k_proj": (kv, hidden),
                "self_attn.v_proj": (kv, hidden),
                "self_attn.o_proj": (hidden, query),
                "mlp.gate_proj": (intermediate, hidden),
                "mlp.up_proj": (intermediate, hidden),
                "mlp.down_proj": (hidden, intermediate),
            }.items():
                shapes[f"{prefix}.{name}.weight"] = shape
        return shapes

    def _ensure_ready(self):
        if self._closed:
            raise RuntimeError("oracle is closed")
        if self._failed:
            raise RuntimeError("oracle is invalid after a failed execution; close it")

    def _validate_token(self, token):
        if type(token) is not int or not 0 <= token < self.config.vocab_size:
            raise ValueError("input token must be an integer within the vocabulary")

    def _norm(self, value, name):
        fp32 = value.float()
        scaled = fp32 * torch.rsqrt(
            fp32.square().mean(dim=-1, keepdim=True) + self.config.rms_norm_eps
        )
        return scaled.to(value.dtype) * self.weights[f"{name}.weight"]

    def _linear(self, value, name):
        return F.linear(value, self.weights[f"{name}.weight"])

    @staticmethod
    def _emit(observer, operation, positions, depth, layer, value):
        if observer is not None:
            observer(Boundary(operation, positions, depth + 1, layer), value.detach())

    def _core(self, hidden, positions, depth, observer):
        config = self.config
        index = torch.tensor(positions, device=self.device, dtype=torch.long)
        angles = torch.outer(index.float(), self._inv_freq)
        phase = torch.cat((angles, angles), dim=-1)
        cos, sin = phase.cos().to(self.dtype)[:, None], phase.sin().to(self.dtype)[:, None]
        context_length = positions[-1] + 1
        context_index = torch.arange(context_length, device=self.device)
        future = context_index[None, :] > index[:, None]

        def rotary(value):
            split = config.head_dim // 2
            rotated = torch.cat((-value[..., split:], value[..., :split]), dim=-1)
            return value * cos + rotated * sin

        for layer in range(config.num_hidden_layers):
            prefix = f"model.layers.{layer}"
            attention_input = self._norm(hidden, f"{prefix}.input_layernorm")
            self._emit(observer, "attention_input", positions, depth, layer, attention_input)
            query = rotary(
                self._linear(attention_input, f"{prefix}.self_attn.q_proj").reshape(
                    len(positions),
                    config.num_attention_heads,
                    config.head_dim,
                )
            )
            key = rotary(
                self._linear(attention_input, f"{prefix}.self_attn.k_proj").reshape(
                    len(positions),
                    config.num_key_value_heads,
                    config.head_dim,
                )
            )
            value = self._linear(attention_input, f"{prefix}.self_attn.v_proj").reshape(
                len(positions),
                config.num_key_value_heads,
                config.head_dim,
            )
            for operation, tensor in (("query", query), ("key", key), ("value", value)):
                self._emit(observer, operation, positions, depth, layer, tensor)
            self.key_cache[depth, layer, index] = key
            self.value_cache[depth, layer, index] = value
            self.initialized[depth, layer, list(positions)] = True
            if not bool(self.initialized[depth, layer, :context_length].all()):
                raise RuntimeError(
                    f"uninitialized oracle history at depth {depth + 1}, layer {layer}"
                )
            groups = config.num_attention_heads // config.num_key_value_heads
            keys = self.key_cache[depth, layer, :context_length].repeat_interleave(groups, dim=1)
            values = self.value_cache[depth, layer, :context_length].repeat_interleave(
                groups, dim=1
            )
            # Preserve the published eager attention's native-dtype rounding.
            scores = (
                torch.matmul(query.transpose(0, 1), keys.permute(1, 2, 0)) * config.head_dim**-0.5
            )
            scores = scores.masked_fill(future[None], -torch.inf)
            probabilities = F.softmax(scores, dim=-1, dtype=torch.float32).to(self.dtype)
            attended = torch.matmul(probabilities, values.transpose(0, 1))
            attended = attended.transpose(0, 1).reshape(len(positions), -1)
            attention_output = self._linear(attended, f"{prefix}.self_attn.o_proj")
            self._emit(observer, "attention_output", positions, depth, layer, attention_output)
            hidden = hidden + self._norm(attention_output, f"{prefix}.input_layernorm_2")
            mlp_input = self._norm(hidden, f"{prefix}.post_attention_layernorm")
            activated = F.silu(self._linear(mlp_input, f"{prefix}.mlp.gate_proj"))
            mlp_output = self._linear(
                activated * self._linear(mlp_input, f"{prefix}.mlp.up_proj"),
                f"{prefix}.mlp.down_proj",
            )
            hidden = hidden + self._norm(mlp_output, f"{prefix}.post_attention_layernorm_2")
            self._emit(observer, "layer_output", positions, depth, layer, hidden)
        hidden = self._norm(hidden, "model.norm")
        self._emit(observer, "loop_hidden", positions, depth, None, hidden)
        gates = F.linear(
            hidden,
            self.weights["model.early_exit_gate.weight"],
            self.weights["model.early_exit_gate.bias"],
        ).squeeze(-1)
        self._emit(observer, "gate_logits", positions, depth, None, gates)
        return hidden, gates

    def _execute(self, tokens, positions, *, policy, forced_depth, observer):
        hidden = self.weights["model.embed_tokens.weight"][list(tokens)]
        maximum = policy.max_loops or self.config.total_ut_steps
        logits_history, probabilities, cumulative = [], [], []
        remaining = 1.0
        for depth in range(maximum):
            hidden, gates = self._core(hidden, positions, depth, observer)
            logit = float(gates[-1].float().item())
            probability = float(gates[-1].float().sigmoid().item())
            if not math.isfinite(logit) or not math.isfinite(probability):
                raise ValueError("nonfinite oracle gate")
            remaining *= 1.0 - probability
            logits_history.append(logit)
            probabilities.append(probability)
            cumulative.append(1.0 - remaining)
            completed = depth + 1
            should_exit = (
                completed == forced_depth
                if forced_depth is not None
                else (
                    completed >= maximum
                    or (
                        policy.exit_threshold < 1.0
                        and completed >= policy.min_loops
                        and cumulative[-1] >= policy.exit_threshold
                    )
                )
            )
            if should_exit:
                break
        if len(positions) == 1:
            position = positions[0]
            for skipped in range(completed, self.config.total_ut_steps):
                self.key_cache[skipped, :, position].copy_(
                    self.key_cache[completed - 1, :, position]
                )
                self.value_cache[skipped, :, position].copy_(
                    self.value_cache[completed - 1, :, position]
                )
                self.initialized[skipped, :, position] = self.initialized[
                    completed - 1, :, position
                ]
        final_hidden = hidden[-1:].clone()
        logits = self._linear(final_hidden, "lm_head")
        self._emit(observer, "logits", (positions[-1],), completed - 1, None, logits)
        if not bool(logits.isfinite().all()):
            raise ValueError("nonfinite oracle logits")
        self.length = positions[-1] + 1
        self.output_index += 1
        return TokenTrace(
            positions[-1],
            self.output_index,
            completed,
            tuple(logits_history),
            tuple(probabilities),
            tuple(cumulative),
            logits[0].clone(),
            final_hidden[0].clone(),
        )

    @torch.inference_mode()
    def prefill(self, prompt_ids: Sequence[int], *, observer: Observer | None = None) -> TokenTrace:
        self._ensure_ready()
        if self.length:
            raise RuntimeError("prefill may run only once per oracle")
        tokens = tuple(prompt_ids)
        if not tokens or len(tokens) > self.capacity:
            raise ValueError("prompt must be nonempty and fit the oracle capacity")
        for token in tokens:
            self._validate_token(token)
        try:
            return self._execute(
                tokens,
                tuple(range(len(tokens))),
                policy=ExitPolicy(),
                forced_depth=self.config.total_ut_steps,
                observer=observer,
            )
        except Exception:
            self._failed = True
            raise

    @torch.inference_mode()
    def advance(
        self,
        input_token_id: int,
        *,
        policy: ExitPolicy | None = None,
        forced_depth: int | None = None,
        observer: Observer | None = None,
    ) -> TokenTrace:
        self._ensure_ready()
        if not self.length:
            raise RuntimeError("advance requires successful prefill")
        self._validate_token(input_token_id)
        if self.length >= self.capacity:
            raise ValueError("oracle token capacity exhausted")
        policy = ExitPolicy() if policy is None else policy
        if not isinstance(policy, ExitPolicy):
            raise ValueError("policy must be an ExitPolicy")
        maximum = policy.max_loops or self.config.total_ut_steps
        if maximum > self.config.total_ut_steps or policy.min_loops > maximum:
            raise ValueError("exit policy exceeds the model's supported depths")
        if forced_depth is not None and (
            type(forced_depth) is not int or not policy.min_loops <= forced_depth <= maximum
        ):
            raise ValueError("forced depth must be an integer within the policy bounds")
        try:
            return self._execute(
                (input_token_id,),
                (self.length,),
                policy=policy,
                forced_depth=forced_depth,
                observer=observer,
            )
        except Exception:
            self._failed = True
            raise

    @torch.inference_mode()
    def snapshot_kv(self, *, positions: Sequence[int] | None = None) -> KVSnapshot:
        self._ensure_ready()
        positions = tuple(range(self.length)) if positions is None else tuple(positions)
        if any(type(pos) is not int or not 0 <= pos < self.capacity for pos in positions):
            raise ValueError("snapshot positions must be integers within oracle capacity")
        if len(set(positions)) != len(positions):
            raise ValueError("snapshot positions must be unique")
        # Values at false initialized slots are unspecified; the mask is authoritative.
        return KVSnapshot(
            positions,
            self.key_cache[:, :, list(positions)].clone(),
            self.value_cache[:, :, list(positions)].clone(),
            self.initialized[:, :, list(positions)].clone(),
        )

    def close(self) -> None:
        """Release only this oracle's references/storage; repeated cleanup is harmless."""
        self.key_cache = self.value_cache = self.initialized = self._inv_freq = None
        self.weights = MappingProxyType({})
        self._closed = True
