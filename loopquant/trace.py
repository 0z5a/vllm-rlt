"""Stream native Ouro activations using the engine's actual row metadata.

This diagnostic path synchronizes statistics to the CPU. It is not a timed
serving backend and must run outside CUDA graphs.
"""

import json
from collections import defaultdict
from dataclasses import dataclass
from typing import TextIO

import torch
from torch import nn

from vllm_rlt.core.kv_cache_manager import KVCacheManager, _PreparedKVBatch
from vllm_rlt.layers import RMSNorm
from vllm_rlt.models.ouro import OuroConfig, OuroForCausalLM

from .stats import ActivationStats


@dataclass(frozen=True)
class TraceRow:
    request_id: str | None
    token_position: int
    loop_id: int
    phase: str
    valid_mask: bool


class ActivationTrace:
    def __init__(self, stream: TextIO, prompt_lengths: dict[str, int]) -> None:
        self.stream = stream
        self.prompt_lengths = dict(prompt_lengths)
        self.rows: tuple[TraceRow, ...] = ()
        self.statistics: dict[tuple[str, str, int, str], ActivationStats] = {}
        self.calls = 0

    def prepare(self, batch: _PreparedKVBatch, size: int) -> None:
        requests = {id(allocation): request for request, allocation in batch.allocations}
        rows = []
        for allocation, loop, position in batch.rows:
            request = requests[id(allocation)]
            phase = "prefill" if position < self.prompt_lengths[request] else "decode"
            rows.append(TraceRow(request, position, loop, phase, True))
        if len(rows) > size:
            raise ValueError("trace metadata exceeds the activation row count")
        rows.extend(TraceRow(None, -1, -1, "padding", False) for _ in range(size - len(rows)))
        self.rows = tuple(rows)

    def record(
        self, name: str, site: str, module: nn.Linear | RMSNorm, values: torch.Tensor
    ) -> None:
        if not self.rows:
            return
        if values.ndim != 2 or values.shape[0] != len(self.rows):
            raise ValueError("native activations must align with the prepared packed rows")
        groups: dict[tuple[int, str], list[int]] = defaultdict(list)
        for index, row in enumerate(self.rows):
            if row.valid_mask:
                groups[row.loop_id, row.phase].append(index)
        for (loop, phase), indices in groups.items():
            valid = torch.zeros(len(self.rows), device=values.device, dtype=torch.bool)
            valid[indices] = True
            key = (name, site, loop, phase)
            self.statistics.setdefault(key, ActivationStats()).update(values, valid)
        self.stream.write(
            json.dumps(
                dict(
                    physical_module_id=name,
                    site=site,
                    loop_id=[row.loop_id for row in self.rows],
                    phase=[row.phase for row in self.rows],
                    request_id=[row.request_id for row in self.rows],
                    token_position=[row.token_position for row in self.rows],
                    valid_mask=[row.valid_mask for row in self.rows],
                    weight_storage_id=module.weight.untyped_storage().data_ptr(),
                    channels=values.shape[-1],
                    call=self.calls,
                )
            )
            + "\n"
        )
        self.calls += 1

    def summary(self) -> list[dict]:
        return [
            dict(physical_module_id=name, site=site, loop_id=loop, phase=phase, **stats.summary())
            for (name, site, loop, phase), stats in sorted(self.statistics.items())
        ]


class TracedOuroForCausalLM(OuroForCausalLM):
    """Same parameters and native execution, with diagnostic hooks on the core."""

    def __init__(self, config: OuroConfig, trace: ActivationTrace) -> None:
        super().__init__(config)
        self.trace = trace
        for name, module in self.named_modules():
            if name.startswith("model.layers.") or name == "model.norm":
                if isinstance(module, (nn.Linear, RMSNorm)):
                    self._observe(name, module)

    def _observe(self, name: str, module: nn.Linear | RMSNorm) -> None:
        def before(layer: nn.Linear | RMSNorm, inputs: tuple[torch.Tensor, ...]) -> None:
            self.trace.record(name, "input", layer, inputs[0])

        module.register_forward_pre_hook(before)
        if isinstance(module, RMSNorm):

            def after(
                layer: RMSNorm, inputs: tuple[torch.Tensor, ...], output: torch.Tensor
            ) -> None:
                self.trace.record(name, "output", layer, output)

            module.register_forward_hook(after)

    def recurrent_prepared(
        self,
        hidden: torch.Tensor,
        batch: _PreparedKVBatch,
        cache: KVCacheManager,
        *,
        compute_gate: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if hidden.is_cuda and torch.cuda.is_current_stream_capturing():
            raise ValueError("activation tracing must run outside CUDA graphs")
        self.trace.prepare(batch, hidden.shape[0])
        result = super().recurrent_prepared(hidden, batch, cache, compute_gate=compute_gate)
        self.trace.rows = ()
        return result
