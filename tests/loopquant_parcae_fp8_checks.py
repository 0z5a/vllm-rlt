"""Decoded FP8 arithmetic and native Parcae engine qualification."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from torch import nn

from loopquant.fused_fp8 import RowwiseFP8Linear
from loopquant.int4_export import core_projections
from loopquant.parcae_fp8 import quantize_parcae_fp8
from loopquant.parcae_fp8_export import export_parcae_fp8, load_parcae_fp8
from loopquant.quantizers import ScaleLayout, fp8_encode
from tests.loopquant_int4_checks import int4_reference_model
from vllm_rlt import CacheConfig, ExecutionConfig, SamplingParams, SchedulerConfig
from vllm_rlt.engine.llm_engine import LLMEngine
from vllm_rlt.models.parcae import ParcaeForCausalLM


@dataclass
class RowDepth:
    loop_ids: torch.Tensor | None = None


class ReferenceFP8Linear(nn.Module):
    def __init__(
        self,
        linear: nn.Linear,
        scales: torch.Tensor,
        layout: ScaleLayout,
        mode: Literal["static", "dynamic"],
        context: RowDepth,
    ) -> None:
        super().__init__()
        weight = linear.weight.detach().float()
        weight_scale = weight.abs().amax().clamp_min(torch.finfo(torch.float32).tiny) / 448
        self.register_buffer(
            "decoded_weight", fp8_encode(weight, weight_scale).float() * weight_scale
        )
        self.register_buffer("scales", scales)
        self.layout, self.mode, self.context = layout, mode, context

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if self.mode == "dynamic":
            scale = values.float().abs().amax(-1, keepdim=True).clamp_min(1e-12) / 448
        else:
            assert self.context.loop_ids is not None
            scale = self.scales[self.layout.indices(self.context.loop_ids)].unsqueeze(-1)
        decoded = fp8_encode(values, scale).float() * scale
        return nn.functional.linear(decoded, self.decoded_weight).to(values.dtype)


class ReferenceParcae(ParcaeForCausalLM):
    def quantize(
        self,
        scales: dict[str, torch.Tensor],
        layout: ScaleLayout,
        mode: Literal["static", "dynamic"],
    ) -> None:
        self.context = RowDepth()
        for name, layer in core_projections(self).items():
            parent, field = name.rsplit(".", 1)
            self.get_submodule(parent).add_module(
                field, ReferenceFP8Linear(layer, scales[name], layout, mode, self.context)
            )

    def recurrent_prepared(self, hidden, batch, cache, *, compute_gate=True):
        self.context.loop_ids = batch.loop_ids
        return super().recurrent_prepared(hidden, batch, cache, compute_gate=compute_gate)


@torch.inference_mode()
def check_parcae_fp8(folder: Path, policy: str, graphs: bool, fuse_norm: bool) -> dict:
    torch.manual_seed(37)
    native = int4_reference_model("parcae").to(device="cuda", dtype=torch.bfloat16).eval()
    reference = ReferenceParcae(native.config).to(device="cuda", dtype=torch.bfloat16).eval()
    reference.freqs_cis = native.freqs_cis.clone()
    reference.load_state_dict(native.state_dict())
    mode = "dynamic" if policy == "DYN" else "static"
    layout = ScaleLayout(3, {"DYN": (), "SH1": (), "ST2": (1,), "LOOP": (1, 2)}[policy])
    scales = {
        name: torch.tensor([0.01, 0.02, 0.03][: layout.stages], device="cuda")
        for name in core_projections(native)
    }
    reference.quantize(scales, layout, mode)
    scratch = quantize_parcae_fp8(native, scales, layout, mode, max_rows=8, fuse_norm=fuse_norm)
    assert scratch.storage
    export_parcae_fp8(native, folder, model_revision="a" * 40)
    restored = load_parcae_fp8(folder, torch.device("cuda"), max_rows=8)
    left, right = native.state_dict(), restored.state_dict()
    assert left.keys() == right.keys()
    assert all(
        torch.equal(left[name].view(torch.uint8), right[name].view(torch.uint8)) for name in left
    )
    assert restored.lm_head.weight is restored.transformer.wte.weight
    assert restored.freqs_cis.dtype == torch.float32
    del native, left, right
    native = restored
    weights = {
        name: layer.packed_weight.data_ptr()
        for name, layer in native.named_modules()
        if isinstance(layer, RowwiseFP8Linear)
    }
    assert len(weights) == len(set(weights.values())) == 12
    common = dict(
        attention_backend="triton",
        cache_config=CacheConfig(256, 4),
        scheduler_config=SchedulerConfig(max_num_seqs=3, max_num_batched_tokens=8),
    )
    engines = [
        LLMEngine(reference, **common),
        LLMEngine(
            native,
            execution_config=ExecutionConfig(
                cuda_graphs=graphs, static_buffers=True, pad_to_power_of_two=True
            ),
            **common,
        ),
    ]
    results = []
    for engine in engines:
        rounds = []
        for _ in range(2):
            torch.manual_seed(31)
            for i, (prompt, depth) in enumerate(zip(([257, 7], [9, 511, 2], [300]), (3, 3, 3))):
                engine.add_request(
                    str(i),
                    prompt,
                    SamplingParams(max_tokens=4, min_loops=depth, max_loops=depth, ignore_eos=True),
                )
            finished = {}
            while engine.has_unfinished_requests():
                for output in engine.step():
                    if output.finished:
                        finished[output.request_id] = (output.token_ids, output.exit_depths)
            assert len(finished) == 3 and engine.cache_manager.num_used_blocks == 0
            for i, depth in enumerate((3, 3, 3)):
                assert finished[str(i)][1] == [3, depth, depth, depth]
            rounds.append(finished)
        results.append(rounds)
    assert results[0] == results[1]
    assert weights == {
        name: layer.packed_weight.data_ptr()
        for name, layer in native.named_modules()
        if isinstance(layer, RowwiseFP8Linear)
    }
    graph = engines[1].model_runner.graphs
    if graphs:
        assert graph.captures > 0 and graph.replays > graph.captures
    return dict(
        policy=policy,
        graphs=graphs,
        fuse_norm=fuse_norm,
        encoded_reload_exact=True,
        physical_packed_weights=len(weights),
        repeated_requests=6,
        decode_depths=[3, 3, 3],
        kv_after_drain=0,
        captures=graph.captures if graphs else 0,
        replays=graph.replays if graphs else 0,
    )
