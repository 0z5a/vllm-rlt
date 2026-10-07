"""GPU model checks for shared FP8 producers and mixed-depth engine execution."""

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from loopquant.fused_fp8 import RowwiseFP8Linear
from loopquant.quantizers import ScaleLayout, fp8_encode
from loopquant.serving import quantize_native_core
from tests.helpers import tiny_nanbeige_config, tiny_ouro_config
from vllm_rlt import LLM, CacheConfig, ExecutionConfig, ExitConfig, SamplingParams, SchedulerConfig
from vllm_rlt.models.nanbeige import NanbeigeForCausalLM
from vllm_rlt.models.ouro import OuroForCausalLM


@dataclass
class RowContext:
    loop_ids: torch.Tensor | None = None


class ReferenceLinear(nn.Module):
    def __init__(self, linear, scales, layout, mode, context):
        super().__init__()
        weight = linear.weight.detach().float()
        weight_scale = weight.abs().amax().clamp_min(torch.finfo(torch.float32).tiny) / 448
        self.register_buffer(
            "decoded_weight", fp8_encode(weight, weight_scale).float() * weight_scale
        )
        self.register_buffer("scales", scales)
        self.layout, self.mode, self.context = layout, mode, context

    def forward(self, values):
        if self.mode == "dynamic":
            scale = values.float().abs().amax(-1, keepdim=True).clamp_min(1e-12) / 448
        else:
            assert self.context.loop_ids is not None
            scale = self.scales[self.layout.indices(self.context.loop_ids)].unsqueeze(-1)
        decoded = fp8_encode(values, scale).float() * scale
        return nn.functional.linear(decoded, self.decoded_weight).to(values.dtype)


class ReferenceProjections:
    def quantize(self, scales, layout, mode):
        self.context = RowContext()
        for name, module in list(self.named_modules()):
            if name.startswith("model.layers.") and isinstance(module, nn.Linear):
                parent, field = name.rsplit(".", 1)
                self.get_submodule(parent).add_module(
                    field, ReferenceLinear(module, scales[name], layout, mode, self.context)
                )

    def recurrent_prepared(self, hidden, batch, cache, *, compute_gate=True):
        self.context.loop_ids = batch.loop_ids
        return super().recurrent_prepared(hidden, batch, cache, compute_gate=compute_gate)


class ReferenceOuro(ReferenceProjections, OuroForCausalLM):
    pass


class ReferenceNanbeige(ReferenceProjections, NanbeigeForCausalLM):
    pass


def check_fused_native_model(
    policy: str,
    graphs: bool,
    fuse_norm: bool,
    *,
    export_folder: Path | None = None,
    family: str = "ouro",
    skip_loop_final_norm: bool = False,
) -> dict:
    torch.manual_seed(37)
    if family == "ouro":
        config = tiny_ouro_config(hidden_size=64, head_dim=32, intermediate_size=128)
        reference = ReferenceOuro(config)
        native = OuroForCausalLM(config)
    elif family == "nanbeige":
        config = tiny_nanbeige_config(
            hidden_size=64,
            head_dim=32,
            intermediate_size=256,
            skip_loop_final_norm=skip_loop_final_norm,
        )
        reference = ReferenceNanbeige(config)
        native = NanbeigeForCausalLM(config)
    else:
        raise ValueError("unknown model family")
    reference = reference.to(device="cuda", dtype=torch.bfloat16)
    native = native.to(device="cuda", dtype=torch.bfloat16)
    native.load_state_dict(reference.state_dict())
    mode = "dynamic" if policy == "DYN" else "static"
    loops = config.total_ut_steps
    layout = ScaleLayout(
        loops, {"DYN": (), "SH1": (), "ST2": (loops // 2,), "LOOP": tuple(range(1, loops))}[policy]
    )
    scales = {
        name: torch.tensor([0.01, 0.02, 0.03, 0.04][: layout.stages], device="cuda")
        for name, module in reference.named_modules()
        if name.startswith("model.layers.") and isinstance(module, nn.Linear)
    }
    reference.quantize(scales, layout, mode)
    scratch = quantize_native_core(native, scales, layout, mode, max_rows=8, fuse_norm=fuse_norm)
    assert scratch.nbytes > 0
    if export_folder is not None:
        from loopquant.native_export import export_native_model, load_native_model

        export_native_model(native, export_folder, model_revision="a" * 40)
        restored = load_native_model(export_folder, torch.device("cuda"), max_rows=8)
        original_state, restored_state = native.state_dict(), restored.state_dict()
        assert original_state.keys() == restored_state.keys()
        for name in original_state:
            assert torch.equal(
                original_state[name].view(torch.uint8), restored_state[name].view(torch.uint8)
            ), name
        native = restored
    weights = {
        name: layer.packed_weight.data_ptr()
        for name, layer in native.named_modules()
        if isinstance(layer, RowwiseFP8Linear)
    }
    assert len(weights) == 7 * config.num_hidden_layers
    common = dict(
        cache_config=CacheConfig(64, 16),
        scheduler_config=SchedulerConfig(max_num_seqs=3, max_num_batched_tokens=8),
        attention_backend="triton",
        exit_config=ExitConfig(mode="ouro_delayed"),
    )
    expected = LLM(reference, **common)
    actual = LLM(
        native,
        execution_config=ExecutionConfig(
            cuda_graphs=graphs,
            async_scheduling=True,
            multi_stream=True,
            static_buffers=True,
            pad_to_power_of_two=True,
        ),
        **common,
    )
    prompts = [[5, 7], [3, 9, 2], [8]]
    params = [
        SamplingParams(max_tokens=4, min_loops=depth, max_loops=depth, ignore_eos=True)
        for depth in [1, loops, 2]
    ]
    for _ in range(2):
        before = expected.generate(prompts, params)
        after = actual.generate(prompts, params)
        assert [(row.token_ids, row.exit_depths) for row in before] == [
            (row.token_ids, row.exit_depths) for row in after
        ]
        assert actual.engine.cache_manager.num_used_blocks == 0
    assert weights == {
        name: layer.packed_weight.data_ptr()
        for name, layer in native.named_modules()
        if isinstance(layer, RowwiseFP8Linear)
    }
    assert all(
        not name.startswith("model.layers.") or "layernorm" in name
        for name, _ in native.named_parameters()
    )
    if graphs:
        assert (
            actual.engine.model_runner.graphs.replays > actual.engine.model_runner.graphs.captures
        )
    return dict(
        model_family=family,
        skip_loop_final_norm=skip_loop_final_norm,
        policy=policy,
        graphs=graphs,
        fused_norm=fuse_norm,
        repeated_requests=6,
        physical_packed_weights=len(weights),
        scratch_bytes=scratch.nbytes,
        captures=actual.engine.model_runner.graphs.captures if graphs else 0,
        replays=actual.engine.model_runner.graphs.replays if graphs else 0,
        export_roundtrip=export_folder is not None,
    )
