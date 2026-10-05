"""Decoded FP8 reference and full-depth LoopFormer native KV/Graph checks."""

from pathlib import Path
from typing import Literal

import torch

from loopquant.fused_fp8 import RowwiseFP8Linear
from loopquant.int4_export import core_projections
from loopquant.loopformer_fp8 import quantize_loopformer_fp8
from loopquant.loopformer_fp8_export import export_loopformer_fp8, load_loopformer_fp8
from loopquant.quality import native_window_nll
from loopquant.quantizers import ScaleLayout
from tests.loopquant_int4_checks import int4_reference_model
from tests.loopquant_serving_checks import ReferenceLinear, RowContext
from vllm_rlt import CacheConfig, ExecutionConfig, ExitConfig, SamplingParams, SchedulerConfig
from vllm_rlt.engine.llm_engine import LLMEngine
from vllm_rlt.models.loopformer import LoopFormerForCausalLM


class ReferenceLoopFormer(LoopFormerForCausalLM):
    def quantize(
        self,
        scales: dict[str, torch.Tensor],
        layout: ScaleLayout,
        mode: Literal["static", "dynamic"],
    ) -> None:
        self.context = RowContext()
        for name, layer in core_projections(self).items():
            parent, field = name.rsplit(".", 1)
            self.get_submodule(parent).add_module(
                field, ReferenceLinear(layer, scales[name], layout, mode, self.context)
            )

    def recurrent_prepared(self, hidden, batch, cache, *, compute_gate=True):
        self.context.loop_ids = batch.loop_ids
        return super().recurrent_prepared(hidden, batch, cache, compute_gate=compute_gate)


@torch.inference_mode()
def check_loopformer_fp8(folder: Path, policy: str, graphs: bool, compact: bool) -> dict:
    torch.manual_seed(37)
    native = int4_reference_model("loopformer").to(device="cuda", dtype=torch.bfloat16).eval()
    reference = ReferenceLoopFormer(native.config).to(device="cuda", dtype=torch.bfloat16).eval()
    reference.load_state_dict(native.state_dict())
    mode = "dynamic" if policy == "DYN" else "static"
    layout = ScaleLayout(8, {"DYN": (), "SH1": (), "ST2": (4,), "LOOP": tuple(range(1, 8))}[policy])
    scales = {
        name: torch.linspace(0.005, 0.04, layout.stages, device="cuda")
        for name in core_projections(native)
    }
    reference.quantize(scales, layout, mode)
    quantize_loopformer_fp8(native, scales, layout, mode, max_rows=8)
    export_loopformer_fp8(native, folder, model_revision="a" * 40)
    restored = load_loopformer_fp8(folder, torch.device("cuda"), max_rows=8)
    left, right = native.state_dict(), restored.state_dict()
    assert left.keys() == right.keys()
    assert all(
        torch.equal(left[name].view(torch.uint8), right[name].view(torch.uint8)) for name in left
    )
    del native, left, right
    native = restored
    weights = {
        name: layer.packed_weight.data_ptr()
        for name, layer in native.named_modules()
        if isinstance(layer, RowwiseFP8Linear)
    }
    assert len(weights) == len(set(weights.values())) == 8
    common = dict(
        attention_backend="triton",
        cache_config=CacheConfig(128, 4, compact_last_exited=compact),
        scheduler_config=SchedulerConfig(max_num_seqs=3, max_num_batched_tokens=8),
        exit_config=ExitConfig(mode="ouro_delayed"),
    )
    engines = [
        LLMEngine(reference, **common),
        LLMEngine(
            native,
            execution_config=ExecutionConfig(
                cuda_graphs=graphs,
                static_buffers=True,
                pad_to_power_of_two=True,
                async_scheduling=True,
                multi_stream=True,
            ),
            **common,
        ),
    ]
    tokens = torch.tensor([4, 7, 3, 9, 2, 8, 11], device="cuda")
    likelihood = [
        native_window_nll(engine.model, tokens, 8, engine.cache_manager) for engine in engines
    ]
    assert likelihood[0][1] == likelihood[1][1] == 6
    torch.testing.assert_close(likelihood[0][0], likelihood[1][0], atol=0.02, rtol=0.02)
    results = []
    for engine in engines:
        rounds = []
        for _ in range(2):
            for index, prompt in enumerate(([4, 7], [9, 3, 2], [8])):
                engine.add_request(
                    str(index),
                    prompt,
                    SamplingParams(
                        max_tokens=4,
                        min_loops=8,
                        max_loops=8,
                        ignore_eos=True,
                    ),
                )
            finished = {}
            while engine.has_unfinished_requests():
                for output in engine.step():
                    if output.finished:
                        finished[output.request_id] = (output.token_ids, output.exit_depths)
            assert len(finished) == 3 and engine.cache_manager.num_used_blocks == 0
            assert all(depths == [8] * 4 for _, depths in finished.values())
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
        compact=compact,
        asynchronous=True,
        encoded_reload_exact=True,
        physical_packed_weights=len(weights),
        repeated_requests=6,
        full_depth=8,
        quality_targets=6,
        native_nll=likelihood[1][0],
        reference_nll=likelihood[0][0],
        kv_after_drain=0,
        captures=graph.captures if graphs else 0,
        replays=graph.replays if graphs else 0,
    )
