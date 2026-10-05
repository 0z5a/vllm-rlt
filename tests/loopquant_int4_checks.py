"""Reusable native INT4 arithmetic qualification without test-runner dependencies."""

from pathlib import Path

import torch

from loopquant.int4_backend import NativeInt4Linear
from loopquant.int4_export import Int4Model
from loopquant.quantizers import int4_pack, int4_unpack


def int4_reference_model(family: str) -> Int4Model:
    from tests.helpers import tiny_nanbeige_config, tiny_ouro_config
    from vllm_rlt.models.hrm_text import HrmTextConfig, HrmTextForCausalLM
    from vllm_rlt.models.loopformer import LoopFormerConfig, LoopFormerForCausalLM
    from vllm_rlt.models.nanbeige import NanbeigeForCausalLM
    from vllm_rlt.models.ouro import OuroForCausalLM

    if family == "ouro":
        return OuroForCausalLM(tiny_ouro_config(hidden_size=64, head_dim=32, intermediate_size=128))
    if family == "nanbeige":
        return NanbeigeForCausalLM(
            tiny_nanbeige_config(hidden_size=64, head_dim=32, intermediate_size=256)
        )
    if family == "hrm_text":
        return HrmTextForCausalLM(
            HrmTextConfig(
                vocab_size=32,
                n_embd=64,
                intermediate_size=128,
                module_layers=2,
                num_attention_heads=2,
                num_key_value_heads=2,
                head_dim=32,
                max_position_embeddings=64,
                embedding_scale=4,
            )
        )
    if family == "loopformer":
        return LoopFormerForCausalLM(
            LoopFormerConfig(
                vocab_size=32,
                n_embd=64,
                intermediate_dim=128,
                n_layer=2,
                n_head=2,
                block_size=64,
                bos_token_id=1,
                eos_token_id=2,
                pad_token_id=0,
            )
        )
    raise ValueError("unknown INT4 model family")


def check_native_int4(rows: int, columns: int, outputs: int) -> dict:
    torch.manual_seed(19)
    weight = torch.randn(outputs, columns, device="cuda") * 0.03
    packed, scales = int4_pack(weight)
    native = NativeInt4Linear(packed, scales, columns)
    values = torch.randn(rows, columns, device="cuda", dtype=torch.bfloat16)
    reference = int4_unpack(packed, scales.bfloat16().float(), columns).bfloat16()
    expected = torch.nn.functional.linear(values, reference)
    actual = native(values)
    torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.02)
    assert native.packed_weight.numel() * 4 == packed.numel()
    return dict(
        rows=rows,
        columns=columns,
        outputs=outputs,
        max_abs=float((actual.float() - expected.float()).abs().max()),
        packed_bytes=packed.numel(),
        padded_columns=native.padded_columns,
    )


def check_int4_native_model(
    family: str, graphs: bool, export_folder: Path, *, compact: bool = False
) -> dict:
    from loopquant.int4_export import core_projections, export_int4_model, load_int4_model
    from vllm_rlt import (
        LLM,
        CacheConfig,
        ExecutionConfig,
        ExitConfig,
        SamplingParams,
        SchedulerConfig,
    )

    torch.manual_seed(43)
    reference = int4_reference_model(family).to(device="cuda", dtype=torch.bfloat16)
    config = reference.config
    physical_projections = len(core_projections(reference))
    export_int4_model(reference, export_folder, model_revision="a" * 40)
    native = load_int4_model(export_folder, torch.device("cuda"))
    restored = load_int4_model(export_folder, torch.device("cuda"))
    first, second = native.state_dict(), restored.state_dict()
    assert first.keys() == second.keys()
    assert all(torch.equal(first[name], second[name]) for name in first)
    del restored, first, second
    tensors = torch.load(export_folder / "tensors.pt", map_location="cuda", weights_only=True)
    protected_names = set(tensors["protected"])
    for name, codes in tensors["codes"].items():
        layer = reference.get_submodule(name)
        layer.weight.data.copy_(
            int4_unpack(codes, tensors["scales"][name].float(), layer.in_features)
        )
    del tensors
    weights = {
        name: layer.packed_weight.data_ptr()
        for name, layer in native.named_modules()
        if isinstance(layer, NativeInt4Linear)
    }
    assert len(weights) == len(set(weights.values())) == physical_projections
    common = dict(
        cache_config=CacheConfig(64, 16, compact_last_exited=compact),
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
    depths = (
        [config.total_ut_steps] * 3
        if family in ("loopformer", "hrm_text")
        else [1, config.total_ut_steps, 2]
    )
    params = [
        SamplingParams(max_tokens=4, min_loops=n, max_loops=n, ignore_eos=True) for n in depths
    ]
    for _ in range(2):
        before = expected.generate([[5, 7], [3, 9, 2], [8]], params)
        after = actual.generate([[5, 7], [3, 9, 2], [8]], params)
        assert [(row.token_ids, row.exit_depths) for row in before] == [
            (row.token_ids, row.exit_depths) for row in after
        ]
        assert actual.engine.cache_manager.num_used_blocks == 0
    assert weights == {
        name: layer.packed_weight.data_ptr()
        for name, layer in native.named_modules()
        if isinstance(layer, NativeInt4Linear)
    }
    assert set(dict(native.named_parameters())) == protected_names
    if graphs:
        assert (
            actual.engine.model_runner.graphs.replays > actual.engine.model_runner.graphs.captures
        )
    return dict(
        model_family=family,
        cache_storage="compact" if compact else "native",
        graphs=graphs,
        repeated_requests=6,
        physical_packed_weights=len(weights),
        export_roundtrip=True,
        captures=actual.engine.model_runner.graphs.captures if graphs else 0,
        replays=actual.engine.model_runner.graphs.replays if graphs else 0,
    )
