"""Reusable native INT4 arithmetic qualification without test-runner dependencies."""

from pathlib import Path

import torch

from loopquant.int4_backend import NativeInt4Linear
from loopquant.quantizers import int4_pack, int4_unpack


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


def check_int4_native_model(family: str, graphs: bool, export_folder: Path) -> dict:
    from loopquant.int4_export import export_int4_model, load_int4_model
    from tests.helpers import tiny_nanbeige_config, tiny_ouro_config
    from vllm_rlt import (
        LLM,
        CacheConfig,
        ExecutionConfig,
        ExitConfig,
        SamplingParams,
        SchedulerConfig,
    )
    from vllm_rlt.models.nanbeige import NanbeigeForCausalLM
    from vllm_rlt.models.ouro import OuroForCausalLM

    torch.manual_seed(43)
    if family == "ouro":
        config = tiny_ouro_config(hidden_size=64, head_dim=32, intermediate_size=128)
        reference = OuroForCausalLM(config)
    elif family == "nanbeige":
        config = tiny_nanbeige_config(hidden_size=64, head_dim=32, intermediate_size=256)
        reference = NanbeigeForCausalLM(config)
    else:
        raise ValueError("unknown INT4 model family")
    reference = reference.to(device="cuda", dtype=torch.bfloat16)
    export_int4_model(reference, export_folder, model_revision="a" * 40)
    native = load_int4_model(export_folder, torch.device("cuda"))
    restored = load_int4_model(export_folder, torch.device("cuda"))
    first, second = native.state_dict(), restored.state_dict()
    assert first.keys() == second.keys()
    assert all(torch.equal(first[name], second[name]) for name in first)
    del restored, first, second
    tensors = torch.load(export_folder / "tensors.pt", map_location="cuda", weights_only=True)
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
    assert len(weights) == len(set(weights.values())) == config.num_hidden_layers * 7
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
    params = [
        SamplingParams(max_tokens=4, min_loops=n, max_loops=n, ignore_eos=True)
        for n in [1, config.total_ut_steps, 2]
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
        graphs=graphs,
        repeated_requests=6,
        physical_packed_weights=len(weights),
        export_roundtrip=True,
        captures=actual.engine.model_runner.graphs.captures if graphs else 0,
        replays=actual.engine.model_runner.graphs.replays if graphs else 0,
    )
