"""Huginn INT4 export and native checks without a test-runner dependency."""

from pathlib import Path

import torch

from loopquant.huginn_int4 import core_projections, export_huginn_int4, load_huginn_int4
from loopquant.huginn_quality import native_huginn_window_nll
from loopquant.int4_backend import NativeInt4Linear
from loopquant.quantizers import int4_unpack
from vllm_rlt import CacheConfig, ExecutionConfig, SamplingParams, SchedulerConfig
from vllm_rlt.core.kv_cache_manager import KVCacheManager
from vllm_rlt.engine.llm_engine import LLMEngine
from vllm_rlt.models.huginn import HuginnConfig, HuginnForCausalLM


def tiny_int4_huginn(device: str = "cpu") -> HuginnForCausalLM:
    model = HuginnForCausalLM(
        HuginnConfig(
            n_embd=64,
            n_heads=2,
            n_layers=4,
            n_layers_in_prelude=1,
            n_layers_in_recurrent_block=2,
            n_layers_in_coda=1,
            intermediate_size=128,
            mean_recurrence=3,
            block_size=64,
            vocab_size=64,
            padded_vocab_size=64,
            bos_token_id=0,
            eos_token_id=1,
            pad_token_id=0,
        )
    )
    freqs = model.freqs_cis
    model.to(device=device, dtype=torch.bfloat16)
    model.freqs_cis = freqs.to(device)
    return model.eval().requires_grad_(False)


@torch.inference_mode()
def check_huginn_int4(folder: Path, graphs: bool) -> dict:
    torch.manual_seed(43)
    reference = tiny_int4_huginn("cuda")
    export_huginn_int4(reference, folder, model_revision="a" * 40)
    native, duplicate = [load_huginn_int4(folder, torch.device("cuda")) for _ in range(2)]
    assert native.lm_head.weight is native.transformer.wte.weight
    assert native.freqs_cis.dtype == torch.float32
    left, right = native.state_dict(), duplicate.state_dict()
    assert left.keys() == right.keys()
    assert all(
        torch.equal(left[name].view(torch.uint8), right[name].view(torch.uint8)) for name in left
    )
    del duplicate, left, right
    tensors = torch.load(folder / "tensors.pt", map_location="cuda", weights_only=True)
    for name, layer in core_projections(reference).items():
        layer.weight.copy_(
            int4_unpack(tensors["codes"][name], tensors["scales"][name].float(), layer.in_features)
        )
    del tensors
    weights = {
        name: layer.packed_weight.data_ptr()
        for name, layer in native.named_modules()
        if isinstance(layer, NativeInt4Linear)
    }
    assert len(weights) == len(set(weights.values())) == 8
    config = native.config
    cache = KVCacheManager(
        config.n_layers,
        config.n_heads,
        config.head_dim,
        128,
        4,
        3,
        device="cuda",
        dtype=torch.bfloat16,
        backend="triton",
        recurrent_layers=native.recurrent_kv_layers,
    )
    tokens = torch.tensor([4, 7, 9, 3, 12], device="cuda")
    cpu_rng, gpu_rng = torch.random.get_rng_state(), torch.cuda.get_rng_state()
    other_devices = [i for i in range(torch.cuda.device_count()) if i != tokens.device.index]
    other_rng = {i: torch.cuda.get_rng_state(i) for i in other_devices}
    actual = native_huginn_window_nll(native, tokens, 3, cache, seed=17)
    assert torch.equal(cpu_rng, torch.random.get_rng_state())
    assert torch.equal(gpu_rng, torch.cuda.get_rng_state())
    assert all(torch.equal(state, torch.cuda.get_rng_state(i)) for i, state in other_rng.items())
    assert actual == native_huginn_window_nll(native, tokens, 3, cache, seed=17)
    expected = native_huginn_window_nll(reference, tokens, 3, cache, seed=17)
    torch.testing.assert_close(
        torch.tensor(actual[0]), torch.tensor(expected[0]), atol=0.02, rtol=0.02
    )
    assert actual[1] == expected[1] == 4 and cache.num_used_blocks == 0
    common = dict(
        attention_backend="triton",
        cache_config=CacheConfig(256, 4),
        scheduler_config=SchedulerConfig(max_num_seqs=3, max_num_batched_tokens=8),
    )
    engines = [
        LLMEngine(reference, **common),
        LLMEngine(native, execution_config=ExecutionConfig(cuda_graphs=graphs), **common),
    ]
    rounds = []
    for engine in engines:
        observed = []
        for _ in range(2):
            torch.manual_seed(31)
            for i, prompt in enumerate(([4, 7], [9, 3, 2], [8])):
                engine.add_request(
                    str(i),
                    prompt,
                    SamplingParams(max_tokens=4, min_loops=3, max_loops=3, ignore_eos=True),
                )
            finished = {}
            while engine.has_unfinished_requests():
                for output in engine.step():
                    if output.finished:
                        finished[output.request_id] = (output.token_ids, output.exit_depths)
            assert len(finished) == 3 and engine.cache_manager.num_used_blocks == 0
            observed.append(finished)
        rounds.append(observed)
    assert rounds[0] == rounds[1]
    assert weights == {
        name: layer.packed_weight.data_ptr()
        for name, layer in native.named_modules()
        if isinstance(layer, NativeInt4Linear)
    }
    graph = engines[1].model_runner.graphs
    if graphs:
        assert graph.captures > 0 and graph.replays > graph.captures
    return dict(
        graphs=graphs,
        repeated_requests=6,
        physical_packed_weights=len(weights),
        export_reload_exact=True,
        tied_readout=True,
        fp32_rotary=True,
        rng_restored=True,
        quality_targets=4,
        kv_after_drain=0,
        captures=graph.captures if graphs else 0,
        replays=graph.replays if graphs else 0,
    )
