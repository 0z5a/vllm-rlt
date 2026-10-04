"""Numerical/statistical gates that must hold before official checkpoint runs."""

import math

import pytest
import torch
from torch import nn

from loopquant.data import Window, validate_windows
from loopquant.quality import DocumentNLL, next_token_nll, ppl_interval
from loopquant.quantizers import FP8FakeLinear, ScaleLayout, fp8_encode, int4_pack, int4_unpack
from loopquant.report import TrialPair, paired_speedup
from loopquant.stats import ActivationStats


def test_fp8_saturation_and_scale_convention():
    values = torch.tensor([-1000.0, -4.0, 0.0, 4.0, 1000.0])
    encoded = fp8_encode(values, torch.tensor(2.0)).float()
    torch.testing.assert_close(encoded, torch.tensor([-448.0, -2.0, 0.0, 2.0, 448.0]))


@pytest.mark.parametrize("columns", [1, 7, 128, 133])
def test_int4_sign_tail_and_error_bound(columns):
    torch.manual_seed(17)
    weight = torch.randn(3, columns)
    packed, scale = int4_pack(weight, group_size=8)
    restored = int4_unpack(packed, scale, columns)
    bounds = scale.repeat_interleave(8, dim=1)[:, :columns] * 0.501
    assert torch.all((restored - weight).abs() <= bounds)
    assert packed.numel() == 3 * math.ceil(columns / 8) * 4


def test_zero_int4_is_finite():
    packed, scale = int4_pack(torch.zeros(3, 129))
    assert torch.isfinite(scale).all() and (scale > 0).all()
    assert torch.count_nonzero(int4_unpack(packed, scale, 129)) == 0


def test_stages_use_absolute_loops_with_reordered_rows():
    layout = ScaleLayout(32, (8, 16))
    actual = layout.indices(torch.tensor([31, 0, 8, 7, 16, 15]))
    assert actual.tolist() == [2, 0, 1, 0, 2, 1]


def test_q0_shared_storage_and_scale_gradients():
    torch.manual_seed(17)
    linear = nn.Linear(16, 8, bias=False)
    module = FP8FakeLinear(linear, torch.tensor([0.001, 0.005]), ScaleLayout(4, (2,)))
    storage = module.packed_weight.data_ptr()
    original = module.packed_weight.float().clone()
    optimizer = torch.optim.SGD([module.log_scale], lr=0.01)
    for _ in range(2):
        before = module.log_scale.detach().clone()
        output = module(torch.randn(4, 16), torch.tensor([0, 3, 1, 2]))
        output.square().mean().backward()
        assert torch.isfinite(module.log_scale.grad).all()
        assert torch.count_nonzero(module.log_scale.grad) == 2
        optimizer.step()
        optimizer.zero_grad()
        assert torch.count_nonzero(module.log_scale.detach() - before) == 2
        assert module.packed_weight.data_ptr() == storage
        torch.testing.assert_close(module.packed_weight.float(), original, rtol=0, atol=0)
    assert list(dict(module.named_parameters())) == ["log_scale"]


def test_padding_does_not_enter_activation_statistics():
    stats = ActivationStats(sample_limit=4)
    stats.update(torch.tensor([[3.0, 4.0], [1e10, 1e10]]), torch.tensor([True, False]))
    stats.update(torch.tensor([[0.0, 0.0]]), torch.tensor([True]))
    result = stats.summary()
    assert result["rms"] == 2.5 and result["amax"] == 4.0
    assert result["rows"] == 2 and result["elements"] == 4
    assert result["channel_energy"] == [4.5, 8.0]


def test_reservoir_is_reproducible_and_bounded():
    a, b = ActivationStats(sample_limit=5), ActivationStats(sample_limit=5)
    for i in range(10):
        x = torch.arange(i * 12, (i + 1) * 12).reshape(3, 4)
        for stats in (a, b):
            stats.update(x, torch.ones(3, dtype=torch.bool))
    assert a.samples.numel() == 5
    torch.testing.assert_close(a.samples, b.samples)


def test_next_token_shift_and_padding():
    ids = torch.tensor([[0, 1, 2, 0]])
    logits = torch.full((1, 4, 3), -20.0)
    logits[0, 0, 1], logits[0, 1, 2] = 20, 20
    nll, tokens = next_token_nll(logits, ids, torch.tensor([[True, True, True, False]]))
    assert tokens == 2 and float(nll) < 1e-5


def make_pairs(ratios):
    return [TrialPair(str(i), 100.0, 100.0 * ratio, 128, 128) for i, ratio in enumerate(ratios)]


@pytest.mark.parametrize("n,index", [(5, 0), (10, 1), (20, 5)])
def test_exact_one_sided_order_statistic(n, index):
    ratios = [1 + i / 100 for i in range(n)]
    result = paired_speedup(make_pairs(ratios))
    assert result.lower_bound == pytest.approx(ratios[index])
    assert result.coverage >= 0.95


def test_small_pilot_has_no_formal_bound():
    assert paired_speedup(make_pairs([1.5] * 4)).lower_bound is None


def test_median_ratio_is_not_ratio_of_medians():
    pairs = [
        TrialPair(str(i), base, candidate, 128, 128)
        for i, (base, candidate) in enumerate(
            [(1, 2), (10, 10), (100, 50), (1000, 4000), (10000, 10000)]
        )
    ]
    assert paired_speedup(pairs).median_ratio == 1


@pytest.mark.parametrize(
    "patch",
    [
        {"failures": 1},
        {"captures_during_measurement": 1},
        {"candidate_tokens": 127},
        {"candidate_tokens_s": float("nan")},
    ],
)
def test_invalid_trial_cannot_make_a_speed_claim(patch):
    fields = dict(
        trial_id="a",
        baseline_tokens_s=100,
        candidate_tokens_s=200,
        baseline_tokens=128,
        candidate_tokens=128,
    )
    fields.update(patch)
    with pytest.raises(ValueError):
        paired_speedup([TrialPair(**fields)])


def test_ppl_uses_token_weighted_nll():
    ratio, upper = ppl_interval(
        [DocumentNLL("a", 1, 1, 2), DocumentNLL("b", 99, 99, 99)], resamples=1000
    )
    assert ratio == pytest.approx(math.exp(0.01))
    assert upper >= ratio


def test_document_leakage_rejected_before_calibration():
    def window(split):
        return Window("doc", "a" * 40, split, 0, (1, 2), "b" * 64)

    validate_windows([window("calibration"), window("qat_train")])
    with pytest.raises(ValueError, match="crosses"):
        validate_windows([window("calibration"), window("dev")])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_torch_only_loader_matches_safetensors(tmp_path, dtype):
    from safetensors.torch import save_file

    from loopquant.checkpoint import load_weights

    original = nn.Sequential(nn.Linear(8, 4), nn.Linear(4, 2)).to(dtype)
    save_file(original.state_dict(), tmp_path / "model.safetensors")
    with torch.device("meta"):
        loaded = nn.Sequential(nn.Linear(8, 4), nn.Linear(4, 2))
    load_weights(loaded, tmp_path, torch.device("cpu"), dtype)
    for key, value in original.state_dict().items():
        torch.testing.assert_close(value, loaded.state_dict()[key], atol=0, rtol=0)
    # The mapping has already closed; parameters own independent writable storage.
    with torch.no_grad():
        loaded[0].weight.add_(1)
    assert not torch.equal(loaded[0].weight, original[0].weight)


@pytest.mark.parametrize("loops", [1, 2, 3, 4])
@pytest.mark.parametrize("attention_backend", ["eager", "sdpa"])
def test_ouro_dense_adapter_matches_independent_oracle(loops, attention_backend):
    from loopquant.adapters.ouro import OuroAdapter
    from tests.helpers import tiny_ouro_config
    from tests.reference import dense_reference
    from vllm_rlt.models.ouro import OuroForCausalLM

    torch.manual_seed(71)
    model = OuroForCausalLM(tiny_ouro_config())
    adapter = OuroAdapter(model, attention_backend=attention_backend)
    tokens = torch.tensor([[5, 7, 2, 11], [3, 9, 6, 0]])
    valid = torch.tensor([[True, True, True, True], [True, True, True, False]])
    actual = adapter(tokens, valid, loops)
    for row, count in [(0, 4), (1, 3)]:
        reference = dense_reference(model, tokens[row, :count], loops)
        for loop, (state, _, logits) in enumerate(reference):
            torch.testing.assert_close(
                actual.states[loop][row, :count], state, atol=3e-6, rtol=3e-5
            )
        torch.testing.assert_close(actual.logits[row, :count], logits, atol=3e-6, rtol=3e-5)


def test_ouro_q0_full_recurrent_gradient_and_explicit_statistics():
    from loopquant.adapters.ouro import OuroAdapter
    from tests.helpers import tiny_ouro_config
    from vllm_rlt.models.ouro import OuroForCausalLM

    torch.manual_seed(71)
    model = OuroForCausalLM(tiny_ouro_config())
    adapter = OuroAdapter(model)
    name = "model.layers.0.self_attn.q_proj"
    layer = model.model.layers[0].self_attn.q_proj
    adapter.attach(name, FP8FakeLinear(layer, torch.tensor([0.01]), ScaleLayout(4)))
    adapter.collect = True
    tokens = torch.tensor([[5, 7, 2, 11]])
    valid = torch.ones_like(tokens, dtype=torch.bool)
    output = adapter(tokens, valid, 4)
    nll, count = next_token_nll(output.logits, tokens, valid)
    (nll / count).backward()
    gradient = adapter.quantized[name.replace(".", "__")].log_scale.grad
    assert torch.isfinite(gradient).all() and torch.count_nonzero(gradient) == 1
    assert all(parameter.grad is None for parameter in model.parameters())
    assert all(state.requires_grad for state in output.states)
    assert {loop for module, loop in adapter.statistics if module == name} == {0, 1, 2, 3}
    assert adapter.statistics[name, 3].rows == 4


@pytest.mark.parametrize("reference_dtype", [torch.float32, torch.float64])
def test_official_operator_order_reference_and_equalization(reference_dtype):
    from loopquant.adapters.ouro import OuroAdapter
    from loopquant.calibrate import fold_equalization
    from loopquant.reference import ouro_reference
    from tests.helpers import tiny_ouro_config
    from vllm_rlt.models.ouro import OuroForCausalLM

    torch.manual_seed(71)
    model = OuroForCausalLM(tiny_ouro_config())
    tokens = torch.tensor([[5, 7, 2, 11]])
    valid = torch.ones_like(tokens, dtype=torch.bool)
    adapter = OuroAdapter(model)
    before = adapter(tokens, valid, 4).logits
    reference = ouro_reference(model, tokens[0], 4, reference_dtype=reference_dtype)[-1][2]
    torch.testing.assert_close(before[0].to(reference_dtype), reference, atol=3e-6, rtol=3e-5)
    layer = model.model.layers[0]
    attention = layer.self_attn
    fold_equalization(
        layer.input_layernorm,
        [attention.q_proj, attention.k_proj, attention.v_proj],
        torch.linspace(0.5, 2.0, model.config.hidden_size),
    )
    torch.testing.assert_close(adapter(tokens, valid, 4).logits, before, atol=3e-6, rtol=3e-5)


def test_bf16_adapter_uses_official_eager_reduction_precision():
    from loopquant.adapters.ouro import OuroAdapter
    from loopquant.reference import ouro_reference
    from tests.helpers import tiny_ouro_config
    from vllm_rlt.models.ouro import OuroForCausalLM

    torch.manual_seed(71)
    model = OuroForCausalLM(tiny_ouro_config()).to(torch.bfloat16)
    tokens = torch.tensor([[5, 7, 2, 11]])
    result = OuroAdapter(model)(tokens, torch.ones_like(tokens, dtype=torch.bool), 4)
    reference = ouro_reference(model, tokens[0], 4, reference_dtype=torch.bfloat16)
    for actual, (expected, _, _) in zip(result.states, reference, strict=True):
        torch.testing.assert_close(actual[0], expected, atol=0, rtol=0)
    torch.testing.assert_close(result.logits[0], reference[-1][2], atol=0, rtol=0)


def test_engine_cohort_refill_drain_and_fixed_work():
    from loopquant.bench import run_cohort
    from tests.helpers import tiny_ouro_config
    from vllm_rlt import LLM, CacheConfig, SchedulerConfig
    from vllm_rlt.models.ouro import OuroForCausalLM

    model = OuroForCausalLM(tiny_ouro_config())
    llm = LLM(
        model,
        cache_config=CacheConfig(num_blocks=64, block_size=2),
        scheduler_config=SchedulerConfig(max_num_seqs=2, max_num_batched_tokens=8),
    )
    records = []
    result = run_cohort(
        llm.engine,
        [[5, 7], [3, 9], [6, 8], [2, 3], [5, 4]],
        concurrency=2,
        output_tokens=3,
        loops=4,
        records=records,
    )
    assert result.completed_requests == 5 and result.output_tokens == 15
    assert result.failed_requests == 0 and result.kv_used_after_drain == 0
    assert result.loop_histogram == {4: 15}
    assert result.output_tokens_s == 15 / result.elapsed_s
    assert records[2].entered_s >= min(row.completed_s for row in records[:2])
    assert all(
        row.entered_s <= row.first_token_s <= row.completed_s <= result.elapsed_s for row in records
    )


def test_export_preserves_encoded_model_without_fp_weight_duplicates(tmp_path):
    from loopquant.adapters.ouro import OuroAdapter
    from loopquant.export import export_ouro, load_exported_ouro
    from tests.helpers import tiny_ouro_config
    from vllm_rlt.models.ouro import OuroForCausalLM

    model = OuroForCausalLM(tiny_ouro_config()).to(torch.bfloat16)
    adapter = OuroAdapter(model)
    for name, module in model.named_modules():
        if name.startswith("model.layers.") and isinstance(module, nn.Linear):
            adapter.attach(name, FP8FakeLinear(module, torch.tensor([0.01]), ScaleLayout(4)))
    destination = tmp_path / "export"
    export_ouro(adapter, destination, model_revision="a" * 40)
    payload = torch.load(destination / "tensors.pt", weights_only=True)
    name = "model.layers.0.self_attn.q_proj"
    module = adapter.quantized[name.replace(".", "__")]
    values = torch.randn(5, model.config.hidden_size, dtype=torch.bfloat16)
    scale = payload["activation_scales"][name]
    activation = fp8_encode(values, scale).float() * scale
    weight = payload["packed"][name].t().float() * payload["weight_scales"][name]
    expected = nn.functional.linear(activation, weight).to(values.dtype)
    torch.testing.assert_close(
        module(values, torch.zeros(5, dtype=torch.long)), expected, atol=0, rtol=0
    )
    assert name + ".weight" not in payload["protected"]
    assert payload["protected"]["lm_head.weight"].dtype == torch.bfloat16
    with pytest.raises(ValueError, match="CUDA"):
        load_exported_ouro(destination, torch.device("cpu"))


def test_q0_two_updates_and_exact_optimizer_resumption(tmp_path):
    from loopquant.adapters.ouro import OuroAdapter
    from loopquant.train import Q0Config, Q0Trainer, TrainingBatch
    from tests.helpers import tiny_ouro_config
    from vllm_rlt.models.ouro import OuroForCausalLM

    torch.manual_seed(17)
    model = OuroForCausalLM(tiny_ouro_config())
    name = "model.layers.0.self_attn.q_proj"

    def trainer():
        adapter = OuroAdapter(model)
        adapter.attach(
            name,
            FP8FakeLinear(
                model.model.layers[0].self_attn.q_proj, torch.tensor([0.001]), ScaleLayout(4)
            ),
        )
        return Q0Trainer(
            adapter,
            Q0Config(tokens_per_update=8, kl_weight=0.1, trajectory_weight=0.1),
        )

    continuous = trainer()
    initial = {name: value.clone() for name, value in model.state_dict().items()}
    batches = [
        TrainingBatch(torch.tensor([[5, 7, 2, 11]]), torch.ones(1, 4, dtype=torch.bool)),
        TrainingBatch(torch.tensor([[3, 9, 6, 8]]), torch.ones(1, 4, dtype=torch.bool)),
    ]
    first = continuous.step(batches)
    assert first["input_tokens"] == 8 and first["supervised_targets"] == 6
    assert first["gradient_norm"] > 0
    path = tmp_path / "qat.pt"
    continuous.save(path)
    expected_rng = torch.rand(3)
    expected = continuous.step(batches)
    resumed = trainer()
    resumed.resume(path)
    torch.testing.assert_close(torch.rand(3), expected_rng, atol=0, rtol=0)
    actual = resumed.step(batches)
    assert actual == expected
    assert resumed.data_position == 4 and resumed.cumulative_tokens == 16
    for key, value in continuous.student.quantized.state_dict().items():
        torch.testing.assert_close(
            value.float(), resumed.student.quantized.state_dict()[key].float(), atol=0, rtol=0
        )
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, initial[key], atol=0, rtol=0)
    with pytest.raises(ValueError, match="token budget"):
        resumed.step(batches[:1])


def test_gptq_diagonal_matches_rtn_and_blocked_compensation():
    from loopquant.gptq import gptq_int4

    torch.manual_seed(17)
    weight = torch.randn(5, 13)
    diagonal = torch.diag(torch.rand(13) + 1)
    packed, scales = gptq_int4(weight, diagonal, group_size=8, block_size=3)
    expected, expected_scales = int4_pack(weight, group_size=8)
    torch.testing.assert_close(packed, expected, atol=0, rtol=0)
    torch.testing.assert_close(scales, expected_scales, atol=0, rtol=0)
    values = torch.randn(100, 13)
    values[:, 1] = values[:, 0] * 0.99 + values[:, 1] * 0.01
    hessian = values.T @ values / 50
    one, scales_one = gptq_int4(weight, hessian, group_size=8, block_size=1)
    blocked, scales_blocked = gptq_int4(weight, hessian, group_size=8, block_size=4)
    torch.testing.assert_close(one, blocked, atol=0, rtol=0)
    torch.testing.assert_close(scales_one, scales_blocked, atol=0, rtol=0)
    error = ((weight - int4_unpack(one, scales_one, 13)) @ values.T).square().sum()
    rtn_error = ((weight - int4_unpack(expected, expected_scales, 13)) @ values.T).square().sum()
    assert error < rtn_error


def test_hessian_counts_all_loops_and_excludes_padding():
    from loopquant.gptq import HessianStats

    stats = HessianStats.create(2, torch.device("cpu"))
    values = torch.tensor([[[1.0, 2.0], [1000.0, 1000.0]]])
    stats.add(values, torch.tensor([[True, False]]), 0)
    stats.add(values * 2, torch.tensor([[True, False]]), 3)
    expected = torch.tensor([[1.0, 2.0], [2.0, 4.0]]) * 5
    torch.testing.assert_close(stats.matrix(), expected, atol=0, rtol=0)
    assert stats.loop_rows == {0: 1, 3: 1} and stats.loop_energy == {0: 5, 3: 20}


@pytest.mark.gpu
def test_native_fp8_matches_same_quantized_model_and_graph():
    from loopquant.backends import NativeFP8Linear

    if not torch.cuda.is_available():
        pytest.skip("requires a reserved CUDA device")
    torch.manual_seed(17)
    linear = nn.Linear(64, 32, bias=False, device="cuda", dtype=torch.bfloat16)
    module = NativeFP8Linear(linear, torch.tensor(0.01, device="cuda"))
    fake = FP8FakeLinear(linear, torch.tensor([0.01], device="cuda"), ScaleLayout(4))
    values = torch.randn(17, 64, device="cuda", dtype=torch.bfloat16)
    storage = module.packed_weight.data_ptr()
    warmup = torch.cuda.Stream()
    warmup.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup):
        for _ in range(3):
            module(values)
    torch.cuda.current_stream().wait_stream(warmup)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = module(values)
    for scale in (0.01, 0.02):
        module.activation_scale.fill_(scale)
        with torch.no_grad():
            fake.log_scale.fill_(math.log(scale))
        graph.replay()
        torch.testing.assert_close(output, module.reference(values), atol=0.02, rtol=0.02)
        torch.testing.assert_close(
            output,
            fake(values, torch.zeros(len(values), dtype=torch.long, device="cuda")),
            atol=0.02,
            rtol=0.02,
        )
        assert module.packed_weight.data_ptr() == storage
