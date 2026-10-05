import pytest
import torch
from torch.nn import functional as F

from loopquant.adapters.hrm_text import HrmTextAdapter
from loopquant.quality import next_token_nll
from loopquant.quantizers import FP8FakeLinear, ScaleLayout
from loopquant.train import Q0Config, Q0Trainer, TrainingBatch
from tests.reference.hrm_text import dense_hrm_reference
from vllm_rlt.models.hrm_text import HrmTextConfig, HrmTextForCausalLM


def make_model(prefix_lm=True):
    torch.manual_seed(71)
    return (
        HrmTextForCausalLM(
            HrmTextConfig(
                vocab_size=32,
                n_embd=16,
                intermediate_size=24,
                module_layers=2,
                num_attention_heads=2,
                num_key_value_heads=2,
                head_dim=8,
                H_cycles=2,
                L_cycles=3,
                max_position_embeddings=64,
                embedding_scale=4,
                prefix_lm=prefix_lm,
            )
        )
        .eval()
        .requires_grad_(False)
    )


@pytest.mark.parametrize("backend", ["eager", "sdpa"])
@pytest.mark.parametrize("prefix_lm", [False, True])
def test_hrm_dense_h_l_states_ragged_masks_and_explicit_statistics(backend, prefix_lm):
    model = make_model(prefix_lm)
    adapter = HrmTextAdapter(model, attention_backend=backend)
    adapter.collect = True
    tokens = torch.tensor([[3, 7, 4, 2, 6, 8, 9], [9, 5, 3, 2, 31, 31, 31]])
    valid = torch.arange(7) < torch.tensor([7, 4])[:, None]
    prefixes = torch.tensor([3, 2])
    actual = adapter(tokens, valid, 2, prefix_lengths=prefixes)
    for row, length in enumerate([7, 4]):
        states, logits, _ = dense_hrm_reference(model, tokens[row, :length], int(prefixes[row]))
        for state, expected in zip(actual.states, states, strict=True):
            torch.testing.assert_close(state[row, :length], expected, atol=3e-6, rtol=3e-5)
        torch.testing.assert_close(actual.logits[row, :length], logits, atol=3e-6, rtol=3e-5)
    assert len(adapter.statistics) == 2 * 2 * 4 * 2
    for (name, _loop), stats in adapter.statistics.items():
        assert stats.rows == 11 * (3 if ".L_module." in name else 1)


@pytest.mark.parametrize("backend", ["eager", "sdpa"])
def test_hrm_prefix_sees_its_prompt_but_never_future_continuation(backend):
    adapter = HrmTextAdapter(make_model(), attention_backend=backend)
    tokens = torch.tensor([[3, 7, 4, 2, 6, 8, 9]])
    valid, prefixes = torch.ones_like(tokens, dtype=torch.bool), torch.tensor([3])
    original = adapter(tokens, valid, 2, prefix_lengths=prefixes).logits
    future = tokens.clone()
    future[:, 5:] = torch.tensor([10, 11])
    changed = adapter(future, valid, 2, prefix_lengths=prefixes).logits
    assert torch.equal(original[:, :5], changed[:, :5])
    prefix = tokens.clone()
    prefix[:, 1] = 12
    changed = adapter(prefix, valid, 2, prefix_lengths=prefixes).logits
    assert not torch.equal(original[:, 0], changed[:, 0])
    with pytest.raises(ValueError, match="explicit valid prefix"):
        adapter(tokens, valid, 2)
    with pytest.raises(ValueError, match="complete H_cycles"):
        adapter(tokens, valid, 1, prefix_lengths=prefixes)


def test_prefix_target_mask_scores_first_continuation_and_excludes_prompt_and_padding():
    tokens = torch.tensor([[3, 7, 4, 2, 6, 8, 9], [9, 5, 3, 2, 31, 31, 31]])
    valid = torch.arange(7) < torch.tensor([7, 4])[:, None]
    batch = TrainingBatch(tokens, valid, torch.tensor([3, 2]))
    logits = torch.randn(2, 7, 32)
    loss, count = next_token_nll(logits, tokens, valid, target_mask=batch.target_mask)
    expected = F.cross_entropy(logits[0, 2:6], tokens[0, 3:7], reduction="sum")
    expected += F.cross_entropy(logits[1, 1:3], tokens[1, 2:4], reduction="sum")
    assert count == 6
    torch.testing.assert_close(loss, expected)
    altered = logits.clone()
    altered[0, :2] = 1000 * torch.randn(2, 32)
    altered[1, 0] = 1000 * torch.randn(32)
    altered[1, 3:] = 1000 * torch.randn(4, 32)
    changed, _ = next_token_nll(altered, tokens, valid, target_mask=batch.target_mask)
    torch.testing.assert_close(changed, loss, rtol=0, atol=0)


def test_hrm_q0_all_inner_updates_receive_gradients_and_only_continuation_is_supervised():
    model = make_model()
    student = HrmTextAdapter(model)
    name = "model.L_module.layers.0.attn.gqkv_proj"
    quantized = FP8FakeLinear(model.get_submodule(name), torch.tensor([0.02]), ScaleLayout(2))
    student.attach(name, quantized)
    tokens = torch.tensor([[3, 7, 4, 2, 6, 8, 9]])
    batch = TrainingBatch(tokens, torch.ones_like(tokens, dtype=torch.bool), torch.tensor([3]))
    projected = []

    def capture(_module, _inputs, output):
        output.retain_grad()
        projected.append(output)

    hook = quantized.register_forward_hook(capture)
    output = student(tokens, batch.valid, 2, prefix_lengths=batch.prefix_lengths)
    output.logits[:, -1].square().mean().backward()
    hook.remove()
    assert len(projected) == 6
    assert all(value.grad is not None and value.grad.norm() > 0 for value in projected)
    frozen = {name: value.clone() for name, value in model.state_dict().items()}
    pointer = quantized.packed_weight.data_ptr()
    trainer = Q0Trainer(
        student,
        Q0Config(
            loops=2,
            tokens_per_update=7,
            kl_weight=0.1,
            trajectory_weight=0.01,
        ),
    )
    assert isinstance(trainer.teacher, HrmTextAdapter)
    for _ in range(2):
        receipt = trainer.step([batch])
        assert receipt["input_tokens"] == 7 and receipt["supervised_targets"] == 4
        assert receipt["gradient_norm"] > 0 and sum(receipt["scale_update_norms"].values()) > 0
        assert quantized.packed_weight.data_ptr() == pointer
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in frozen.items())
