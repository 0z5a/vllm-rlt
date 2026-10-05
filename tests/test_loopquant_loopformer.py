import pytest
import torch

from loopquant.adapters.loopformer import LoopFormerAdapter
from loopquant.quantizers import FP8FakeLinear, ScaleLayout
from loopquant.train import Q0Config, Q0Trainer, TrainingBatch
from tests.reference.loopformer import dense_loopformer_reference
from vllm_rlt.models.loopformer import LoopFormerConfig, LoopFormerForCausalLM


def make_model():
    torch.manual_seed(92)
    return (
        LoopFormerForCausalLM(
            LoopFormerConfig(vocab_size=32, n_embd=16, n_head=2, n_layer=2, intermediate_dim=24)
        )
        .requires_grad_(False)
        .eval()
    )


@pytest.mark.parametrize("backend", ["eager", "sdpa"])
def test_loopformer_clock_positions_and_masked_rows(backend):
    model = make_model()
    adapter = LoopFormerAdapter(model, attention_backend=backend)
    adapter.collect = True
    tokens = torch.tensor([[3, 7, 2, 9], [8, 4, 0, 0]])
    valid = torch.tensor([[True, True, True, True], [True, True, False, False]])
    actual = adapter(tokens, valid, 8)
    for row, length in enumerate([4, 2]):
        states, logits, _ = dense_loopformer_reference(model, tokens[row, :length])
        for got, expected in zip(actual.states, states, strict=True):
            torch.testing.assert_close(got[row, :length], expected, atol=3e-6, rtol=3e-5)
        torch.testing.assert_close(actual.logits[row, :length], logits, atol=3e-6, rtol=3e-5)
    assert len(adapter.statistics) == 2 * 4 * 8
    assert all(stats.rows == 6 for stats in adapter.statistics.values())
    for loop, state in enumerate(actual.states):
        assert torch.equal(state[..., -1], torch.full((2, 4), (loop + 1) / 8))
    with pytest.raises(ValueError, match="eight steps"):
        adapter(tokens, valid, 4)


def test_loopformer_all_step_gradients_and_protected_parameters():
    model = make_model()
    student = LoopFormerAdapter(model)
    name = "gpt.transformer.h.blocks.0.attn.c_attn"
    quantized = FP8FakeLinear(model.get_submodule(name), torch.tensor([0.002]), ScaleLayout(8))
    student.attach(name, quantized)
    batch = TrainingBatch(torch.tensor([[3, 7, 2, 9]]), torch.ones(1, 4, dtype=torch.bool))
    projected = []

    def capture(_module, _inputs, values):
        values.retain_grad()
        projected.append(values)

    hook = quantized.register_forward_hook(capture)
    student(batch.token_ids, batch.valid, 8).logits.square().mean().backward()
    hook.remove()
    assert len(projected) == 8
    assert all(values.grad is not None and values.grad.norm() > 0 for values in projected)
    before = {name: value.clone() for name, value in model.state_dict().items()}
    pointer = quantized.packed_weight.data_ptr()
    trainer = Q0Trainer(student, Q0Config(loops=8, tokens_per_update=4, kl_weight=0.1))
    assert isinstance(trainer.teacher, LoopFormerAdapter)
    for _ in range(2):
        receipt = trainer.step([batch])
        assert receipt["gradient_norm"] > 0 and sum(receipt["scale_update_norms"].values()) > 0
        assert quantized.packed_weight.data_ptr() == pointer
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in before.items())
