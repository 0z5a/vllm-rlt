import pytest
import torch
from vllm_rlt.engine.rl_engine import RLEngine
from vllm_rlt.models.config import OuroConfig
from vllm_rlt.models.ouro import OuroForCausalLM
from vllm_rlt.models.reference import dense_reference
from vllm_rlt.sampling_params import SamplingParams


def test_depth_probabilities_publication_and_actual_work():
    torch.manual_seed(9)
    model = OuroForCausalLM(OuroConfig.tiny())
    engine = RLEngine(model, num_blocks=64, attention_backend="torch")
    prompt = [1, 4, 7]
    for k in (4, 2, 3, 4):
        output = engine.generate_batch([prompt], loops=k, max_tokens=4, seed=17)[0]
        logits = dense_reference(model, torch.tensor(prompt + output.token_ids), k)[-1][2]
        expected = (
            logits[len(prompt) - 1 : -1]
            .log_softmax(-1)
            .gather(-1, torch.tensor(output.token_ids)[:, None])
            .squeeze(-1)
        )
        torch.testing.assert_close(torch.tensor(output.log_probs), expected)
        assert output.exit_depths == [k] * len(output.token_ids)
        assert engine.block_tokens == {
            "prefill": 2 * k * len(prompt),
            "decode": 2 * k * (len(output.token_ids) - 1),
        }
    weights = {name: value.clone() for name, value in model.named_parameters()}
    weights["lm_head.weight"].normal_()
    engine.publish(weights, 1)
    torch.testing.assert_close(model.lm_head.weight, weights["lm_head.weight"])
    assert engine.generate_batch([prompt], loops=2, max_tokens=2, seed=7)[0].policy_version == 1
    with pytest.raises(ValueError, match="newer version"):
        engine.publish(weights, 1)
    incomplete = dict(weights)
    incomplete.pop("lm_head.weight")
    with pytest.raises(ValueError, match="all physical"):
        engine.publish(incomplete, 2)
    assert engine.ready and engine.policy_version == 1


def test_reject_active_publication_and_different_sampling_policy():
    model = OuroForCausalLM(OuroConfig.tiny())
    engine = RLEngine(model, num_blocks=64, attention_backend="torch")
    with pytest.raises(ValueError, match="uniform fixed depth"):
        engine.add_request(
            "invalid", [1, 2], SamplingParams(temperature=1, min_loops=2, max_loops=3)
        )
    engine.add_request("active", [1, 2], SamplingParams(temperature=1, min_loops=4, max_loops=4))
    with pytest.raises(ValueError, match="idle"):
        engine.publish(dict(model.named_parameters()), 1)
    with pytest.raises(RuntimeError, match="Finish"):
        engine.set_loop_budget(2)
    engine.abort_request("active")
    engine.ready = False
    with pytest.raises(RuntimeError, match="did not complete"):
        engine.generate_batch([[1]], loops=2, max_tokens=1, seed=3)
