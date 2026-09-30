import copy
import unittest
from dataclasses import replace

import torch

from vllm_rlt.config import ExecutionConfig
from vllm_rlt.engine.rl_engine import RLEngine
from vllm_rlt.models.config import OuroConfig
from vllm_rlt.models.ouro import OuroForCausalLM


def validate_graph_publication():
    torch.manual_seed(9)
    model = OuroForCausalLM(replace(OuroConfig.tiny(), head_dim=64, eos_token_id=None)).to(
        "cuda", torch.bfloat16
    )
    eager = RLEngine(copy.deepcopy(model), num_blocks=64)
    graphed = RLEngine(model, num_blocks=64, execution_config=ExecutionConfig(cuda_graphs=True))
    prompts = [[1, 4, 7], [2, 5, 8, 9]]
    for version in (0, 1):
        for k in (4, 2, 3, 4):
            reference = eager.generate_batch(prompts, loops=k, max_tokens=5, seed=17)
            actual = graphed.generate_batch(prompts, loops=k, max_tokens=5, seed=17)
            assert [x.token_ids for x in actual] == [x.token_ids for x in reference]
            for result, expected in zip(actual, reference, strict=True):
                torch.testing.assert_close(
                    torch.tensor(result.log_probs), torch.tensor(expected.log_probs)
                )
                assert result.policy_version == version
                assert result.exit_depths == [k] * len(result.token_ids)
            assert graphed.block_tokens == eager.block_tokens
            assert graphed.cache_manager.num_used_blocks == 0
        if version == 0:
            graphs = graphed.model_runner.graphs
            captures = graphs.captures
            pointers = {name: value.data_ptr() for name, value in model.named_parameters()}
            weights = {name: value.clone() for name, value in model.named_parameters()}
            weights["model.layers.0.mlp.down_proj.weight"].mul_(0.5)
            weights["lm_head.weight"].normal_()
            eager.publish(weights, 1)
            graphed.publish(weights, 1)
            assert {name: value.data_ptr() for name, value in model.named_parameters()} == pointers
    assert graphs.captures == captures
    assert graphs.replays > graphs.captures


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestRLGraphs(unittest.TestCase):
    def test_publication_depths_and_work(self):
        validate_graph_publication()


if __name__ == "__main__":
    unittest.main()
