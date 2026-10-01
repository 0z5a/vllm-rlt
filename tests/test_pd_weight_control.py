"""PD publication control with two real CPU engines; no NIXL transport claim."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from tests.helpers import tiny_ouro_config
from vllm_rlt import LLM, SamplingParams
from vllm_rlt.engine.llm_engine import LLMEngine
from vllm_rlt.models import OuroForCausalLM
from vllm_rlt.pd.config import PDConfig
from vllm_rlt.pd.engine import PDEngine
from vllm_rlt.pd.worker import PDWorker


def control_pair():
    coordinator = PDEngine.__new__(PDEngine)
    coordinator.config = PDConfig()
    coordinator.closed, coordinator.failure = False, None
    coordinator.requests, coordinator.transfers = {}, {}
    coordinator._weight_version, coordinator._update_version = 0, None
    coordinator._update_failed, coordinator._paused = False, False
    coordinator._weight_replies = {}
    coordinator.peers = {
        role: SimpleNamespace(name=role, role=role) for role in ("prefill", "decode")
    }
    workers = {}
    for role, peer in coordinator.peers.items():
        worker = PDWorker.__new__(PDWorker)
        worker.engine = LLMEngine(OuroForCausalLM(tiny_ouro_config()))
        worker.work = {}
        worker.send = lambda kind, peer=peer, **fields: coordinator._message(
            peer, dict(kind=kind, **fields)
        )
        workers[role] = worker
    coordinator._send = lambda name, kind, **fields: workers[name].command(
        dict(kind=kind, **fields)
    )
    return coordinator, workers


class PDWeightControlTest(unittest.TestCase):
    def test_all_workers_commit_and_partial_commit_requires_new_epoch(self):
        coordinator, workers = control_pair()
        facade = LLM(coordinator)
        weights = list(workers["prefill"].engine.model.named_parameters())
        pointers = {
            role: [p.data_ptr() for p in worker.engine.model.parameters()]
            for role, worker in workers.items()
        }
        facade.start_weight_update()
        facade.update_weights(weights)
        facade.finish_weight_update()
        self.assertEqual(facade.get_weight_version(), 1)
        coordinator.start_weight_update()
        coordinator.update_weights(weights)
        with patch.object(
            workers["decode"].engine,
            "finish_weight_update",
            side_effect=RuntimeError("copy failure"),
        ):
            with self.assertRaisesRegex(RuntimeError, "copy failure"):
                coordinator.finish_weight_update()
        self.assertEqual(workers["prefill"].engine.get_weight_version(), 2)
        self.assertEqual(workers["decode"].engine.get_weight_version(), 1)
        with self.assertRaises(RuntimeError):
            coordinator.step()
        with self.assertRaises(RuntimeError):
            coordinator.add_request("blocked", [3])
        coordinator.start_weight_update()
        coordinator.update_weights(weights)
        coordinator.finish_weight_update()
        self.assertEqual(coordinator.get_weight_version(), 3)
        for role, worker in workers.items():
            self.assertEqual(worker.engine.get_weight_version(), 3)
            self.assertEqual(
                pointers[role], [p.data_ptr() for p in worker.engine.model.parameters()]
            )
            output = LLM(worker.engine).generate(
                [[3, 4]], SamplingParams(max_tokens=2, logprobs=0)
            )[0]
            self.assertEqual(output.weight_version, 3)
            self.assertEqual(len(output.log_probs), len(output.token_ids))
        coordinator.reset_prefix_cache()


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
