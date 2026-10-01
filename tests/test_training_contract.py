"""Training contract checks runnable with unittest on CPU or CUDA/Triton."""

import os
import unittest
from dataclasses import replace

import torch

from tests.reference import dense_reference
from vllm_rlt import (
    LLM,
    CacheConfig,
    ExecutionConfig,
    ExitConfig,
    SamplingParams,
    SchedulerConfig,
    SpeculativeConfig,
)
from vllm_rlt.engine.llm_engine import LLMEngine
from vllm_rlt.models import OuroConfig, OuroForCausalLM
from vllm_rlt.request import Request
from vllm_rlt.worker.sampling import generator_for, probabilities

DEVICE = os.environ.get("CONTRACT_DEVICE", "cpu")


def model():
    torch.manual_seed(17)
    config = OuroConfig(
        vocab_size=64,
        hidden_size=256 if DEVICE == "cuda" else 32,
        intermediate_size=512 if DEVICE == "cuda" else 64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64 if DEVICE == "cuda" else 8,
        max_position_embeddings=128,
        total_ut_steps=4,
        eos_token_id=0,
    )
    return OuroForCausalLM(config).to(DEVICE)


def llm(native, *, graphs=False, asynchronous=False, speculative=False, prefix=False):
    return LLM(
        native,
        attention_backend="triton" if DEVICE == "cuda" else "torch",
        cache_config=CacheConfig(num_blocks=128, block_size=2, enable_prefix_caching=prefix),
        scheduler_config=SchedulerConfig(max_num_seqs=4, max_num_batched_tokens=32),
        exit_config=ExitConfig("ouro_delayed" if asynchronous else "ouro"),
        execution_config=ExecutionConfig(cuda_graphs=graphs, async_scheduling=asynchronous),
        speculative_config=SpeculativeConfig(3) if speculative else None,
    )


class TrainingContractTest(unittest.TestCase):
    def test_selected_probabilities_speculative_target_and_post_update(self):
        native = model()
        prompt = [3, 4, 5]
        for graphs in [False, True] if DEVICE == "cuda" else [False]:
            for speculative in (False, True):
                rollout = llm(native, graphs=graphs, speculative=speculative)
                pointers = [p.data_ptr() for p in native.parameters()]
                for version in (0, 1):
                    if version:
                        weights = [(n, p.detach() * 0.97) for n, p in native.named_parameters()]
                        rollout.start_weight_update(version)
                        rollout.update_weights(weights[:5])
                        rollout.update_weights(weights[5:])
                        rollout.finish_weight_update()
                    for mode in ("raw", "processed"):
                        params = SamplingParams(
                            max_tokens=8,
                            temperature=0.8,
                            top_k=13,
                            top_p=0.9,
                            logprobs=0,
                            logprobs_mode=mode,
                            seed=29,
                            ignore_eos=True,
                        )
                        output = rollout.generate([prompt], params)[0]
                        logits = dense_reference(native, prompt + output.token_ids[:-1], 4)[-1][2]
                        rows = logits[len(prompt) - 1 :]
                        expected = torch.stack(
                            [
                                probabilities(row, params)[token].log()
                                if mode == "processed"
                                else row.float().log_softmax(-1)[token]
                                for row, token in zip(rows, output.token_ids, strict=True)
                            ]
                        )
                        torch.testing.assert_close(
                            torch.tensor(output.log_probs, device=DEVICE),
                            expected,
                            atol=4e-5,
                            rtol=4e-5,
                        )
                        self.assertEqual(output.weight_version, version)
                        self.assertEqual(output.sampling_params, replace(params, max_loops=4))
                    self.assertEqual(pointers, [p.data_ptr() for p in native.parameters()])
                if graphs:
                    bank = (
                        rollout.engine.speculative_runner.graphs
                        if speculative
                        else rollout.engine.model_runner.graphs
                    )
                    self.assertGreater(bank.replays, 0)

    def test_early_exit_async_stop_and_abort(self):
        for graphs in [False, True] if DEVICE == "cuda" else [False]:
            for asynchronous in (False, True):
                rollout = llm(model(), graphs=graphs, asynchronous=asynchronous)
                rollout.start_weight_update(2)
                rollout.update_weights(rollout.engine.model.named_parameters())
                rollout.finish_weight_update()
                params = SamplingParams(max_tokens=5, exit_threshold=0, logprobs=0, ignore_eos=True)
                output = rollout.generate([[3, 4]], params)[0]
                self.assertEqual(output.weight_version, 2)
                self.assertEqual(output.exit_depths, [4] + [3 if asynchronous else 2] * 4)
                self.assertEqual(len(output.log_probs), 5)
                stopped = rollout.generate(
                    [[3, 4]], replace(params, stop_token_ids=(output.token_ids[0],))
                )[0]
                self.assertEqual(stopped.token_ids, output.token_ids[:1])
                self.assertEqual(len(stopped.log_probs), 1)
                self.assertEqual(stopped.finish_reason, "stop")
                rollout.engine.add_request("abort", [3, 4], params)
                while not rollout.engine.step():
                    pass
                aborted = rollout.engine.abort_request("abort")
                self.assertEqual(len(aborted.log_probs), len(aborted.token_ids))
                self.assertEqual(aborted.finish_reason, "abort")

    def test_transaction_invalidates_prefixes_and_recovers_without_mixed_policy(self):
        rollout = llm(model(), prefix=True)
        native = rollout.engine.model
        params = SamplingParams(max_tokens=3, logprobs=0, ignore_eos=True)
        rollout.generate([[3, 4, 5, 6]], params)
        self.assertTrue(rollout.engine.cache_manager.lookup_prefix([3, 4, 5, 6]))
        rollout.pause_generation()
        with self.assertRaises(RuntimeError):
            rollout.generate([[3]], params)
        rollout.resume_generation()
        rollout.engine.add_request("active", [3], params)
        with self.assertRaises(RuntimeError):
            rollout.start_weight_update()
        rollout.engine.abort_request("active")
        rollout.start_weight_update(7)
        self.assertEqual(rollout.engine.cache_manager.num_used_blocks, 0)
        self.assertFalse(rollout.engine.cache_manager.lookup_prefix([3, 4, 5, 6]))
        names = list(native.named_parameters())
        before = names[0][1].detach().clone()
        with self.assertRaises(ValueError):
            rollout.update_weights(
                [(names[0][0], torch.zeros_like(before)), ("invalid", torch.zeros(1))]
            )
        torch.testing.assert_close(before, names[0][1], rtol=0, atol=0)
        with self.assertRaises(RuntimeError):
            rollout.finish_weight_update()
        with self.assertRaises(RuntimeError):
            rollout.generate([[3]], params)
        rollout.start_weight_update(7)
        rollout.update_weights(names[:1])
        with self.assertRaises(ValueError):
            rollout.finish_weight_update()
        rollout.update_weights(names[1:])
        rollout.finish_weight_update()
        self.assertEqual(rollout.generate([[3]], params)[0].weight_version, 7)
        rollout.reset_prefix_cache()
        self.assertEqual(rollout.engine.cache_manager.num_used_blocks, 0)

    def test_seed_none_and_validation(self):
        params = SamplingParams(seed=None, temperature=1)
        requests = [Request(str(i), [3], params) for i in range(2)]
        seeds = [generator_for(r, DEVICE).initial_seed() for r in requests]
        self.assertNotEqual(*seeds)
        for values in ({"logprobs": 1}, {"logprobs_mode": "invalid"}, {"stop_token_ids": (-1,)}):
            with self.assertRaises(ValueError):
                SamplingParams(**values)
        with self.assertRaises(ValueError):
            llm(model()).generate([[3]], SamplingParams(stop_token_ids=(64,)))

    def test_entropy_seed_is_reported_and_replays_the_same_policy(self):
        params = SamplingParams(
            max_tokens=8, temperature=0.9, seed=None, logprobs=0, ignore_eos=True
        )
        for options in ({}, {"asynchronous": True}, {"speculative": True}):
            rollout = llm(model(), graphs=DEVICE == "cuda", **options)
            output = rollout.generate([[3, 4, 5]], params)[0]
            self.assertIsInstance(output.sampling_params.seed, int)
            self.assertLess(output.sampling_params.seed, 2**63)
            replay = rollout.generate([[3, 4, 5]], output.sampling_params)[0]
            self.assertEqual(output.token_ids, replay.token_ids)
            self.assertEqual(output.log_probs, replay.log_probs)
            self.assertEqual(output.exit_depths, replay.exit_depths)
            rollout.engine.add_request("unstarted", [3], replace(params, temperature=0))
            aborted = rollout.engine.abort_request("unstarted")
            self.assertIsInstance(aborted.sampling_params.seed, int)
            self.assertEqual(aborted.log_probs, [])
            rollout.close()
        self.assertIsNone(params.seed)

    def test_preemption_preserves_selected_scores_and_rng(self):
        native = model()
        params = SamplingParams(
            max_tokens=8, temperature=0.8, seed=45, logprobs=0, exit_threshold=0, ignore_eos=True
        )
        results = []
        for suspend in (False, True):
            engine = LLMEngine(
                native,
                cache_config=CacheConfig(64, 2),
                scheduler_config=SchedulerConfig(enable_preemption=True),
                attention_backend="triton" if DEVICE == "cuda" else "torch",
            )
            engine.add_request("a", [1, 2, 3], params)
            while len(engine.scheduler.requests["a"].generated_token_ids) < 3:
                engine.step()
            if suspend:
                engine.add_request("b", [5], SamplingParams(max_tokens=1))
                engine.scheduler.selected_request_ids.clear()
                self.assertTrue(engine.preemption.preempt(engine.scheduler.requests["b"]))
            while engine.has_unfinished_requests():
                for output in engine.step():
                    if output.request_id == "a" and output.finished:
                        results.append(output)
            if suspend:
                self.assertEqual(engine.preemption.resumptions, 1)
            engine.close()
        self.assertEqual(results[0].token_ids, results[1].token_ids)
        self.assertEqual(results[0].log_probs, results[1].log_probs)
        self.assertEqual(results[0].exit_depths, results[1].exit_depths)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
