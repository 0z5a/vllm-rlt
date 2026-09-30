import argparse
import copy
import json
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from vllm_rlt.config import ExecutionConfig
from vllm_rlt.engine.rl_engine import RLEngine
from vllm_rlt.models.ouro import OuroForCausalLM

parser = argparse.ArgumentParser()
parser.add_argument("model", type=Path)
parser.add_argument("output", type=Path)
args = parser.parse_args()
torch.set_num_threads(4)
model = OuroForCausalLM.from_pretrained(args.model, device="cuda", dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
prompts = [tokenizer.encode("What is 1+1? Give a short answer.")] * 2
other = copy.deepcopy(model)
engines = [
    RLEngine(model, num_blocks=256, max_num_seqs=2),
    RLEngine(
        other, num_blocks=256, max_num_seqs=2, execution_config=ExecutionConfig(cuda_graphs=True)
    ),
]
records = []
for loops in (2, 3, 4):
    for index, engine in enumerate(engines):
        engine.generate_batch(prompts, loops=loops, max_tokens=32, seed=17)
    for repeat in range(5):
        outputs = []
        for index in (0, 1) if repeat % 2 == 0 else (1, 0):
            engine = engines[index]
            torch.cuda.synchronize()
            start = time.monotonic()
            result = engine.generate_batch(prompts, loops=loops, max_tokens=32, seed=17 + repeat)
            torch.cuda.synchronize()
            elapsed = time.monotonic() - start
            records.append(
                {
                    "K": loops,
                    "repeat": repeat,
                    "graphs": bool(index),
                    "seconds": elapsed,
                    "tokens": [x.token_ids for x in result],
                    "logprobs": [x.log_probs for x in result],
                    "block_tokens": dict(engine.block_tokens),
                }
            )
            outputs.append((index, result))
        reference = next(result for index, result in outputs if index == 0)
        actual = next(result for index, result in outputs if index == 1)
        assert [x.token_ids for x in actual] == [x.token_ids for x in reference]
        for actual_one, reference_one in zip(actual, reference, strict=True):
            torch.testing.assert_close(
                torch.tensor(actual_one.log_probs), torch.tensor(reference_one.log_probs)
            )
    print("DONE", loops, flush=True)
weights = {name: parameter.clone() for name, parameter in model.named_parameters()}
weights["model.layers.0.mlp.down_proj.weight"].mul_(0.5)
for engine in engines:
    engine.publish(weights, 1)
result = [engine.generate_batch(prompts, loops=4, max_tokens=8, seed=27) for engine in engines]
assert [x.token_ids for x in result[0]] == [x.token_ids for x in result[1]]
for left, right in zip(*result, strict=True):
    torch.testing.assert_close(torch.tensor(left.log_probs), torch.tensor(right.log_probs))
    assert left.policy_version == right.policy_version == 1
args.output.write_text(
    json.dumps(
        {
            "records": records,
            "published_graph_match": True,
            "device": torch.cuda.get_device_name(),
            "torch": torch.__version__,
        },
        indent=2,
    )
)
print("PASS", flush=True)
