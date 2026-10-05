"""Pinned author comparison with all eight recurrent boundaries retained."""

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import torch

from loopquant.adapters.loopformer import LoopFormerAdapter
from vllm_rlt.models.loopformer import LoopFormerConfig, LoopFormerForCausalLM

root = Path(__file__).parent
path = root / "evidence/model-inventory/loopformer-3block-8iterations/modeling_loopformer.py"
digest = hashlib.sha256(path.read_bytes()).hexdigest()
assert digest == "a4fdd1330a88a859f5f726a15fdce5b6c37d523fd7f04b9c9765d6c58133e665"
spec = importlib.util.spec_from_file_location("loopformer_author_adapter", path)
author = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = author
spec.loader.exec_module(author)
torch.set_num_threads(1)
torch.manual_seed(92)
config = LoopFormerConfig(vocab_size=32, n_embd=16, n_layer=2, n_head=2,
                          intermediate_dim=24, block_size=64)
model = LoopFormerForCausalLM(config).requires_grad_(False).eval()
original = author.GPT(author.GPTConfig(**{
    k: v for k, v in config.to_dict().items() if k not in ("model_type", "architectures")
})).requires_grad_(False).eval()
weights = {name.removeprefix("gpt."): value for name, value in model.state_dict().items()}
weights["lm_head.weight"] = weights["transformer.wte.weight"]
original.load_state_dict(weights, strict=True)
observed = []


def capture(_module, _inputs, output):
    observed.append(output)


hook = original.transformer.h.blocks[-1].register_forward_hook(capture)
rows = []
with torch.inference_mode():
    for dtype, atol, rtol in [(torch.float32, 3e-6, 3e-5), (torch.bfloat16, 0.03, 0.02)]:
        model.to(dtype)
        original.to(dtype)
        adapter = LoopFormerAdapter(model)
        for batch in (1, 2):
            for length in (3, 7):
                tokens = torch.arange(batch * length).reshape(batch, length) * 3 % 32
                observed.clear()
                expected, _ = original(tokens)
                actual = adapter(tokens, torch.ones_like(tokens, dtype=torch.bool), 8)
                assert len(observed) == len(actual.states) == 8
                for state, target in zip(actual.states, observed, strict=True):
                    torch.testing.assert_close(state[..., :-1], target, atol=atol, rtol=rtol)
                torch.testing.assert_close(actual.logits, expected.float(), atol=atol, rtol=rtol)
                rows.append(dict(dtype=str(dtype), batch=batch, length=length, atol=atol, rtol=rtol,
                                 state_max_abs=[float((a[..., :-1].float() - b.float()).abs().max())
                                                for a, b in zip(actual.states, observed, strict=True)],
                                 logits_max_abs=float((actual.logits - expected.float()).abs().max()),
                                 state_exact=all(torch.equal(a[..., :-1], b)
                                                 for a, b in zip(actual.states, observed, strict=True)),
                                 logits_exact=torch.equal(actual.logits, expected.float())))
hook.remove()
receipt = dict(status="pass", scope="pinned author tiny weights, all8steps; not official weights/GPU/performance",
               model_revision="2b4fbaaf4e2510353ef7cfe07a07c671b5226739", author_sha256=digest,
               source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root / "source-families", text=True).strip(),
               runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), cases=rows)
(root / "evidence/loopformer-author-adapter-attempt1.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(receipt))
