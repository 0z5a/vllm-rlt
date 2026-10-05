"""Compare official LoopFormer weights through unchanged author and dense adapter."""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

import torch
from safetensors.torch import load_file

from loopquant.adapters.loopformer import LoopFormerAdapter
from vllm_rlt.models.loopformer import LoopFormerConfig, LoopFormerForCausalLM

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--root", type=Path, required=True)
parser.add_argument("--requests", type=Path, required=True)
args = parser.parse_args()
root = args.root
folder = (
    root.parent / "loopkv-zero-copy-20261005/models/loopformer-3block-8iterations-cpu"
)
receipt = json.loads(
    (root / "evidence/loopformer-official-reuse-verified.json").read_text()
)
assert receipt["revision"] == "2b4fbaaf4e2510353ef7cfe07a07c671b5226739"
assert receipt["status"] == "verified" and Path(receipt["folder"]) == folder
author_source = (
    root
    / "evidence/model-inventory/loopformer-3block-8iterations/modeling_loopformer.py"
)
digest = hashlib.sha256(author_source.read_bytes()).hexdigest()
assert digest == "a4fdd1330a88a859f5f726a15fdce5b6c37d523fd7f04b9c9765d6c58133e665"
spec = importlib.util.spec_from_file_location(
    "loopformer_official_author", author_source
)
author = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = author
spec.loader.exec_module(author)

started = time.time()
torch.set_num_threads(2)
config = LoopFormerConfig.from_dict(json.loads((folder / "config.json").read_text()))
with torch.device("meta"):
    native = LoopFormerForCausalLM(config)
    original = author.GPT(
        author.GPTConfig(
            **{
                key: value
                for key, value in config.to_dict().items()
                if key not in ("model_type", "architectures")
            }
        )
    )
native.load_state_dict(
    load_file(folder / "model.safetensors"), strict=True, assign=True
)
native.to(dtype=torch.bfloat16).eval().requires_grad_(False)
weights = {
    name.removeprefix("gpt."): value for name, value in native.state_dict().items()
}
weights["lm_head.weight"] = weights["transformer.wte.weight"]
original.load_state_dict(weights, strict=True, assign=True)
original.lm_head.weight = original.transformer.wte.weight
original.eval().requires_grad_(False)
official_parameters = dict(original.named_parameters())
assert all(
    parameter.data_ptr() == official_parameters[name.removeprefix("gpt.")].data_ptr()
    for name, parameter in native.named_parameters()
)
del weights
adapter = LoopFormerAdapter(native, attention_backend="sdpa")
requests = json.loads(args.requests.read_text())
assert len(requests) == 32
records, observed = [], []


def capture(_module, _inputs, output):
    observed.append(output.detach().clone())


hook = original.transformer.h.blocks[-1].register_forward_hook(capture)
with torch.inference_mode():
    for batch in (1, 2):
        for start in range(0, len(requests), batch):
            selected = requests[start : start + batch]
            length = max(len(row["token_ids"]) for row in selected)
            tokens = torch.full((batch, length), config.pad_token_id, dtype=torch.long)
            valid = torch.zeros_like(tokens, dtype=torch.bool)
            for index, request in enumerate(selected):
                count = len(request["token_ids"])
                tokens[index, :count] = torch.tensor(request["token_ids"])
                valid[index, :count] = True
            observed.clear()
            expected, _ = original(tokens)
            actual = adapter(tokens, valid, 8)
            assert len(actual.states) == len(observed) == 8
            states = [
                (a[..., :-1][valid], b[valid])
                for a, b in zip(actual.states, observed, strict=True)
            ]
            logits, target = actual.logits[valid], expected[valid].float()
            row = dict(
                request_ids=[r["request_id"] for r in selected],
                batch=batch,
                valid_tokens=int(valid.sum()),
                padded_length=length,
                loops=8,
                atol=0.03,
                rtol=0.02,
                states_within_budget=all(
                    torch.allclose(a, b, atol=0.03, rtol=0.02) for a, b in states
                ),
                logits_within_budget=torch.allclose(
                    logits, target, atol=0.03, rtol=0.02
                ),
                state_exact=all(torch.equal(a, b) for a, b in states),
                logits_exact=torch.equal(logits, target),
                argmax_exact=torch.equal(logits.argmax(-1), target.argmax(-1)),
                state_max_abs=[
                    float((a.float() - b.float()).abs().max()) for a, b in states
                ],
                logits_max_abs=float((logits - target).abs().max()),
                clock_exact=all(
                    torch.equal(
                        a[..., -1][valid],
                        torch.full_like(a[..., -1][valid], (i + 1) / 8),
                    )
                    for i, a in enumerate(actual.states)
                ),
            )
            records.append(row)
            print(json.dumps(row), flush=True)
hook.remove()
result = dict(
    status="pass"
    if all(
        r["states_within_budget"]
        and r["logits_within_budget"]
        and r["argmax_exact"]
        and r["clock_exact"]
        for r in records
    )
    else "fail",
    scope="official BF16 CPU dense adapter vs unchanged author; B1 and ragged right-padded B2, R8; no native cache or performance claim",
    model_revision=receipt["revision"],
    author_sha256=digest,
    runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    requests_sha256=hashlib.sha256(args.requests.read_bytes()).hexdigest(),
    pid=os.getpid(),
    start_unix=started,
    end_unix=time.time(),
    records=records,
)
(root / "evidence/official-loopformer-adapter-cpu-attempt1.json").write_text(
    json.dumps(result, indent=2) + "\n"
)
raise SystemExit(0 if result["status"] == "pass" else 1)
