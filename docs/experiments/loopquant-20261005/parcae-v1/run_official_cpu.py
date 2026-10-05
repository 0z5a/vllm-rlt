"""Official Parcae weights: unchanged author and dense adapter with paired seeds."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch

from loopquant.adapters.parcae import ParcaeAdapter
from vllm_rlt.models.parcae import ParcaeConfig, ParcaeForCausalLM

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--root", type=Path, required=True)
parser.add_argument("--requests", type=Path, required=True)
args = parser.parse_args()
root = args.root
peer = root.parent / "loopkv-zero-copy-20261005"
folder = peer / "models/parcae-370m-cpu"
author = peer / "references/parcae"
receipt = json.loads(
    (root / "evidence/parcae-official-reuse-verified.json").read_text()
)
assert receipt["revision"] == "439284464ee4999bd1f762da7d044613a4828efe"
assert receipt["status"] == "verified" and Path(receipt["folder"]) == folder
revision = subprocess.check_output(
    ["git", "rev-parse", "HEAD"], cwd=author, text=True
).strip()
assert revision == "69284c13746e849104f738d6d1a347b1f457df76"
assert not subprocess.check_output(["git", "diff", "HEAD"], cwd=author)
sys.path.insert(0, str(author))
from parcae_lm.models.parcae.config import ParcaeConfig as AuthorConfig
from receval.models.parcae import ModelingParcae

started = time.time()
torch.set_num_threads(2)
values = json.loads((folder / "config.json").read_text())
values.pop("_class_name")
values.pop("rope_settings")
with torch.device("meta"):
    official = ModelingParcae(AuthorConfig(**values)).eval()
    native = ParcaeForCausalLM(
        ParcaeConfig.from_dict(json.loads((folder / "config.json").read_text()))
    )
weights = torch.load(
    folder / "pytorch_model.bin", map_location="cpu", mmap=True, weights_only=True
)
assert torch.equal(weights["lm_head.weight"], weights["transformer.wte.weight"])
official.load_state_dict(weights, strict=True, assign=True)
rotary = weights["freqs_cis"]
official.to(dtype=torch.bfloat16)
official.freqs_cis = rotary
official.lm_head.weight = official.transformer.wte.weight
official.requires_grad_(False)
native.load_state_dict(official.state_dict(), strict=True, assign=True)
native.lm_head.weight = native.transformer.wte.weight
native.eval().requires_grad_(False)
assert official.emb_scale == official.config.init.logit_scale == 1
assert native.freqs_cis.dtype == torch.float32
expected_parameters = dict(official.named_parameters())
assert all(
    parameter.data_ptr() == expected_parameters[name].data_ptr()
    for name, parameter in native.named_parameters()
)
del weights
adapter = ParcaeAdapter(native, attention_backend="sdpa")
requests = json.loads(args.requests.read_text())
assert len(requests) == 32
records, observed = [], []


def capture(module, inputs, output):
    observed.append(output.detach().clone())


hook = official.transformer.core_block[-1].register_forward_hook(capture)
with torch.inference_mode():
    for index, request in enumerate(requests):
        tokens = torch.tensor([request["token_ids"]])
        valid = torch.ones_like(tokens, dtype=torch.bool)
        for loops in [1, 4, 8]:
            observed.clear()
            torch.manual_seed(17 + index)
            expected = official.forward_for_generation(tokens, num_steps=loops)[
                "logits"
            ]
            torch.manual_seed(17 + index)
            actual = adapter(tokens, valid, loops)
            assert len(actual.states) == len(observed) == loops
            row = dict(
                request_id=request["request_id"],
                loops=loops,
                tokens=tokens.numel(),
                state_exact=all(
                    torch.equal(a, b)
                    for a, b in zip(actual.states, observed, strict=True)
                ),
                logits_exact=torch.equal(actual.logits, expected),
                state_max_abs=[
                    float((a.float() - b.float()).abs().max())
                    for a, b in zip(actual.states, observed, strict=True)
                ],
                logits_max_abs=float((actual.logits - expected).abs().max()),
            )
            records.append(row)
            print(json.dumps(row), flush=True)
hook.remove()
result = dict(
    status="pass"
    if all(r["state_exact"] and r["logits_exact"] for r in records)
    else "fail",
    scope="official Parcae BF16 CPU SDPA,32 fixed inputs x R1/4/8,all dense recurrent states and logits; no native cache or performance claim",
    model_revision=receipt["revision"],
    author_revision=revision,
    runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    requests_sha256=hashlib.sha256(args.requests.read_bytes()).hexdigest(),
    pid=os.getpid(),
    start_unix=started,
    end_unix=time.time(),
    records=records,
)
(root / "evidence/official-parcae-adapter-cpu-attempt1.json").write_text(
    json.dumps(result, indent=2) + "\n"
)
raise SystemExit(0 if result["status"] == "pass" else 1)
