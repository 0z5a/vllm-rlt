"""Unmodified pinned Huginn class, explicit paired initial state and full recurrence."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tokenizers import Tokenizer

from loopquant.adapters.huginn import HuginnAdapter
from vllm_rlt.models.huginn import HuginnConfig, HuginnForCausalLM

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--root", type=Path, default=Path(__file__).parent)
root = parser.parse_args().root
folder = root / "models/huginn-0125"
manifest = json.loads((folder / "verified-manifest.json").read_text())
assert manifest["revision"] == "bb6621b65e90b6a4b9b29ef88dc83866d450470c"
assert manifest["independently_verified"]
started = time.time()
tokenizer = AutoTokenizer.from_pretrained(
    folder, trust_remote_code=True, local_files_only=True
)
ouro_tokenizer = Tokenizer.from_file(str(root / "data/tokenizer.json"))
requests = json.loads((root / "data/smoke-g0.json").read_text())
for request in requests:
    text = ouro_tokenizer.decode(request["token_ids"], skip_special_tokens=False)
    request["token_ids"] = tokenizer.encode(text, add_special_tokens=True)
    request["text_sha256"] = hashlib.sha256(text.encode()).hexdigest()
(root / "data/huginn-smoke-g0.json").write_text(json.dumps(requests, indent=2) + "\n")
official = (
    AutoModelForCausalLM.from_pretrained(
        folder,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
    )
    .eval()
    .requires_grad_(False)
)
with torch.device("meta"):
    native = HuginnForCausalLM(
        HuginnConfig.from_dict(json.loads((folder / "config.json").read_text()))
    )
native.load_state_dict(official.state_dict(), assign=True, strict=True)
native.lm_head.weight = native.transformer.wte.weight
expected_parameters = dict(official.named_parameters())
assert all(
    value.data_ptr() == expected_parameters[name].data_ptr()
    for name, value in native.named_parameters()
)
assert native.lm_head.weight.data_ptr() == native.transformer.wte.weight.data_ptr()
adapter = HuginnAdapter(native)
records = []
observed = []


def capture_state(module, inputs, output):
    observed.append(output)


handle = official.transformer.core_block[-1].register_forward_hook(capture_state)
with torch.inference_mode():
    for index, request in enumerate(requests):
        tokens = torch.tensor([request["token_ids"]], dtype=torch.long)
        valid = torch.ones_like(tokens, dtype=torch.bool)
        torch.manual_seed(17 + index)
        initial = adapter.initialize_state(tokens)
        for loops in [8, 16, 32]:
            observed.clear()
            expected = official(
                tokens,
                input_states=initial,
                num_steps=torch.tensor([0, loops]),
                use_cache=False,
            )
            actual = adapter(tokens, valid, loops, initial)
            assert len(observed) == len(actual.states) == loops
            state_errors = [
                float((a.float() - b.float()).abs().max())
                for a, b in zip(actual.states, observed, strict=True)
            ]
            row = dict(
                request_id=request["request_id"],
                loops=loops,
                tokens=tokens.numel(),
                state_max_abs=state_errors,
                logits_max_abs=float((actual.logits - expected.logits).abs().max()),
                state_exact=all(
                    torch.equal(a, b)
                    for a, b in zip(actual.states, observed, strict=True)
                ),
                logits_exact=bool(torch.equal(actual.logits, expected.logits)),
            )
            records.append(row)
            print(json.dumps(row), flush=True)
handle.remove()
(root / "evidence/official-huginn-cpu-attempt1.json").write_text(
    json.dumps(
        dict(
            status="pass"
            if all(row["state_exact"] and row["logits_exact"] for row in records)
            else "fail",
            scope=(
                "official full checkpoint CPU BF16 SDPA, 32 fixed inputs x R8/16/32; "
                "every recurrent state and final logits; no serving performance"
            ),
            model_revision=manifest["revision"],
            records=records,
            start_unix=started,
            end_unix=time.time(),
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        ),
        indent=2,
    )
    + "\n"
)
