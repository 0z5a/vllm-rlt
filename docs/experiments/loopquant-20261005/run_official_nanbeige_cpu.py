"""Compare the complete pinned Nanbeige checkpoint with shared BF16 weights."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import torch
from tokenizers import Tokenizer
from transformers import AutoModelForCausalLM

from loopquant.adapters.nanbeige import NanbeigeAdapter
from vllm_rlt.models.nanbeige import NanbeigeConfig, NanbeigeForCausalLM

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--root", type=Path, required=True)
root = parser.parse_args().root
folder = root / "models/nanbeige-4.2-3b"
manifest = json.loads((folder / "verified-manifest.json").read_text())
assert manifest["revision"] == "b82e54bd609793562a75cbf9337970a93369eab5"
assert manifest["independently_verified"]
started = time.time()
tokenizer = Tokenizer.from_file(str(folder / "tokenizer.json"))
ouro_tokenizer = Tokenizer.from_file(str(root / "data/tokenizer.json"))
requests = json.loads((root / "data/smoke-g0.json").read_text())
for request in requests:
    text = ouro_tokenizer.decode(request["token_ids"], skip_special_tokens=False)
    request["token_ids"] = tokenizer.encode(text, add_special_tokens=True).ids
    request["text_sha256"] = hashlib.sha256(text.encode()).hexdigest()
(root / "data/nanbeige-smoke-g0.json").write_text(json.dumps(requests, indent=2) + "\n")
official = (
    AutoModelForCausalLM.from_pretrained(
        folder,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    )
    .eval()
    .requires_grad_(False)
)
config = NanbeigeConfig.from_dict(json.loads((folder / "config.json").read_text()))
assert not config.skip_loop_final_norm and config.num_loops == 2
with torch.device("meta"):
    native = NanbeigeForCausalLM(config)
native.load_state_dict(official.state_dict(), assign=True, strict=True)
native.model.rotary_emb.inv_freq = native.model.rotary_emb.frequencies(device="cpu")
parameters = dict(official.named_parameters())
assert all(
    p.data_ptr() == parameters[name].data_ptr() for name, p in native.named_parameters()
)
adapter = NanbeigeAdapter(native)
observed: list[torch.Tensor] = []


def capture_state(module, inputs, output):
    observed.append(output)


handle = official.model.norm.register_forward_hook(capture_state)
records = []
with torch.inference_mode():
    for request in requests:
        tokens = torch.tensor([request["token_ids"]], dtype=torch.long)
        valid = torch.ones_like(tokens, dtype=torch.bool)
        observed.clear()
        official.model(tokens, attention_mask=valid.long(), use_cache=False)
        assert len(observed) == 2
        for loops in [1, 2]:
            actual = adapter(tokens, valid, loops)
            expected = official.lm_head(observed[loops - 1]).float()
            state_errors = [
                float((a.float() - b.float()).abs().max())
                for a, b in zip(actual.states, observed[:loops], strict=True)
            ]
            row = dict(
                request_id=request["request_id"],
                loops=loops,
                tokens=tokens.numel(),
                state_max_abs=state_errors,
                logits_max_abs=float((actual.logits - expected).abs().max()),
                state_exact=all(
                    torch.equal(a, b)
                    for a, b in zip(actual.states, observed[:loops], strict=True)
                ),
                logits_exact=bool(torch.equal(actual.logits, expected)),
            )
            records.append(row)
            print(json.dumps(row), flush=True)
handle.remove()
(root / "evidence/official-nanbeige-cpu-attempt1.json").write_text(
    json.dumps(
        dict(
            status="pass"
            if all(r["state_exact"] and r["logits_exact"] for r in records)
            else "fail",
            scope="complete official BF16 eager checkpoint, 32 fixed texts x R1/R2; no serving performance",
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
