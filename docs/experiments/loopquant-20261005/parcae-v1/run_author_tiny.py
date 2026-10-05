"""Compare the dense adapter with unchanged pinned author forward and all recurrent states."""

import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import sys

import torch

from loopquant.adapters.parcae import ParcaeAdapter
from tests.loopkv.test_parcae import tiny_config
from vllm_rlt.models.parcae import ParcaeForCausalLM

root = Path(__file__).parent
peer = root.parent / "loopkv-zero-copy-20261005"
author = peer / "references/parcae"
revision = subprocess.check_output(
    ["git", "rev-parse", "HEAD"], cwd=author, text=True
).strip()
assert revision == "69284c13746e849104f738d6d1a347b1f457df76"
assert not subprocess.check_output(["git", "diff", "HEAD"], cwd=author)
sys.path.insert(0, str(author))
from parcae_lm.models.parcae.config import ParcaeConfig as AuthorConfig
from receval.models.parcae import ModelingParcae

torch.set_num_threads(1)
torch.manual_seed(41)
config = tiny_config()
native = ParcaeForCausalLM(config).eval()
values = json.loads(
    (peer / "evidence/model-catalog/parcae-370m-config.json").read_text()
)
values.pop("_class_name")
values.pop("rope_settings")
for key, value in config.to_dict().items():
    if key not in ("model_type", "_class_name", "rope_base", "eos_token_id"):
        values[key] = value
values.update(
    padded_vocab_size=config.vocab_size,
    recurrent_embedding_dimension=config.n_embd,
    recurrent_intermediation_embedding_dimension=config.intermediate_size,
)
original = ModelingParcae(AuthorConfig(**values)).eval()
original.load_state_dict(native.state_dict(), strict=True)
assert original.emb_scale == original.config.init.logit_scale == 1
rows, author_states = [], []


def capture(module, inputs, output):
    author_states.append(output.detach().clone())


hook = original.transformer.core_block[-1].register_forward_hook(capture)
with torch.inference_mode():
    for dtype, backend, batch, length in itertools.product(
        [torch.float32, torch.bfloat16], ["eager", "sdpa"], [1, 2], [1, 3, 7]
    ):
        native.to(dtype)
        original.to(dtype)
        original.freqs_cis = native.freqs_cis.clone()
        adapter = ParcaeAdapter(native, attention_backend=backend)
        tokens = (
            torch.arange(batch * length).reshape(batch, length) * 263 + 257
        ) % config.vocab_size
        author_states.clear()
        torch.manual_seed(109)
        expected = original.forward_for_generation(tokens)["logits"]
        torch.manual_seed(109)
        actual = adapter(tokens, torch.ones_like(tokens, dtype=torch.bool), 3)
        assert len(author_states) == len(actual.states) == 3
        atol, rtol = (3e-6, 3e-5) if dtype == torch.float32 else (0.03, 0.02)
        states_close = all(
            bool(((a - b).abs() <= atol + rtol * b.abs()).all())
            for a, b in zip(actual.states, author_states)
        )
        logits_close = bool(
            ((actual.logits - expected).abs() <= atol + rtol * expected.abs()).all()
        )
        rows.append(
            dict(
                dtype=str(dtype),
                backend=backend,
                batch=batch,
                length=length,
                states_close=states_close,
                logits_close=logits_close,
                states_exact=all(
                    torch.equal(a, b) for a, b in zip(actual.states, author_states)
                ),
                logits_exact=torch.equal(actual.logits, expected),
                max_abs_logit=float((actual.logits - expected).abs().max()),
                greedy_exact=torch.equal(actual.logits.argmax(-1), expected.argmax(-1)),
                atol=atol,
                rtol=rtol,
            )
        )
hook.remove()
result = dict(
    scope="tiny copied weights, unchanged pinned author generation and full recurrent-state hooks; not official checkpoint or native engine",
    author_revision=revision,
    runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    adapter_sha256=hashlib.sha256(
        (root / "source-parcae/loopquant/adapters/parcae.py").read_bytes()
    ).hexdigest(),
    cases=rows,
)
(root / "evidence/parcae-adapter-author-tiny-attempt2.json").write_text(
    json.dumps(result, indent=2) + "\n"
)
for backend in ["eager", "sdpa"]:
    selected = [r for r in rows if r["backend"] == backend]
    print(
        json.dumps(
            dict(
                backend=backend,
                cases=len(selected),
                passed=sum(r["states_close"] and r["logits_close"] for r in selected),
                bitwise=sum(r["states_exact"] and r["logits_exact"] for r in selected),
            )
        )
    )
