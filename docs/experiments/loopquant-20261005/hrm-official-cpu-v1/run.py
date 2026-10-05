"""Compare HRM official weights with author recurrence using an explicit CPU SDPA shim."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import ModuleType

import torch
from safetensors.torch import load_file
from torch.nn import functional as F

from loopquant.adapters.hrm_text import HrmTextAdapter
from vllm_rlt.models.hrm_text import HrmTextConfig, HrmTextForCausalLM


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--root", type=Path, required=True)
parser.add_argument("--requests", type=Path, required=True)
parser.add_argument("--author-root", type=Path, required=True)
parser.add_argument("--tiny", action="store_true")
args = parser.parse_args()
root = args.root
author_root = args.author_root
folder = root.parent / "loopkv-zero-copy-20261005/models/hrm-text-1b"
torch.set_num_threads(2)
replacements = 0


def cpu_attention(query, key, value, is_causal, *, prefix_lens, valid_rows):
    """Replace the unavailable FlashAttention dependency, not the H/L recurrence."""
    global replacements
    replacements += 1
    batch, length = query.shape[:2]
    allowed = (
        torch.ones(length, length, dtype=torch.bool)
        .tril()
        .expand(batch, -1, -1)
        .clone()
    )
    if not is_causal:
        for row, prefix in enumerate(prefix_lens.tolist()):
            allowed[row, :, :prefix] = True
    allowed &= valid_rows[:, None, :]
    return F.scaled_dot_product_attention(
        query.transpose(1, 2),
        key.transpose(1, 2),
        value.transpose(1, 2),
        attn_mask=allowed[:, None],
    ).transpose(1, 2)


def unsupported_cache(*args, **kwargs):
    raise NotImplementedError(
        "this dense CPU comparison does not implement author CUDA KV cache"
    )


prefix_module = ModuleType("models.flash_attention_prefixlm_v2")
prefix_module.flash_attn_varlen_prefixlm = cpu_attention
flash_module = ModuleType("flash_attn_interface")
flash_module.flash_attn_with_kvcache = unsupported_cache
sys.modules[prefix_module.__name__] = prefix_module
sys.modules[flash_module.__name__] = flash_module
sys.path.insert(0, str(author_root))
from models.baselines.hrm_nocarry_bp_warmup import HierarchicalReasoningModel
from models.layers import RotaryEmbedding
from models.lm_head import LMHead

started = time.time()
config = HrmTextConfig.from_dict(json.loads((folder / "config.json").read_text()))
if args.tiny:
    config = HrmTextConfig(
        vocab_size=256,
        n_embd=256,
        intermediate_size=768,
        module_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=64,
        max_position_embeddings=64,
        embedding_scale=16,
    )
    torch.manual_seed(71)
    native = HrmTextForCausalLM(config).to(dtype=torch.bfloat16)
else:
    receipt = json.loads(
        (root / "evidence/hrm-official-reuse-verified.json").read_text()
    )
    assert receipt["status"] == "verified" and Path(receipt["folder"]) == folder
    with torch.device("meta"):
        native = HrmTextForCausalLM(config)
    native.load_state_dict(
        load_file(folder / "model.safetensors"), strict=True, assign=True
    )
    native.model.rotary_emb.inv_freq = 1 / (
        config.rope_theta
        ** (torch.arange(0, config.head_dim, 2).float() / config.head_dim)
    )
native.eval().requires_grad_(False)
author_config = dict(
    max_seq_len=config.max_position_embeddings,
    n_layers=config.module_layers,
    hidden_size=config.n_embd,
    num_heads=config.num_attention_heads,
    expansion=config.intermediate_size * 3 / (config.n_embd * 2),
    attn_type="prefixlm",
    init_type="lecun_normal",
    norm_type="pre",
    norm_eps=config.rms_norm_eps,
    pos_emb_type="rope",
    rope_theta=config.rope_theta,
    H_cycles=config.H_cycles,
    L_cycles=config.L_cycles,
    vocab_size=config.vocab_size,
)
with torch.device("meta"):
    original = LMHead(HierarchicalReasoningModel(author_config), author_config)
weights = {}
for name, value in native.state_dict().items():
    key = name.replace("model.H_module.layers.", "model.H_level.core.layers.")
    key = key.replace("model.L_module.layers.", "model.L_level.core.layers.")
    key = key.replace("model.z_L_init", "model.zL_init")
    if key == "model.embed_tokens.weight":
        key = "embed_tokens.embedding_weight"
    weights[key] = value
original.load_state_dict(weights, strict=True, assign=True)
original_weights = original.state_dict()
assert all(
    value.data_ptr() == original_weights[name].data_ptr()
    for name, value in weights.items()
)
for core in (original.model.H_level.core, original.model.L_level.core):
    core.rotary_emb = RotaryEmbedding(
        config.head_dim, config.max_position_embeddings, base=config.rope_theta
    )
original.eval().requires_grad_(False)
adapter = HrmTextAdapter(native)
requests = json.loads(args.requests.read_text())
if args.tiny:
    requests = [
        dict(request_id=f"tiny-{i}", token_ids=list(range(1, length + 1)))
        for i, length in enumerate((7, 4))
    ]
low, high, records = [], [], []


def capture_low(_module, _inputs, output):
    low.append(output.detach().clone())


def capture_high(_module, _inputs, output):
    high.append(output.detach().clone())


hooks = [
    original.model.L_level.register_forward_hook(capture_low),
    original.model.H_level.register_forward_hook(capture_high),
]
with torch.inference_mode():
    for batch, boundary in ((1, "first"), (1, "half"), (1, "all"), (2, "half")):
        for start in range(0, len(requests), batch):
            selected = requests[start : start + batch]
            lengths = [len(row["token_ids"]) for row in selected]
            length = max(lengths)
            tokens = torch.full((batch, length), config.pad_token_id, dtype=torch.long)
            valid = torch.arange(length)[None] < torch.tensor(lengths)[:, None]
            prefixes = torch.tensor(
                [
                    1
                    if boundary == "first"
                    else n
                    if boundary == "all"
                    else max(1, n // 2)
                    for n in lengths
                ]
            )
            for index, row in enumerate(selected):
                tokens[index, : lengths[index]] = torch.tensor(row["token_ids"])
            low.clear()
            high.clear()
            before = replacements
            _, expected = original(
                None,
                dict(
                    inputs=tokens,
                    position_ids=torch.arange(length),
                    prefix_lens=prefixes,
                    valid_rows=valid,
                ),
            )
            actual = adapter(tokens, valid, config.H_cycles, prefix_lengths=prefixes)
            assert len(low) == config.H_cycles * config.L_cycles
            assert len(high) == len(actual.states) == config.H_cycles
            states = [
                (state[valid], torch.cat((high_state, low_state), -1)[valid])
                for state, high_state, low_state in zip(
                    actual.states,
                    high,
                    low[config.L_cycles - 1 :: config.L_cycles],
                    strict=True,
                )
            ]
            logits, target = actual.logits[valid], expected[valid].float()
            row = dict(
                request_ids=[r["request_id"] for r in selected],
                batch=batch,
                boundary=boundary,
                prefix_lengths=prefixes.tolist(),
                valid_tokens=int(valid.sum()),
                padded_length=length,
                loops=config.H_cycles,
                atol=0.03,
                rtol=0.02,
                states_within_budget=all(
                    torch.allclose(a, b, atol=0.03, rtol=0.02) for a, b in states
                ),
                logits_within_budget=torch.allclose(
                    logits, target, atol=0.03, rtol=0.02
                ),
                states_exact=all(torch.equal(a, b) for a, b in states),
                logits_exact=torch.equal(logits, target),
                argmax_exact=torch.equal(logits.argmax(-1), target.argmax(-1)),
                state_max_abs=[
                    float((a.float() - b.float()).abs().max()) for a, b in states
                ],
                logits_max_abs=float((logits - target).abs().max()),
                sdpa_dependency_replacements=replacements - before,
            )
            records.append(row)
            print(json.dumps(row), flush=True)
for hook in hooks:
    hook.remove()
result = dict(
    status="pass"
    if all(
        r["states_within_budget"] and r["logits_within_budget"] and r["argmax_exact"]
        for r in records
    )
    else "fail",
    scope="BF16 CPU dense author recurrence with explicit FlashAttention-to-SDPA dependency shim; not unchanged author attention, CUDA, native KV, quantized quality or E2E",
    official_weights=not args.tiny,
    model_revision="22097cbcecdd1301afe30a19a3ee61b96a9863e5",
    author_source_sha256={
        str(p.relative_to(author_root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (author_root / "models").rglob("*.py")
    },
    runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    requests_sha256=hashlib.sha256(args.requests.read_bytes()).hexdigest(),
    pid=os.getpid(),
    start_unix=started,
    end_unix=time.time(),
    records=records,
)
label = "tiny" if args.tiny else "official"
(root / f"evidence/{label}-hrm-adapter-cpu-sdpa-attempt1.json").write_text(
    json.dumps(result, indent=2) + "\n"
)
raise SystemExit(0 if result["status"] == "pass" else 1)
