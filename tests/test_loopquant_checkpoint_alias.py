import json

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from loopquant.checkpoint import load_weights


class TiedCheckpoint(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(8, 4)
        self.head = nn.Linear(4, 8, bias=False)
        self.head.weight = self.embedding.weight
        self.register_buffer("frequencies", torch.arange(4, dtype=torch.float32))


@pytest.mark.parametrize("broken_tie", [False, True])
def test_sharded_loader_preserves_weight_aliases_and_buffer_dtype(tmp_path, broken_tie):
    original = TiedCheckpoint()
    state = {name: value.clone() for name, value in original.state_dict().items()}
    if broken_tie:
        state["head.weight"] += 1
    save_file({"embedding.weight": state["embedding.weight"]}, tmp_path / "first.safetensors")
    save_file(
        {name: value for name, value in state.items() if name != "embedding.weight"},
        tmp_path / "second.safetensors",
    )
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            dict(
                weight_map={
                    name: "first.safetensors"
                    if name == "embedding.weight"
                    else "second.safetensors"
                    for name in state
                }
            )
        )
    )
    with torch.device("meta"):
        loaded = TiedCheckpoint()
    if broken_tie:
        with pytest.raises(ValueError, match="tied checkpoint tensors differ"):
            load_weights(loaded, tmp_path, torch.device("cpu"), torch.bfloat16)
        return
    load_weights(loaded, tmp_path, torch.device("cpu"), torch.bfloat16)
    assert loaded.head.weight is loaded.embedding.weight
    assert loaded.embedding.weight.dtype == torch.bfloat16
    assert loaded.frequencies.dtype == torch.float32
    torch.testing.assert_close(loaded.frequencies, original.frequencies, atol=0, rtol=0)
    torch.testing.assert_close(
        loaded.embedding.weight, original.embedding.weight.bfloat16(), atol=0, rtol=0
    )
