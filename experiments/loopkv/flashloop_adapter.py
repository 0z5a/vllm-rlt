"""Adapt a separately loaded native Ouro's weight views to the pinned FlashLoop API."""

from types import SimpleNamespace

from torch import nn
from torch.nn import functional as F


class FlashLoopRotary(nn.Module):
    def __init__(self, rotary):
        super().__init__()
        self.rotary = rotary

    def forward(self, hidden, positions):
        cos, sin = self.rotary(hidden, positions)
        return cos.squeeze(1), sin.squeeze(1)


def adapt_native_ouro(model):
    """Consume this model instance; FlashLoop may repack its projection weights."""
    config = model.config
    model.config = SimpleNamespace(**config.to_dict())
    model.model.rotary_emb = FlashLoopRotary(model.model.rotary_emb)
    for layer in model.model.layers:
        layer.self_attn.head_dim = config.head_dim
        layer.self_attn.scaling = config.head_dim**-0.5
        layer.self_attn.num_key_value_groups = (
            config.num_attention_heads // config.num_key_value_heads
        )
        layer.mlp.act_fn = F.silu
    return model
