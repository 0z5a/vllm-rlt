"""Register native and Torch kernels once without initializing a GPU."""

from .activation import silu_and_mul

__all__ = ["silu_and_mul", "single_launch_ops"]

_LAYER_OPS = None


def single_launch_ops():
    """``kernels.layer_ops``, loaded on first use, or ``False`` without Triton."""
    global _LAYER_OPS
    if _LAYER_OPS is None:
        try:
            from . import layer_ops as module
        except ImportError:
            _LAYER_OPS = False
        else:
            _LAYER_OPS = module
    return _LAYER_OPS
