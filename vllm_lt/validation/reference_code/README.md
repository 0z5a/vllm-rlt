# Pinned official Ouro reference

The two Python source files in this directory are byte-identical copies from
ByteDance/Ouro-1.4B revision `574fa66cb8bf5abdc979642d01cf2b79b16bfab1`.
They are used only by the Q1 fixed-four-loop validation adapter, under Apache-2.0.
The repository LICENSE applies; upstream attribution is retained in the
configuration source and in the repository NOTICE. Do not format these files.

| File | SHA-256 |
| --- | --- |
| [modeling_ouro.py](https://huggingface.co/ByteDance/Ouro-1.4B/blob/574fa66cb8bf5abdc979642d01cf2b79b16bfab1/modeling_ouro.py) | `c5c68fbb368ce2909c257ae2afc50719be8c91539333d3295e19312c4316f413` |
| [configuration_ouro.py](https://huggingface.co/ByteDance/Ouro-1.4B/blob/574fa66cb8bf5abdc979642d01cf2b79b16bfab1/configuration_ouro.py) | `950443e32929047aa08d02abad2e1888bc1914b3db988d3d675f70787f65dafb` |

Reviewed execution dependencies: Transformers 4.55.0, PyTorch, huggingface-hub,
tokenizers, and safetensors. The adapter requires Transformers 4.55.0 and the
absence of optional `kernels`. Transformers 4.55.0 constructs `LayerRepository`
descriptors without a revision or version; installed kernels 0.16.0 rejects
those arguments during import. With `kernels` absent, the published RMSNorm
decorator is an identity. The adapter also verifies original RMSNorm method
identity before device allocation and each forward. No AutoModel, remote-code
loader, checkpoint loader, generation API,
SDPA, Flash Attention, or official cache is used by the adapter.

The official forward executes all four loops and `exit_at_step=3` selects the
fourth loop's normalized hidden state. Eager attention uses native-dtype QK and
PV products, with FP32 softmax cast to query dtype. This is an independent
fixed-depth reference; it does not implement LAST-EXITED adaptive KV histories.

The adapter constructs the unchanged class on the meta device, strictly assigns
provided detached weight tensors without copying their storage, and recreates
the unchanged official rotary module on the weights' device. Its inverse
frequencies remain FP32. No published forward method is patched.
