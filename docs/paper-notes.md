# Source notes and implementation boundaries

Sources were checked on 2026-09-10. These notes distinguish the published method, the released model, and the engineering choices made for vllm-lt.

## Continuous depth batching

[Schwethelm et al., *Depth-adaptive Inference of Looped Language Models via Continuous Depth Batching*, arXiv:2608.09444v1](https://arxiv.org/pdf/2608.09444v1) separates prefill, prelude, recurrent core, and coda. Refill batches tokens at different loop depths, prioritizing coda and admissible prefill over recurrent work. No-refill shrinks a cohort until every token exits, then executes coda. Last-exited KV fills skipped depths with the last computed per-layer state; shared KV instead overwrites one depth slot and changes attention semantics (Sections 4.1–4.3).

The paper's asynchronous scheduler uses a separately distilled lookahead gate for Ouro. Its gate predicts an exit one iteration in advance; simply delaying the released gate is not equivalent (Appendix C.2). The performance experiments replay recorded outputs and exits and use shared KV on an H100 80 GB. Appendix C.1 reports Ouro GSM8K accuracy of 77.86% with full depth-indexed KV, 0.23% with one shared slot, and 71.34% with first-then-shared KV. These results motivate preserving depth-specific KV here. Chunked prefill is listed as future work, rather than a reproduced implementation detail.

The [authors' repository](https://github.com/kschwethelm/continuous-depth-batching/tree/260ca350cc35bb579879f40fee47c077a2c244ca) contains only `README.md` and `LICENSE` at commit `260ca350cc35bb579879f40fee47c077a2c244ca`. Its README says “Coming soon...”. No CDB implementation or lookahead checkpoint was available there when checked.

## Released Ouro checkpoint

The first target is [ByteDance/Ouro-1.4B](https://huggingface.co/ByteDance/Ouro-1.4B/tree/574fa66cb8bf5abdc979642d01cf2b79b16bfab1), revision `574fa66cb8bf5abdc979642d01cf2b79b16bfab1`.

| Property | Released configuration |
| --- | --- |
| Architecture | `OuroForCausalLM`, 24 shared transformer layers |
| Hidden / FFN width | 2,048 / 5,632 |
| Attention | 16 query heads, 16 KV heads, head dimension 128 |
| Recurrent steps | 4 |
| Vocabulary | 49,152 |
| Position encoding | RoPE, theta 1,000,000; maximum position 65,536 |
| Normalization / activation | RMSNorm epsilon `1e-6`; SwiGLU |
| Weights | BF16; untied embeddings and output head |

These values come from the [pinned configuration](https://huggingface.co/ByteDance/Ouro-1.4B/blob/574fa66cb8bf5abdc979642d01cf2b79b16bfab1/config.json).

The [pinned model code](https://huggingface.co/ByteDance/Ouro-1.4B/blob/574fa66cb8bf5abdc979642d01cf2b79b16bfab1/modeling_ouro.py) applies four sandwich RMSNorms per layer. It applies `model.norm` after every complete loop and feeds that normalized state into the next loop. The learned gate includes a bias and operates on the normalized state. Its conditional probabilities form a cumulative exit distribution:

```text
remaining = 1
after loop r:
    conditional_exit = sigmoid(gate(normalized_hidden))
    remaining *= 1 - conditional_exit
    cumulative_exit = 1 - remaining
```

Thresholding each conditional probability separately would implement a different policy. The released implementation computes every loop before choosing the output hidden state; its adaptive output selection does not skip deeper KV writes.

## What this implementation establishes

The [engine design](design.md) defines a synchronous first version with full-depth chunked prefill, cumulative-gate adaptive decoding, and last-exited paged KV. Chunked prefill and conservative request reservation are explicit project choices. Threshold `1.0` explicitly selects fixed-depth execution, avoiding accidental early exits from numerical rounding. Adaptive decoding defaults to a minimum of two loops.

For verification, fixed-depth execution can be compared with the released Hugging Face model. True early-exit decoding needs a reference that implements the same skipped-computation cache policy: comparing it directly with Hugging Face's adaptive output selection would compare different histories. Refill and no-refill should produce identical outputs when model weights, token selection, gate rules, and cache semantics are fixed.

This version does not establish a throughput improvement or reproduce the paper's asynchronous results. CUDA graphs, scheduling/computation overlap, and a calibrated lookahead checkpoint remain separate work. Future performance comparisons must state their cache and gate policies, preserve workload traces, and measure variability under controlled hardware and software conditions.
