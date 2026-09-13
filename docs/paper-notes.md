# Source notes and implementation boundaries

CDB sources were checked on 2026-09-10; original Ouro precision and evaluation
guidance was checked on 2026-09-12. These notes distinguish the published method,
the released model, and the engineering choices made for vllm-lt.

## Original Ouro paper and precision guidance

The original Ouro paper is [Zhu et al., *Scaling Latent Reasoning via Looped
Language Models*, arXiv:2510.25741v5](https://arxiv.org/html/2510.25741v5).
The CDB paper below is a separate source for scheduling/cache design.
The inspected Ouro paper mentions BF16 training in its auxiliary experiments
(Appendix B), but provides no per-kernel inference dtype/accumulation recipe.

For inference, the [pinned release configuration](https://huggingface.co/ByteDance/Ouro-1.4B/blob/574fa66cb8bf5abdc979642d01cf2b79b16bfab1/config.json)
sets `torch_dtype` to `bfloat16`. Its [Quick Start](https://huggingface.co/ByteDance/Ouro-1.4B/blob/574fa66cb8bf5abdc979642d01cf2b79b16bfab1/README.md)
loads with `torch_dtype="auto"`. The following explicit boundaries come from
the [released model code](https://huggingface.co/ByteDance/Ouro-1.4B/blob/574fa66cb8bf5abdc979642d01cf2b79b16bfab1/modeling_ouro.py):

| Operation | Published inference behavior |
| --- | --- |
| Projections, MLP, LM head | Model-dtype operands/outputs; no full-model FP32 promotion. Backend accumulation flags are not fixed here. |
| Eager attention | Native-dtype QK and PV products; softmax explicitly computes in FP32, then casts probabilities back to query dtype before PV. |
| RMSNorm | FP32 normalization arithmetic, then cast back to input dtype. |
| RoPE | FP32 phase and trigonometric computation, then cast factors to activation dtype. |
| Exit distribution | Sigmoid, remaining probability, and CDF use tensors without an explicit FP32 promotion. |

The downloaded model source matches the vendored official reference exactly:
SHA-256 `c5c68fbb368ce2909c257ae2afc50719be8c91539333d3295e19312c4316f413`.
The existing [reference source notes](../vllm_lt/validation/reference_code/README.md)
record the pin, dependency compatibility, and eager adapter scope.

These observations define the initial fidelity baseline. Our paged attention
retains FP32 score/probability/value intermediates; our runner computes sigmoid
in FP32 and updates cumulative probability as host Python floats. Those are
project choices requiring declared comparisons, not author-mandated precision.
Fused kernels may have different rounding boundaries; document and validate
such differences without requiring a full-model FP32 fallback. See the
[precision policy](precision-policy.md).

## Accuracy evidence and the original evaluation protocol

The [existing Q2 screen](https://github.com/hsliuustc0106/vllm-lt/blob/8a0654ad73cf59af4546517fab583b4f002cacf4/docs/benchmarks/q2-fp32-quality-20260911.md)
used FP32, fixed four loops, a custom zero-shot prompt/parser, and at most 256
output tokens. It scored 4/64 correct (6.25%); 58 answers were unparseable and
57 hit the output limit. No paired BF16 task-accuracy result exists in that
report. Numerical agreement tests do not establish language-task accuracy.

The original paper reports 78.92% GSM8K for Ouro-1.4B at four loops.
[Appendix C.1 / Table 16](https://arxiv.org/html/2510.25741v5#A3.T16)
specifies strict match, **3-shot CoT**, and **lm-eval-harness**. Our screen does
not reproduce that protocol, so the two scores are not a controlled comparison
and their difference does not identify a dtype or engine defect.

A successor accuracy evaluation must start from those published settings and
the pinned BF16 release. Freeze the harness/task revision, three demonstrations,
prompt formatting, extraction rules, EOS/stop behavior, and token limits before
execution; identify details not specified by the paper as project choices.
Compare native and official fixed-four-loop BF16 under the same protocol,
then assess adaptive behavior separately. Size context/output budgets for the
few-shot protocol instead of inheriting the old 512/256 limits. Retain the old
4/64 result; a bounded subset is a regression screen, not full-paper replication.

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
