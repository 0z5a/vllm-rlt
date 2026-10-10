# H100 FP32 correctness qualification

The original full-checkpoint W2 and public-entry FP32 protocols pass on one NVIDIA H100 with 65,247 MiB observed memory. Source `e8aa1ee57734bc5c1118c143ebdb68d0b53210a1`, the unchanged production base `d700ecca664da2e8f3122f66ad9d45de4295e45a`, Ouro-1.4B revision `574fa66cb8bf5abdc979642d01cf2b79b16bfab1`, inputs and numerical criteria remain pinned.

| Check | Complete scope | Result |
|---|---|---|
| Stateful Triton W2 | 12 cases, 108 readouts, 96 forced continuations; four historical planes | PASS at original atol/rtol 1e-4 |
| K/V history | 82,944 logical vector positions each, checked by the original runtime assertions | PASS |
| Readout and lifecycle | Exact unchanged readout, 21 ownership checks, stale-owner rejection and drain | PASS |
| Public tokenizer | 52 original frozen text/ID records | PASS |
| Public parameter loader | All 269 parameters byte-exact after FP32 cast | PASS |
| Public facade and dense continuation | Eight facade outputs and six forced dense readouts | PASS |
| Off-node record review | All 103 payload hashes and 114 recorded selected-logprob scalar pairs | PASS |
| Natural completion | Both children, paired parent, resource wrapper and scheduler step | All return 0 |

Recorded W2 maximum absolute state/K/V/logit/selected-logprob errors are 6.29425e-5 / 3.09944e-5 / 1.08480e-5 / 5.69820e-5 / 2.86102e-5. The independent selected-logprob review uses the original `atol + rtol * abs(expected)` budget; maximum error/budget is 0.00938062. Full K/V/state/logit vectors were not persisted, so this review does not claim independent replay of those complete tensors. Their numerical evidence is the executed original assertions and byte-pinned source/report provenance.

A preceding attempt failed while loading the Triton driver JIT module with an unresolved `libimf.so` dependency; the public stage remained unrun. The successful retry used command-scoped existing GCC 11.5.0 and new private caches, with all scientific children, package pins, model inputs and thresholds unchanged. Generated host-module dependency inspection found no `libimf` dependency. The actual host build argument list was not traced. Triton's [original host builder](https://github.com/triton-lang/triton/blob/v3.7.1/python/triton/runtime/build.py) reads `CC`; its host-module cache key excludes that compiler choice.

The two qualification stages took 63.358 seconds together. This is qualification elapsed time, not a benchmark speedup. Torch 2.12.1+cu130, CUDA 13.0, Triton 3.7.1, Transformers 4.54.1, tokenizers 0.21.4, safetensors 0.8.0, huggingface-hub 0.35.3 and NumPy 2.4.6 were qualified with Python 3.12.1, driver 595.91.07 and TF32 disabled.

The small W2 cache (128 blocks of four tokens) and public 2 GiB cache do not qualify the original 64/102 GiB KV profiles. Those profiles retain their capacity stop on this device. Final quality, high-concurrency E2E, HTTP, Graph, training and external-system speed comparisons remain unmeasured. The original seven W5 targets and 653-ID / three-arm / 1,959-output confirmation remain required.

[Machine-readable scientific summary](H100_FP32_QUALIFICATION-v3.json). Original failures and operational source/raw/exit/compiler evidence remain private.
