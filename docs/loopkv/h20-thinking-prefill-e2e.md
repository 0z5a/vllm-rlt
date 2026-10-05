# Official Thinking prefill CUDA E2E

NVIDIA H20 ran the official BF16 `ByteDance/Ouro-1.4B-Thinking` checkpoint at revision `3aaa2224253a92ca45cf2e3d427c360e1ef9c93d`. Enabling batched prefill metadata changed paired median throughput by **-1.37% to +0.59%** across the six comparisons. Descriptive paired bootstrap intervals include no change in **6/6** comparisons. The option remains off by default.

The frozen implementation is `aa60941d6238072e05a9a571615f1e2726907670`, running Torch 2.12.1+cu130 and Triton 3.7.1 in the previously qualified isolated environment. All seven checkpoint assets were checked against their pinned SHA256 before loading. The native PyTorch checkpoint reader loaded the full official weights once; they remained resident while fresh engines reset each arm's cache and scheduler.

## Complete generation comparison

Each case uses P128/D64, fixed R4, B=1/8/32 and C=2B, BF16 weights and KV, eager Triton attention, and the same token IDs and seed. Native and compact storage each compare `prefill_batch_metadata=False` against `True`. Six paired repetitions alternate off/on and on/off. Every measured arm follows one complete generation warmup. Timing spans request submission through synchronized full drain, including equal CPU work counters; it excludes loading, output serialization and disk writes.

| B / C | Storage | Baseline tok/s | Candidate tok/s | Paired improvement | Paired bootstrap 95% |
|---|---|---:|---:|---:|---:|
| 1 / 2 | native | 16.49 | 16.16 | -1.37% | -3.00% to +0.92% |
| 1 / 2 | compact | 16.41 | 16.38 | +0.48% | -1.13% to +2.02% |
| 8 / 16 | native | 111.02 | 111.63 | +0.21% | -1.26% to +1.91% |
| 8 / 16 | compact | 106.52 | 108.02 | +0.59% | -0.70% to +4.82% |
| 32 / 64 | native | 300.23 | 301.48 | -0.96% | -1.85% to +5.57% |
| 32 / 64 | compact | 287.50 | 285.98 | -0.26% | -3.45% to +3.80% |

Throughput columns use output tokens divided by median complete-generation time. Improvement and its 95% interval use paired ratios, with 10,000 bootstrap resamples and seed 20261006. These are repeated E2E screening results within one model process; independent fresh-process confirmation and HTTP serving measurements remain separate. No comparison between storage rows is claimed because storage order was not balanced.

## Correctness and realized work

All **72 measured arms, 72 complete warmups and three traced references** finished naturally: **4,018 requests and 257,152 generated tokens**, including 125,952 measured tokens. Every complete request object, output token, exit sequence and work counter matches its case's native reference. The independent audit recomputes recurrent batch histograms and prefill work from the raw reference schedules. All physical blocks return to the pool; compact live and reserved records both drain to zero.

| Requested B / C | Observed peak residents | Recurrent effective batch mean | Recurrent p10 / p50 / p90 |
|---|---:|---:|---|
| 1 / 2 | 1 | 1.00 | 1 / 1 / 1 |
| 8 / 16 | 8 | 7.89 | 8 / 8 / 8 |
| 32 / 64 | 32 | 30.15 | 27 / 32 / 32 |

## Allocated memory

| B / C | Storage | Baseline median peak allocated (bytes) | Candidate median peak allocated (bytes) | Change |
|---|---|---:|---:|---:|
| 1 / 2 | native | 3,060,877,312 | 3,060,892,672 | +15,360 |
| 1 / 2 | compact | 3,060,883,456 | 3,060,894,208 | +10,752 |
| 8 / 16 | native | 4,117,870,592 | 4,117,885,952 | +15,360 |
| 8 / 16 | compact | 4,117,926,912 | 4,117,937,664 | +10,752 |
| 32 / 64 | native | 7,741,847,552 | 7,741,862,912 | +15,360 |
| 32 / 64 | compact | 7,742,075,904 | 7,742,086,656 | +10,752 |

These are PyTorch peak allocated bytes per measured arm, reset after engine construction; they include the model and fixed cache pool. The evidence also retains reserved bytes. Metadata construction reductions are separate from these observed allocations and from device traffic.

## Reproduction and evidence

The exact benchmark driver was `6a405962dd73ff15cc15594a2ac07a63b7bf03d7e0e39a46b52e60653d8763e3`. [prefill_pairs.py](../../experiments/loopkv/prefill_pairs.py) and [score_prefill_pairs.py](../../experiments/loopkv/score_prefill_pairs.py) have the same parsed Python AST as the executed driver and independent auditor; publication changes only formatting. Supply the recorded plan with local model, source and prompt paths. Source and input hashes are checked before loading.

[Machine-readable results](evidence/h20-thinking-prefill-e2e.json) contain every measured/warmup summary, output-file hashes, paired times, checkpoint manifest and natural-exit receipts. The raw archive is `h20-thinking-e2e-v1-results.tar.gz`, SHA256 `8bad5657616d2165497f0fa1d5735c8f7fafbf82e689c015074835d56ac05650`; all 354 archived files were independently verified offbox.

The initial local smoke fixture requested R2 while its full-depth prefill returned R4 for the first token. That assertion failure is retained; the corrected full-R4 fixture passed all 34 generation arms, and the production driver explicitly checks the model's full depth. The production plan was R4 throughout. The first SSH attempt failed before staging because its shared socket had expired; normal authentication restored the connection and the frozen package then ran unchanged.
