# E2E throughput summary — 2026-10-13

Inference columns measure complete generated-token throughput. ScaleRLT measures the full update including rollout, training, offload/onload, weight publication and scheduled checkpointing. Positive percentages mean higher throughput. Native, HTTP, depth reduction and GPU scaling use different baselines; they are not added. Kernel or NVTX timings are excluded from this table.

| Project / change | Completed E2E result | Scope / acceptance limit |
| --- | ---: | --- |
| PR82 Coda only | B1 **+0.34%**, B4 **+0.85%** | Accepted Coda-only numbers; broader7.36%/8.02% changes excluded |
| LoopCD at the same depth3 | B16/C32 **−1.1313%**, B32/C64 **−1.2699%** | Old qualified source,48GiB resource variant; no pure CD speedup |
| LoopCD depth4 off →depth3 CD on | **+15.7121% /+11.3490%** | Includes depth reduction; quality uncertain; original64GiB case not run |
| LoopGraph formal W1 | **Not qualified** | Original numerical/token failures remain; no speedup is claimed |
| ReLoop uniform scalar →vector readout | **−0.0413% to+1.8435%**, five workloads | Same-depth private native optimization; all negative/slow results retained |
| ReLoop additional reference batch-copy | Old **+0.1985%**, latest **+0.2861%** | Both against their same vector-readout baseline; latest orders **−0.3002% /+0.8752%**; not a stable gain |
| ScaleRLT full-update1node/2GPU →2nodes/3GPU | Sequential seed42 **+27.0972%**, seed43 **+26.8026%** | Two complete audited pairs; more GPUs; all pilot regressions retained; further tests stopped |

The latest private LoopGraph18-Norm control preserves both full token/depth/work waves, but still has **2,146/8,192 differing hidden/logit positions across38 requests** (previously2,410 across39). This narrows the numerical diagnosis; it adds no qualified speedup to the table. See [the complete audited control](loopgraph-eighteen-located-norm32-control.json).

All inference depth comparisons below retain fixed input/output work, but depth4→depth3 changes model computation and does not establish quality noninferiority. The private vector column is a separate comparison of the same guided implementation; it is not added to either preceding column or transferred to HTTP.

| ReLoop workload | Native guidance, same depth3 | HTTP guidance, same depth3 | Native depth4→guided depth3 | HTTP depth4→guided depth3 | Private native vector vs scalar |
| --- | ---: | ---: | ---: | ---: | ---: |
| L1 | -1.0941% | -0.8271% | +27.8949% | +30.2722% | +1.8435% |
| T32 | -1.4964% | -2.8493% | +13.5231% | +13.3035% | +0.7629% |
| T64_short | -3.3631% | -3.5180% | +11.4011% | +10.3182% | +1.1865% |
| C8_long | -0.3571% | -0.9635% | +13.0379% | +12.1927% | -0.0413% |
| Q64 | -0.8287% | -0.6976% | +10.2414% | +10.5315% | +0.0932% |

The separately declared52GiB C4_longer pool variant has native same-depth guidance **+0.0813%** and HTTP **−0.1363%**. Its depth4 comparisons are **+7.15% /+7.17%**, including depth reduction. Original48GiB capacity stops remain, and these variants do not complete the original seven-case matrix.

The captured current PR86 pin also passed **64/64 original LoopCD CPU tests**, retaining all mathematical and lifecycle assertions; its two GPU cases were excluded and remain unqualified. See [current CPU evidence](loopcd-current-pr86-original-cpu-tests.json).

The current PR86 capture at48932b2 returned **74 pass /33 fail /7 skip**. Its28 optional-FlashAttention dependency failures,3 hidden-device1 PD failures and2 Coda counter failures remain. The original Coda final token/depth comparison was not reached. Existing LoopCD cost data therefore do not qualify this current pin. The final [two-GPU follow-up](loopcd-current-pr86-final-native-followup.json) retained five original failures: three PD cases lack NIXL and two Coda cases still fail the original fallback counter assertion. Separate passive comparison found exact completed eager/graph token-depth trajectories for both Coda implementations. This is tiny-model correctness evidence, without a current-pin speed claim. LoopCD testing is concluded with opt-in/default-off scope.

All ScaleRLT pair outcomes:

| Stage | Seed | Complete-update throughput change |
| --- | ---: | ---: |
| pilot | 42 | -10.8556% |
| pilot | 43 | +17.0348% |
| pilot | 44 | +20.4006% |
| pilot | 45 | +20.4061% |
| pilot | 46 | +35.0814% |
| sequential_confirmation | 42 | +27.0972% |
| sequential_confirmation | 43 | +26.8026% |

Confirmation seeds44..46 were not started after the user's stop. The two completed confirmations preserve all five measured updates per arm, warmups, slow points, numerical/checkpoint checks and matched token/sample counts. Shared host/storage activity still limits general scaling inference; this is not a same-resource inference optimization.

Full sources: [LoopCD costs](loopcd-cost.json), [current PR86 outcomes](loopcd-current-pr86-native-gpu-regressions.json), [ScaleRLT pilots](scalerlt-scaling-pilot.json), [sequential pairs](scalerlt-sequential-pairs.json), [native isolated T32/C8](reloop-native-cpu-isolation.json), [native L1/T64/Q64](reloop-native-cpu-isolation-remaining.json), [HTTP T32/C8](reloop-http-cpu-isolation.json), [HTTP L1/T64/Q64](reloop-http-cpu-isolation-remaining.json), [private vector T64](reloop-vector-readout-e2e.json), [vector T32/C8](reloop-vector-readout-t32-c8-e2e.json), [vector L1/Q64](reloop-vector-readout-l1-q64-e2e.json), [latest copy E2E](reloop-mixed-depth-reference-batch-copy-e2e.json). Exact summary values and source hashes are in [e2e-throughput-summary.json](e2e-throughput-summary.json).
