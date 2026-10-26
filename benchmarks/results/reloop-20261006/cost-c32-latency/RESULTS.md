# Existing C32 native latency comparison

Measured results below use the existing FP32/Triton eager P512/output128 jobs: B16, initial/total C32, actual A/S16. They cover submission through final delivery and drain, including the same observer overhead on each arm. The second fresh round reverses policy order.

| Metric | D4 off | D3 off | D3 guided |
|---|---:|---:|---:|
|Mean trial duration (s)|28.416|24.512|24.871|
|Mean trial throughput (tokens/s)|144.177|167.121|164.701|
|Mean TTFT (ms)|11945.174|10934.280|11076.655|
|Mean TPOT (ms)|74.588|59.380|60.521|
|P95 ITL (ms)|72.130|56.332|57.699|
|P99 request latency (ms)|29402.655|25055.404|25066.075|

Matched elapsed-time speed improvements are +15.9342% for D3 off and +14.2596% for D3 guided. These paired estimates use the same repetition index; dividing independent TPS means gives a different statistic. Guidance increases allocated memory by 128 KiB and reserved memory by 2 MiB.

TTFT includes queue/admission waiting. TPOT is each request’s first-to-last delivery time divided by 127. Tail quantiles use nearest rank over 320 measured requests per arm, or all 40,640 token intervals for ITL. Request and token counts are descriptive observations, not independent environment repeats: only two fresh repeats exist.

This initial-C test does not establish closed-loop 4C refill, mixed/long prompts, new-physical-H20 qualification, HTTP/async/Graph claims or final quality. All original seven W5 cases remain pending actual runs.

Full selected-raw provenance and values: [summary.json](summary.json), SHA256 `a99b736f3b6d6e086065157acf706df703108ae8f2ae7ad74a48693d04f82f38`.

The 30 selected raw files were independently re-read and SHA256-verified, then every reported metric and matched ratio was recomputed. Each portable path in the summary resolves to the byte-identical member in [the existing full C32 scientific archive](../cost-c32-full.tar.gz).
