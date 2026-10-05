# Finite HTTP load and client-visible latency

`experiments.loopkv.http_bench` runs frozen request cohorts through real localhost
TCP connections. A single engine owner thread performs admission and generation;
the client and transport share one asyncio event loop. Each response is streamed
as token-ID chunks with client receipt timestamps. Inputs are already tokenized.
This experiment adapter uses the installed Python standard library; it does not
claim to benchmark the production aiohttp text endpoint or tokenizer throughput.

An open-loop workload fixes each arrival offset. Its SLO latency starts at that
offset, including client dispatch lag. A closed-loop workload limits outstanding
clients and starts request latency when the client sends. Every admitted request
drains; the throughput denominator covers first cohort launch through final drain
and engine synchronization. Startup, model loading, warmup and result writes are
outside that interval. Graph captures during measurement are counted, not hidden.

Requests record generated IDs, exit depths, send/completion times, all chunk
receipts and per-token timestamps. Tokens sharing a chunk share its timestamp;
they are not invented as independently timed events. Summaries distinguish global
token ITL from each request's P99 ITL. Goodput counts only complete requests meeting
all frozen TTFT, request-latency and per-request P99 ITL limits. Without a declared
SLO, goodput is unavailable.

| Validation | Result |
|---|---|
| CPU TCP streaming, native/compact, open/closed loop | Four cases match direct-engine tokens and exits across two request lifetimes |
| Goodput and chunk timestamp accounting | Known request-level SLO and full-drain denominator pass |
| Official GPU HTTP workload | 24 arms / 1,392 requests / 72,384 tokens completed; natural Graph differences retained |
| 120-second / 2,000-request formal cohort | Not run |

The reported quantiles are descriptive nearest-rank values. The minimum-duration
flag is an execution check, not a confidence-interval or performance verdict.
Formal comparisons still require independent paired runs, frozen SLO/arrival
plans, quiet resources and request/run-level uncertainty analysis.

| HTTP comparison | Native tokens/s | Candidate tokens/s | Speedup |
|---|---:|---:|---:|
| Official checkpoint | — | — | Not measured |


[Executed official diagnostic and per-point divergences](http-diagnostic.md) includes complete raw hashes and shared-node measurement limits.
