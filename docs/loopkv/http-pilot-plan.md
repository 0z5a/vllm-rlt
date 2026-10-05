# Independent-process HTTP baseline calibration

This frozen pilot calibrates later SLO and arrival-rate thresholds. It contains
only native baselines and contributes no observations to the formal candidate
comparison. All 12 workers completed naturally under a quiet reservation of both RTX5090
cards and shared IO. The 74 raw files (55,947,035 bytes) were archived, copied off
the node and independently hash-verified. The audit reconciled 6,144 measured
requests, 1,536 warmup requests and 319,488 measured output tokens.

| Setting | Frozen value |
|---|---|
| Source | `cd8a742f437c6ead5c33331f62f2af71c6c4c8d0` |
| Checkpoint | Existing verified Ouro-1.4B, BF16 |
| B / client concurrency | 32 / 64 and 64 / 128 |
| Execution | Native async and native Graph |
| Independent process repetitions | Three per point, 12 workers total |
| Measured / warmup requests per worker | 512 / 128 |
| Controlled prompt / output lengths | P32/64/128/256; D16/32/64/96 |
| Order | Frozen shuffle with seed 20261005 |
| GPU | RTX5090 #0; #1 remains quiet |
| Process lifecycle | Fresh model load and full natural drain; stop on first worker failure |

Each worker serves token-ID chunks over localhost TCP. The same engine performs
the separate warmup cohort and measured cohort. Model loading, warmup and result
writes are outside the measured full-drain interval. Captures during measurement
remain counted. This does not measure tokenizer throughput or the production
text endpoint.

For each B/concurrency and execution mode, derive each SLO from 1.5 times the
maximum of the three native-run nearest-rank P99 values. Round TTFT and complete
request latency upward to 0.1 seconds, and request P99 ITL upward to 0.01 seconds.
Freeze those values before candidate measurements and use the same limits for
both cards and all candidates at that point.

For open-loop follow-up, use arrival rates of 0.5, 0.8, 1.0 and 1.2 times the
median native completed-requests/second rate from these three closed-loop runs.
Formal arrivals use separately frozen seeds. These are campaign-specific SLOs,
not product latency requirements.

Formal comparisons require at least seven independently restarted balanced pairs
per card, at least 120 measured seconds and 2,000 requests per arm/run. Report
output, exit and work differences alongside P99, goodput and uncertainty. Keep
all failures and do not delete runs based on their observed performance.

| Frozen artifact | SHA256 |
|---|---|
| Quiet controller plan | `7a388563a2f2a0f405875a1cdca6169ba4045b16fef3f53f2bf1df9fe01bae4d` |
| Worker experiment plan | `1102850fbfbeecbd5b93715b6e13496c03f8fb9e553b69a9dcb351a1103d350b` |
| Measured cohort | `7391ac8ffee440f1f7470a8f668e2e6644cca6235e2435e31392db700568d4cc` |
| Warmup cohort | `19d931b68dc25f4ca431606d772b3b8f76fe09dc8cba9f49e9fdb6163e932d25` |
| Controller | `0cfb91a6b2a4cb57b5f60be081f417fc9d01a0fd573a7617368df317a4a0b22d` |
| Source bundle | `d75fbc0f7d1c32f689eaa1e6a23492d9eafc68d7a3426fdbd6aaf8db8f4ef293` |


## Measured calibration and frozen limits

These are three native-only runs per point, each serving 512 measured requests.
The pilot is excluded from candidate speed comparisons. SLO values below follow
the rule above; arrival rates in the raw report retain full precision.

| B / C | Native mode | Median requests/s | Median tokens/s | TTFT / request / request-P99-ITL SLO (s) | Open-loop arrival rates (requests/s) | Candidate speedup |
|---|---|---:|---:|---|---|---|
| 32 / 64 | native-async | 5.6609 | 294.37 | 11.7 / 25.7 / 0.54 | 2.8304 / 4.5287 / 5.6609 / 6.7931 | Not measured |
| 32 / 64 | native-graph | 5.7896 | 301.06 | 20.4 / 28.8 / 0.56 | 2.8948 / 4.6317 / 5.7896 / 6.9476 | Not measured |
| 64 / 128 | native-async | 7.5019 | 390.10 | 18.7 / 38.1 / 0.69 | 3.7509 / 6.0015 / 7.5019 / 9.0022 | Not measured |
| 64 / 128 | native-graph | 7.4411 | 386.94 | 20.1 / 42.8 / 0.60 | 3.7206 / 5.9529 / 7.4411 / 8.9293 | Not measured |

## Repeatability and realized work

Compare each repetition with the first observed run of the same case and mode.
Non-Graph output tokens, exit depths and complete work records match across all
three repetitions. Graph results vary across independently restarted workers;
these differences remain in the report and preclude an exact-output claim.
Every run submits 61,440 prefill tokens and produces 26,624 output tokens.

| B / C | Mode | Token-different requests across 3 runs | Exit-different requests across 3 runs | Recurrent rows | Effective B | Measured Graph replays / fallbacks |
|---|---|---|---|---|---|---|
| 32 / 64 | native-async | 0/0/0 | 0/0/0 | 89974/89974/89974 | 29.66/29.66/29.66 | Off |
| 32 / 64 | native-graph | 0/8/4 | 0/270/186 | 89845/89878/89884 | 10.06/14.88/12.07 | 8656/278; 4521/1518; 6514/935 |
| 64 / 128 | native-async | 0/0/0 | 0/0/0 | 89923/89923/89923 | 54.63/54.63/54.63 | Off |
| 64 / 128 | native-graph | 0/0/10 | 0/0/100 | 89929/89929/89921 | 48.35/48.35/50.66 | 408/1452; 408/1452; 289/1486 |

All Graph captures occurred in initialization/warmup; measured captures are
zero. Measured fallbacks are nonzero and explicitly counted. Peak realized
residents are 32 and 64 respectively. These are localhost token-ID transport
measurements, with a shared asyncio event loop; they do not measure production
text serving or tokenizer throughput. All workers use one frozen combined source;
this does not complete independent parent/candidate source qualification per PR.

[Audit, all twelve worker summaries and raw-file hashes](evidence/http-pilot-v1.json).
The auditor is `experiments/loopkv/score_http_pilot.py`; run it with `--plan` pointing
to the frozen controller plan, `--results` to the raw archive directory and
`--out` to the audit output. The plan's sibling workload files must be present.
Archive SHA256: `deaf6e87b614d1c94e953a4a744267933a3f4832bf9511e13fdd5aa3650b704f`.
Completion marker SHA256: `27c8126373ad3d14c83f95fc683f23d79f32783c1964f5a62a48ba496773471a`.
