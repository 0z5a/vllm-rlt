# Independent-process HTTP baseline calibration

This frozen pilot calibrates later SLO and arrival-rate thresholds. It contains
only native baselines and contributes no observations to the formal candidate
comparison. Execution is pending a quiet reservation of both RTX5090 cards and
shared IO.

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
