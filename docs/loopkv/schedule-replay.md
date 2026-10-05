# Fixed logical schedule diagnostics

The capture hook records natural async admission, exact batch row order and
request/position/depth, prefill spans, recurrent bucket size and coda-delivery
order. Replay forces those decisions through the same engine and runner. Only
the engine exit decision uses the captured trace; the runner and Graph retain
the real gate computation. Cache allocations remain local to each implementation.
This scope currently excludes preemption and prefix reuse.

Native self-replay must reproduce complete token/exit objects before an alias
comparison is accepted. All replay work counters must equal the captured run,
including batch histograms, per-depth loop rows, prefill tokens and peak residents.
The driver retains raw request objects, logical events and actual Graph counters.
It does not report diagnostic elapsed times as performance or serving latency.

The serialized numerical oracle is separate: it collects each stage before
policy update and compares hidden states and gate/sample results between native
and alias Graphs. It stops comparison if their logical schedules diverge. This
changes overlap and is not a Native-A performance baseline.

| Validation | Result |
|---|---|
| Serialized oracle CPU policy and deliberate gate-divergence control | 4 passed |
| Logical replay CPU native/alias, reversed readback readiness, natural/full4/forced1 depth | 12 passed |
| CUDA and official checkpoints | Pending |
| Fixed-work E2E speedup | Not measured |
