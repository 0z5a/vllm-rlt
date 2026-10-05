# Pre-registered fixed-work replay comparison

Compare native Graph with the resident prefix-aware alias Graph on each RTX5090
separately. Use the already captured official Ouro1.4B delayed-policy B16/C32 and
B32/C64 P128/D128 schedules. Both implementations replay exactly the same
admissions, coda deliveries, row order, positions, loop depths and Graph bucket
sizes. Native self-replay and the parent alias have passed this logical plan.
The prefix-aware candidate must pass each arm's full warmup before timing.

Each point uses seven paired runs per card, with four native-first and three
alias-first orders shuffled using seed1705. Every arm builds a fresh engine,
executes one complete untimed warmup, resets the logical replay, then times request
submission through full drain. Gate kernels, replay scheduling, and the same CPU
work counters are included. Weight loading, trace parsing, comparisons and disk
writes are outside the measurement. There is no HTTP/SLO/P99 claim.

The outer job must hold both original GPU locks and heavy-IO during the whole
quiet measurement window. Do not admit downloads, hashing, builds or peer GPU
jobs until its actual completion and explicit handoff. Keep natural failures and
all valid regressions. Work counters and complete request objects must match the
frozen reference before accepting an arm.

Report each card separately: all seven paired ratios, median speedup and a
95% paired-bootstrap interval (10,000 resamples, statistics seed1705). Retain
peak CUDA allocation/reservation, actual Graph captures/replays/fallbacks and
warmup/measurement boundaries. Input hashes and the balanced order are frozen
in the external plan before launch. This is track-A fixed-work replay; separate
natural-scheduling performance and serving tests remain required.

Executed without changing the frozen plan. [All four paired result rows and raw evidence](resident-replay-performance.md) report a consistent regression; controller and both card processes naturally returned0.
