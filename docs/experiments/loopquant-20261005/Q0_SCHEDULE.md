# Ouro Q0 registration

Status: implemented and CPU-tested; the full-model schedule has not run. The
earlier two 128-token gradient audits do not count toward this schedule.

| Setting | Registered value |
|---|---|
| Model / depth | Pinned Ouro-1.4B, native R4 |
| Format | E4M3 weights and activations; BF16 residual/state/KV |
| Trainable values | Shared producer log-scales only |
| Spatial sharing | Q/K/V share one parameter; gate/up share one parameter |
| Initial policies | SH1, then ST2 only if dev warrants it; LOOP for the surviving candidate's matched-budget control |
| Microbatch | One sequence of 256 valid tokens |
| Update | 32 microbatches, 8,192 input tokens and 8,160 next-token targets |
| Schedule | 20-update preflight; a further 200 updates only after cost and dev review |
| Optimizer | AdamW, LR 1e-4, weight decay 0, gradient norm limit 1 |
| Loss | Target-normalized CE; KL and trajectory coefficients both 0 |
| Seed | 17 for the pilot; final three-seed comparison remains unrun |

The [pool registration](q0-data-registration.json) contains 238 complete
256-token windows from 108 calibration documents: 60,928 unique token positions.
The 2,065 short-tail tokens are omitted; sequences never cross a document
boundary. Calibration overlap is declared. Dev and locked-test document overlap
are both zero. The preflight consumes 163,840 repeated input tokens (2.689 pool
epochs); the additional pilot consumes 1,638,400. Combined consumption would be
29.580 pool epochs, not 220 unique examples.

Each epoch uses a deterministic permutation indexed by seed and epoch. The
checkpoint stores consumed microbatches, optimizer state, shared scale values,
and CPU/CUDA RNG state. Resumption checks the model/source/data/scale revisions,
policy, seed, sequence length, and loss configuration. Every update records
sampled pool indices, target counts, gradient/update norms, clipping, wall time,
and verifies that protected weights have no gradient or storage-version change.

```bash
python -m loopquant.cli train --config q0-run.json --level Q0 --output run-q0-20
python -m loopquant.cli train --config q0-continuation.json --level Q0 \
  --resume run-q0-20/checkpoint.pt --output run-q0-220
```

The second command is conditional and has not been launched. `updates` is the
cumulative stop value (20 or 220), so continuation cannot silently repeat the
preflight. The configuration supplies `format`, full `model_revision` and
`runtime_revision`, `model`, `windows`, `windows_sha256`, `scales`,
`scales_sha256`, `policy`, `sequence_length`, `seed`, `updates`, `locks`,
`gpu_index`, and the `Q0Config` fields under `training`. Hashes must come from
the selected full calibration receipt and frozen source before launch.

The native conversion transposes each existing packed training matrix without
re-encoding or copying its storage. It expands the scalar weight scale to the
native GEMM's N-vector ABI and transfers the trained producer scales. Saving
the native checkpoint removes optimizer, reference weights, and dense core
weights from the deployment payload. Native reload, dev quality, mixed-depth
execution, and E2E performance remain separate required validations.
