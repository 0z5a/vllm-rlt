# Re:Loop full-model HTTP entry preparation

The finite HTTP driver now connects the public full-checkpoint loader to the
production serving frontend. It preserves the original W5 and distinct Re:Loop
C128 protocols: 4C requests per trial, five warmups, five measurements and six
fresh processes in reversed three-arm order. This is unexecuted source preparation.

| Prepared behavior | Qualification state |
|---|---|
| Pinned full model/tokenizer, FP32 P4/R4 and original cache/scheduler settings | Runtime pending |
| Actual client and backend outstanding peaks both reach declared C | Runtime pending; configuration alone cannot pass |
| Actual resident A and effective decode S observed | Runtime pending |
| Original tokens/depths/text/usage, repeated parity and natural drain | Runtime pending |
| Partial SSE/non-200/engine-failure raw retained before failure | CPU stream-failure control pending |
| Source-only grant rejected before other inputs or model read | Actual local negative check passed |
| Syntax, Ruff and driver CLI | Actual local checks passed |

The source review caught lazy tokenizer loading in the public offline facade;
the HTTP owner factory explicitly loads and passes the pinned tokenizer.
The original parser/runtime interface patch and neutral native observer remain
byte-identical. All 14 frozen source/control files were independently checked.

Actual canonical parser/CLI/socket tests, production string roundtrip, healthy
and truncated-stream CPU integration, whole-window numerical/resource binding
and complete raw/offbox auditing remain required before execution. This driver
does not consume Loopquant's separate 1024-request supplement or close
Graph/async, other-model qualification, quality or performance. Whole-task pass
contribution is zero.

[Source hashes and scope](HTTP_FULLMODEL_ENTRY_SOURCE-v5.json).
