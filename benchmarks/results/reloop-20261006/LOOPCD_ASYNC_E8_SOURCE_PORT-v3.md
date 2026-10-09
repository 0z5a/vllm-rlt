# Re:Loop e8 async reference source port

The private default-off reference patch now uses the complete verified e8
source. Independent review checked all 298 archive files/297 payload hashes,
both complete 133-file source trees and the applied patch's resulting bytes.

| Actual source check | Result |
|---|---|
| Original source versus verified e8 | All 133 files byte-identical |
| Candidate source changes | Four files changed; 129 unchanged |
| Capture/readout/execute and engine step/async-step math | Six function ASTs unchanged from e8 |
| Graph bank | Byte-identical to e8 |
| Fresh private patch check and application | Both zero; all 133 resulting files equal candidate |
| Experimental flag | False default independently verified |
| Candidate import/model/CPU/GPU qualification | 0; not run |

This resolves the earlier [source incompatibility](LOOPCD_ASYNC_SOURCE_COMPATIBILITY-v2.md).
It establishes a source port only. Cross-stream reference ownership, stop/abort,
ID reuse, partial prefill, readback, drain and full-model eager/Graph/async
tensor checks remain required. The fixed-greedy reference path does not close
adaptive async, G3, W5/C128, all-model serving, quality or performance.

Canonical source and original experiments were not modified. Combining this
private source with the separate HTTP patch needs a new complete source
binding and actual qualification; old numerical and HTTP receipts do not
qualify that combination. Whole-task pass contribution is zero.

[Immutable archive and source checks](LOOPCD_ASYNC_E8_SOURCE_PORT-v3.json).
