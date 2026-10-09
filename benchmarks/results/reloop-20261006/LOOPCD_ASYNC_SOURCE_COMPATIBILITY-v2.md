# Re:Loop async reference source compatibility

The private fixed-greedy async candidate remains disabled and unexecuted.
Independent review verified all 39 tar files/38 payload hashes and copied all
133 verified e8 source files into a fresh local compatibility directory.

| Check | Actual result |
|---|---|
| Complete source packet bytes | Pass |
| Complete e8 source bytes | Pass |
| Direct application of the older reference patch to e8 | Rejected: three source-file hunks do not match |
| Actual e8 shared submit path | Reaches LoopCD capture and readout |
| Candidate capture/readout/execute math versus its older origin | AST unchanged |
| Older origin capture/readout/execute math versus e8 | AST differs |
| Candidate async engine step versus its origin and e8 | AST unchanged |
| Candidate model import or CPU/GPU qualification | Not run |

Preserving the older origin's math does not prove preservation of e8 math.
The next source change needs a minimal port onto the complete e8 source;
the old core and Graph files cannot be copied wholesale. The existing capture
and readout path is present. Admission restrictions and cross-stream reference
ownership, stop/abort, ID reuse, partial prefill, readback and drain remain
unqualified for the proposed path.

Existing native/Graph results retain their original source and test scopes.
Neither this source check nor unsupported-mode rejection establishes a new
full-model eager/Graph/async result, high-C/multi-batch E2E, quality or speed.
Whole-task pass contribution remains zero.

[Immutable hashes and compatibility checks](LOOPCD_ASYNC_SOURCE_COMPATIBILITY-v2.json).
