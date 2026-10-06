# Re:Loop

Re:Loop tests whether guided readout recovers quality after reducing decode
loops, and whether that recovery lowers measured generation cost. Prompt
prefill retains all four KV planes. The completed public-checkpoint development
screen and two-fresh fixed-work case select P4D3 guided for independent
confirmation. Final quality remains unmeasured.

This increment depends on [upstream #86](https://github.com/ThinkFlowLab/vllm-rlt/pull/86).
The fork comparison targets `feature/loopcd-ouro-task0` at
`d700ecca664da2e8f3122f66ad9d45de4295e45a`. Tested source is
`e8aa1ee57734bc5c1118c143ebdb68d0b53210a1`: its production package is unchanged
from that base; the intervening additions are continuation and scale tests.

## Completed protocol

- Public Ouro-1.4B Base checkpoint
  `574fa66cb8bf5abdc979642d01cf2b79b16bfab1`; all nine generation inputs hashed
  before CUDA. Public loader and canonical tokenizer, with a hash-pinned native
  research driver.
- NVIDIA H20; Torch 2.12.1+cu130, CUDA 13.0, Triton 3.7.1,
  Transformers 4.54.1 and tokenizers 0.21.4. FP32 parameters and KV;
  TF32 disabled, eager Triton attention.
- GSM8K train revision `740312add88f781978c0658806c59bc2815b9866`:
  128 frozen questions, three fixed demonstrations, greedy generation,
  natural EOS/task stops, common 1024-token maximum. Strict first-match
  answer extraction and the complete frozen denominator.
- B/C16, FCFS/refill, prompt chunk128 and batch-token cap512;
  common configured64GiB R4 `last_exited` KV. Actual A/S are not inferred.
  Off uses disabled capability and null sampling guidance; on uses
  `two_head`, reference1, strength0.3 for prefill and decode.

All six arms completed128 outputs:768 completions and82041 tokens in669.544
seconds of GPU residence. All children and the controller exited0 naturally.
The complete original archive is5682525 bytes, SHA
`2cd4fbfe157d3faa6889fb372f13ce3db6e0009de270ceec135b52f603effae2`;
all106 payload hashes and every prompt/ID/depth/stop trace were verified.

## Test result

| P/D, guidance | Correct /128 | Limit hits | Output tokens | Native natural seconds | Natural elapsed ratio vs D4 off |
|---|---:|---:|---:|---:|---:|
| 4/4 off | 80 | 0 | 13234 | 113.162 | 1.000x |
| 4/4 two_head0.3 | 83 | 0 | 13320 | 104.377 | 1.084x |
| 4/3 off | 79 | 0 | 13079 | 89.428 | 1.265x |
| 4/3 two_head0.3 | 80 | 2 | 15222 | 166.367 | 0.680x |
| 4/2 off | 59 | 0 | 12460 | 65.629 | 1.724x |
| 4/2 two_head0.3 | 62 | 2 | 14726 | 118.474 | 0.955x |

Timing covers native submission through synchronized drain, including CPU
text-stop decoding; model load and serialization are excluded. Output lengths
vary. Two D3-guided truncations remain in the result and produce long tails.
These ratios describe variable-length development generation. The separate
two-fresh fixed-work measurement is below; HTTP remains unmeasured.

| Exploratory comparison | Wins / losses | Difference pp | Conservative interval pp | Status |
|---|---:|---:|---|---|
| D4 guided / D4 off | 10/7 | +2.344 | -9.545 to+14.022 | UNCERTAIN |
| D3 off / D4 off | 5/6 | -0.781 | -10.635 to+9.153 | UNCERTAIN |
| D3 guided / D4 off | 8/8 | 0.000 | -11.538 to+11.538 | UNCERTAIN |
| D3 guided / D3 off | 7/6 | +0.781 | -9.847 to+11.334 | UNCERTAIN |
| D2 off / D4 off | 5/26 | -16.406 | -29.665 to-1.798 | DEGRADED |
| D2 guided / D4 off | 2/20 | -14.062 | -25.126 to-1.715 | DEGRADED |
| D2 guided / D2 off | 10/7 | +2.344 | -9.545 to+14.022 | UNCERTAIN |

Intervals use the retained paired Clopper–Pearson envelope. All seven
development comparisons are exploratory. Equal80/128 counts do not establish
1pp noninferiority. Independent re-scoring reproduces all binary outcomes and
decisions; Linux/macOS interval endpoints differ by at most1.1463e-14, within
the original1e-10 statistics criterion. Independent Decimal300 and SciPy270
numeric checks also pass that criterion.

## Correctness and capacity

The backend-matched full-checkpoint FP32/Triton gate passes12 cases,108 readouts,
96 forced continuations,82944 checks of each K/V history and21 lifecycle checks
at the original absolute/relative1e-4 criterion. Maximum hidden/K/V/logit/
selected-logprob errors are1.90735e-5/3.05176e-5/9.03010e-6/3.43323e-5/1.33514e-5.
Unchanged readouts are exact. The previous full FP32 gate used Torch attention;
its result was insufficient to qualify this Triton screen.

The separate BF16/Triton gate failed its first P4 prefill state check:
maximum difference0.23046875,998/2048 elements outside the frozen0.02
absolute/relative criteria. No reduced-depth decode or later KV/logit/lifecycle
check was reached. Original failure and thresholds remain; BF16 is unqualified
for this quality screen.

FP32 KV requires393216 bytes/token/depth, or24MiB per16-token four-plane block
group. All eight canonical-tokenized cohorts pass the configured pool bound:
maximum span1415, required groups1367, available groups2730. P4D2 retains the
P4 prefix planes. Peak allocated memory rounds to69.4261GiB in every arm;
guided reference peak is131072 bytes. Off's capability-gated work counters do
not measure actual body work.

The original W5 capacity targets remain separate from the P512 candidate-cost
recipe. CPU bounds round each P+128−1 computed span to16-token R4 groups:

| Target | B / C | Target A | Input tokens | FP32 R4 KV GiB | Static status | Exact native E2E |
|---|---|---:|---|---:|---|---|
| L1 | 1 /1 | 1 | 512 | 0.938 | KV bound fits | NOT_RUN |
| T32 | 32 /32 | 32 | 512 | 30.000 | KV bound fits | NOT_RUN |
| Q64 example | 32 /64 | 32 | 512/2048 mixed | 66.000 | First32 target exceeds pool | NOT_RUN |
| T64_short | 64 /64 | 64 | 256 | 36.000 | KV bound fits | NOT_RUN |
| C8_long | 8 /8 | 8 | 2048 | 25.500 | KV bound fits | NOT_RUN |
| C4_longer | 4 /4 | 4 | 8192 | 48.750 | KV bound fits | NOT_RUN |
| A32_large | 32 /32 | 32 | 2048 | 102.000 | PLANNED_CAPACITY_EXCEEDED | NOT_RUN |

These bounds exclude weights, activations, scratch, Graph buffers and actual
free memory. All positions fit the model's65536 limit. Q64 uses an example
alternating mix's first32 residents; the64-request queue is unmeasured, and the
final mix/arrival rule remains unfrozen. A32/P2048 exceeds the64GiB pool before
execution; it is not an observed OOM. Actual A/S and long-context compatibility
are still unverified. See the unchanged [preflight table](../benchmarks/results/reloop-20261006/W5_CAPACITY_PREFLIGHT-v1.md)
and [raw page bounds](../benchmarks/results/reloop-20261006/w5-fp32-capacity-preflight-v1.json).
The initial totalC cost trace does not qualify the original default4C refill.

The complete [W5 input preparation](../benchmarks/results/reloop-20261006/w5-workload-preparation/W5_WORKLOAD_PREPARATION-v1.md)
now retains all7 targets and820 unique request templates with original4C
closed-loop replacement, explicit length ordering/scheduler fields, three
selected arms and two fresh rounds. [Exact five-file hashes](../benchmarks/results/reloop-20261006/w5-workload-preparation/scientific-files.json)
bind the CPU builder, workload manifest, trace templates and review. These
inputs are not canonical-tokenized, capacity-admitted or GPU-run; no actual
A/S, refill, quality or speed is inferred. A new device requires a fresh
matching numerical gate, local baselines and a complete operational protocol.

## Complete fixed-work cost and frozen confirmation

The complete P512/output128 case runs true D4 off, matched D3 off and D3 guided
at B16/totalC32, observed A=S16 and common64GiB R4 FP32 KV. Each of the six fresh
processes completes five warmups and five measurements. Order is A–B–C / C–B–A;
all60 trials,1920 requests and245760 tokens passed the frozen raw auditor.
InitialC/totalC has no refill; it does not qualify the plan's default4C trace.

| Arm | Mean tokens/s | Mean TTFT ms | Mean TPOT ms | Peak allocated GiB | Mean matched speed vs D4 off | Two-fresh mean ratio range |
|---|---:|---:|---:|---:|---:|---|
| P4D4 off | 144.177 | 11945.174 | 74.588 | 69.385046 | 1.000000x | baseline |
| P4D3 off | 167.121 | 10934.280 | 59.380 | 69.385046 | 1.159342x | 1.158870–1.159814x |
| P4D3 guided | 164.701 | 11076.655 | 60.521 | 69.385168 | 1.142596x | 1.140710–1.144482x |

The same 30 measured raw trials also provide the following descriptive tails.
All values were independently recomputed from the byte-identical request traces.

| Metric | P4D4 off | P4D3 off | P4D3 guided |
|---|---:|---:|---:|
| P95 ITL ms | 72.130 | 56.332 | 57.699 |
| P99 native request latency ms | 29402.655 | 25055.404 | 25066.075 |

These are pooled nearest-rank quantiles over 320 measured requests and 40,640
token intervals per arm. Only two fresh environment repeats exist. TTFT includes
queue/admission waiting, and native latency runs from submission through final
delivery. The [latency table and portable raw provenance](../benchmarks/results/reloop-20261006/cost-c32-latency/RESULTS.md)
refer to the same earlier physical H20 and initial-C policy.

The10 matched measured ratios are baseline elapsed/candidate elapsed.
Guidance versus D3off gives a mean speed ratio0.985644x; independently averaged
elapsed overhead is1.475311%. The means of reciprocal ratios are not exact
reciprocals. Guided allocation adds128KiB and reserved memory adds2MiB;
retaining the common R4 pool provides no KV or peak memory reduction.
Neutral core/head observers apply the same wrappers to every arm.

The guided mean is below the predeclared1.15x reference. The execution plan
and pre-read policy define it as a continuation reference rather than a hard
acceptance gate. Both fresh rounds show about14% fixed-work improvement,
so one D3-guided candidate is selected for independent confirmation. The
80/128 versus79/128 development recovery remains UNCERTAIN; selection is not
noninferiority or evidence that guidance creates the depth-reduction speedup.

All99 original payload hashes and outer2407222-byte archive SHA
`06d960ebb83e97eb1dee8af3a98b5c3cc956815eb558c7cdabccc37077ec5cb4`
were verified. The unchanged scientific subset, complete per-request
traces and frozen audit scripts are in [the C32 archive](../benchmarks/results/reloop-20261006/cost-c32-full.tar.gz)
and [its receipt](../benchmarks/results/reloop-20261006/cost-c32-full.json).
The earlier first-process [partial archive](../benchmarks/results/reloop-20261006/cost-c32-first-a.tar.gz)
is retained separately. B32/C64 and the seven-target native/long-context matrix
remain NOT_RUN; fixed-work cost does not establish HTTP or async serving.

The final protocol freezes505 unseen mathematics and148 unseen HumanEval+
IDs, three arms and M2 before any final outputs. It binds actual SELECTED
receipt, the FP32/Triton numerical gate, exact scoring/cache/source hashes,
1024-token natural-stop maximum,2,006,016-token generation budget,1800-second
admission and600-second finite natural drain. The packet is LOCAL_READY_NOT_GRANTED;
no final outputs have been generated. Code reference preparation is complete
for all148 tasks under its separate trusted-reference budget; the candidate
executor retains20M opcodes/5 seconds per call. With148 zero-discordance code
pairs, the interval is still about±3.371pp. Sample size, gate and time guard
remain unchanged if resources expire before a new window.

The actual frozen package passes an independent review of653 unique IDs,
exact505 mathematics ranking/all148 code ordering, seen/development exclusion,
11 input hashes and the complete scorer/cache/budget bindings. The unchanged
[confirmation inputs](../benchmarks/results/reloop-20261006/confirmation-fp32-frozen-inputs.tar.gz)
and [receipt](../benchmarks/results/reloop-20261006/confirmation-fp32-frozen-inputs.json)
record this declaration before outputs. [The review](../benchmarks/results/reloop-20261006/W4_FROZEN_PACKET_REVIEW-v1.md)
qualifies pretokenization inputs only; no final CUDA execution or accuracy is
claimed. Runtime must bind actual canonical tokenization/per-task capacity and
the final global protocol before generating any new output.

At the H20 admission review, only872 seconds remained before the frozen
evidence deadline. The1800-second admission/600-second finite drain could
not fit; the complete mathematics extrapolation alone is about1456 seconds,
with code runtime still unknown. The confirmation request was withdrawn
before staging or tokenization. All1959 planned arm/question completions,
mathematics scoring and code execution remain NOT_RUN; no zero scores,
synthetic outputs or reduced denominators are substituted. The frozen inputs
remain unchanged for a separately declared future run. See [the admission
record](../benchmarks/results/reloop-20261006/confirmation-not-run.json).

## Evidence and reproduction

[Scientific archive](../benchmarks/results/reloop-20261006/w3-fp32-v2.tar.gz)
contains all development/qualification outputs, strict scores, resolved
configuration, tokenizer bounds, SciPy rows, frozen driver and pinned public
support. Each scientific file is an unchanged copy; the archive excludes
checkpoint weights, dependency caches and cloud/SSH controls. Its109 files and
SHA are listed in [the manifest](../benchmarks/results/reloop-20261006/scientific-files.json)
and [archive receipt](../benchmarks/results/reloop-20261006/archive.json).

Extract into a new directory and use the pinned source and checkpoint above.
The included `w3/harness/generate_quality.py` supports `tokenize` and `generate`;
`w3/harness/score_math.py` consumes the tokenized plan and generated records.
For another host, copy the input plan and explicitly freeze its model path,
GPU UUID and finite time budget before tokenization. Keep IDs, all six arms,
scorer and precision settings; retain the original protocol for comparison.
The reproduction support directory is `reproduction/public-entry`.

Historical FP8 pilots, Thinking metadata timing, D4 Graph timing and14 exposed
code arms have their separate source/dtype/data boundaries. They do not supply
this FP32 candidate's fixed-work or independent quality result.


The seven-point W5 input resolver and single-arm native driver are available in [execution preparation](../benchmarks/results/reloop-20261006/w5-execution-preparation/README.md). They retain all820 IDs,4C refill,the three selected arms,two fresh rounds and the original byte-identical observer. Original CLI/unresolved-template rejection receipts and an independent source review are included. Canonical tokenization,native numerical qualification and all seven device runs remain NOT_RUN; these interfaces add no quality,A/S,refill,speed or VRAM result.

The same preparation now includes a per-job raw auditor and the original
seven-case matrix summary. The auditor verifies sampling, guidance, cache and
scheduler configuration before checking every request and raw distribution.
Public-interface checks accept all six actual earlier configurations and keep
all seven cases and 42 missing jobs in an empty matrix. Complete new 4C job
integration awaits actual output; numerical qualification, owning processes,
resource identity and complete offbox archives remain separate requirements.
