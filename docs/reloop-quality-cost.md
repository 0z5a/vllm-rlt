# Re:Loop

Re:Loop tests whether guided readout recovers quality after reducing decode
loops, and whether that recovery lowers measured generation cost. Prompt
prefill retains all four KV planes. The completed public-checkpoint development
screen makes P4D3 guided the provisional cost candidate; independent quality
and matched fixed-work cost are still pending.

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
These ratios describe the development run; fixed-work, fresh-process and HTTP
performance require their own measurements.

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

## Next frozen work

D3 guided is `PROVISIONAL_FOR_COST`. Its point count matches D4 and gains one
answer over D3 off; the intervals remain uncertain. The pre-read policy permits
at most one candidate. A separate fixed128-output, P512 C32/C64 protocol uses
two fresh starts, reversed arm order, five warmups and five measurements for
true D4 off, matched D3 off and D3 guided. Each trial submits and completes C
requests; it does not use the plan's default4C refill trace. Only the first
full-D4 C32 calibration is currently running; the complete cost gate is pending.

Independent505-question mathematics and148-task HumanEval+ confirmation,
family M2, is prepared with148 verified canonical code references. Final
selection and final IDs require actual fixed-work evidence before any final
outputs. With148 zero-discordance code pairs, the interval is still about
±3.371pp. Numerical qualification alone cannot certify independent quality.

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
