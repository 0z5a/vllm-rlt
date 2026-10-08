# Re:Loop frozen H100 code-output diagnosis

This is a read-only diagnosis of the original confirmation. All 148 outputs in each primary arm match the immutable records and exported completions. It does not change generation, stop rules, scoring, references or the original non-inferiority result.

| Arm | Empty completion | Leading `\ndef ` stopped immediately | Empty raw EOS | Nonempty | Base/plus correct |
|---|---:|---:|---:|---:|---:|
| D4 true off | 60/148 | 56/148 | 4/148 | 88/148 | 1/148 |
| D3 guided | 68/148 | 55/148 | 13/148 | 80/148 | 0/148 |

The original plain function-body prompts stopped on a new function/class, main guard or print statement. Many outputs began a new top-level function and stopped before supplying a body; the frozen export correctly produced an empty completion. Both arms also emitted markdown and invalid syntax. These observations explain failure classes without establishing which prompt policy would perform better.

A future prompt-format investigation must be a distinct development experiment with its own frozen protocol and inputs. Reinterpreting or repairing these exposed outputs cannot establish confirmation quality. The third arm still has 132 missing outputs; its 16-output subset remains unscored. No new CUDA generation, speed, memory, serving or SOTA result is claimed. Whole-task pass contribution remains zero.

[Machine-readable diagnosis](H100_CODE_FAILURE_DIAGNOSIS-v1.json).
