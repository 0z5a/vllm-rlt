# Re:Loop H100 confirmation, common48GiB — incomplete

Original653 IDs×3 arms were preserved. Native FP32/eager Triton/B16 generation completed1827 of1959 outputs and213,498 tokens;132 matched-off code outputs were NOT_RUN_BUDGET. All1959 original keys, canonical prompt tokens, emitted IDs/depths/configuration and drain were checked independently. Original1800-second batch admission,600-second natural current-batch drain,1024-token maximum,2,006,016-token cap andM2/1pp scoring were unchanged.

| Arm | Mathematics correct /505 | Code correct /148 | Generation complete /653 |
|---|---:|---:|---:|
| D4 true off | 316 (62.57%) | 1 (0.68%) | 653 |
| D3 guided, ref1 w0.3 | 301 (59.60%) | 0 (0.00%) | 653 |
| D3 true off | 287 (56.83%) | Unscored;16 generated,132 not run | 521 |

| Comparison | Difference | Frozen M2 interval | Result |
|---|---:|---:|---|
| Mathematics guided vs D4, primary | −2.970pp | [−9.009,+3.129]pp | UNCERTAIN |
| Code guided vs D4, primary | −0.676pp | [−4.749,+3.367]pp | UNCERTAIN |
| Mathematics guided vs D3 off, exploratory | +2.772pp | [−3.222,+8.710]pp | UNCERTAIN |
| Code guided vs D3 off, exploratory | — | — | INCOMPLETE |

The selected candidate does not establish1pp non-inferiority. Code accuracy is near zero under the exact frozen prompts/raw completions and original cooperative scorer; no output repair or reference regeneration was applied. The matched-off code arm was not scored on its16-output subset. Missing work retains the148/1959 denominators. The overall experiment remains INCOMPLETE, even though both primary arms were fully generated and scored.

| H100 matched fixed-work speed comparison | Result |
|---|---|
| D3 off vs D4 | NOT_RUN: original complete C32 window did not fit |
| D3 guided vs D4 | NOT_RUN: original complete C32 window did not fit |

These natural-stop quality outputs do not supply a fixed-work speed ratio, memory saving, HTTP/async serving result or SOTA comparison. The earlier H20 fixed-work table remains a separate device/pool experiment. Original seven-case W5/high-C/multi-BS scope is still pending.

All1300 manifested task payloads plus4 archive-control files and the whole archive were independently checked. Original source,model,driver,raw outputs,JIT/configuration/controls and scoring inputs were retained. Scientific Node child/controller/step returned0; the original connection returned255 and is preserved. Later complete offbox verification/copy returned0. Fresh final GPU/actor/locks release after the original allocation ended was not observed. These return and transport facts are distinct from scientific completeness.

Both original complete148-ID code scorers joined naturally with0. Their overall controller returned1 because the third arm is incomplete. The custom cooperative candidate budget remains20M opcodes/5 seconds per call; this is not official EvalPlus process-timeout execution. [Scientific allowlist summary](H100_48GiB_CONFIRMATION-v1.json) binds actual protocol/records and full archive hashes; operational/raw access metadata stays private.
