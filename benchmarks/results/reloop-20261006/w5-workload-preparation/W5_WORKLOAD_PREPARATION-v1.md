# Re:Loop W5 workload preparation

All seven original cases now have CPU input templates with820 unique request IDs, closed-loop4C totals, complete length order, explicit scheduler fields, three selected arms,5 warmups/5 measurements and two fresh processes in reverse configuration order. No template is capacity-admitted or GPU-tested.

| Case | B | C | Requests/trial | Prompt tokens | First-B target KV GiB | Native result |
|---|---:|---:|---:|---|---:|---|
|L1|1|1|4|512|0.9375|NOT_RUN|
|T32|32|32|128|512|30.0|NOT_RUN|
|Q64|32|64|256|512/2048|66.0|NOT_RUN|
|T64_short|64|64|256|256|36.0|NOT_RUN|
|C8_long|8|8|32|2048|25.5|NOT_RUN|
|C4_longer|4|4|16|8192|48.75|NOT_RUN|
|A32_large|32|32|128|2048|102.0|NOT_RUN|

Q64 alternates512/2048 across all256 requests. Its66GiB first32 target estimate does not reject the queued workload. A32 retains the original102GiB FP32 KV target and needs a larger memory budget/device. A static fit excludes weights, activations, reference/scratch and actual free memory. Neither estimate is observed OOM or evidence of actual A/S.

Before native cost, resolve exact token IDs with the pinned canonical tokenizer, freeze the prompt SHA/order, obtain a new whole-window grant, verify the actual device/source/W2/model, and record actual per-device memory plan/admission. Reuse the byte-identical C32 metadata observer; extend the existing hardcoded P512/C-total caller only through a separately frozen protocol. A hardware change requires all three local baselines.

The H20 W4 window was withdrawn with zero actual completions; all505+148IDs, M2 and1024/1800+600 budgets remain unchanged. This CPU preparation establishes no speed, quality, actual residency or refill result.

Inputs: [workload-templates.json](w5-original-input-templates-v2/workload-templates.json), [prompt-trace-template.json](w5-original-input-templates-v2/prompt-trace-template.json). Verification: [w5-workload-preparation-review-v1.json](w5-workload-preparation-review-v1.json).
