# W5 execution preparation

`audit_w5_native_trials.py` checks each complete job's frozen configuration,
all request traces, distributions and repeat parity. `summarize_w5_raw_matrix.py`
retains the original seven cases and compares only ten matching measurements
after all six fresh jobs for a case are present. Their interface and configuration
review is in [native-auditor-interface-review](native-auditor-interface-review/receipt.json),
with source bindings in [native-auditor-source-review.json](native-auditor-source-review.json).
These are prepared interfaces: all seven new native cases remain NOT_RUN.

The resolver freezes all original seven request traces. The arm driver reads those inputs and uses the original native cost observer. Canonical tokenization and all seven GPU points remain **NOT_RUN**.

| Step | Available evidence | Execution status |
|---|---|---|
| Original inputs | Seven cases,820 unique IDs,4C request traces | CPU templates complete |
| `resolve_w5_prompts.py` | Source review and actual CLI receipt | Canonical tokenization NOT_RUN |
| `run_w5_cost_arm.py` | Source review,CLI receipt,unresolved-template rejection | Native GPU NOT_RUN |
| Original cost observer | Completed C32 implementation,unchanged SHA | Seven-point validation NOT_RUN |

The resolver accepts the original `workload-templates.json`,trace,selected confirmation `plan.json`,clean runtime source and pinned local model directory. It checks the original hashes and tokenizer files,requires existing Transformers4.54.1/tokenizers0.21.4,and uses the production tokenizer without special tokens or chat formatting. It repeats the frozen seed token IDs to exact target lengths and writes `resolved-prompts.json` plus `resolved-protocol.json`. No checkpoint weight is loaded and CUDA stays uninitialized. This tokenization has not run yet.

The driver requires resolved inputs,a numerical gate qualified on the actual device,and admission issued by its direct owning controller. The owner verifies the complete clean source,checkpoint bytes,boot/device,original locks,empty-compute state and real capacity. Admission binds protocol/gate SHA,GPU UUID,allowed job,parent PID and completion/archive budget. A historical gate or copied admission cannot qualify a new device or grant a window.

P4 history stays FP32/Triton LAST_EXITED. The arms are true capability-off D4,true capability-off D3,and selected guided D3 with two_head0.3/reference1. Each case preserves its full4C request trace and scheduler fields. Each arm uses five warmups/five measurements;two fresh rounds use A–C–D then D–C–A. Completing all seven cases would produce420 trials,49200 request completions and6297600 output tokens. These counts are planned only.

Extract [the existing C32 archive](../cost-c32-full.tar.gz) into a new directory and pass its `scripts/fp32_candidate_cost_trial.py` as `--observer`. Its SHA256 is `8f581b7c1c2fd474a53c2126519964bb7cd3d7bb18cb5096a99437b72b14e5a9`. The observer records actual A/S,queueing,per-request timing,token/depth/finish traces,allocation/reference use and synchronized final drain. Its original bytes are preserved;no duplicate formatted observer is substituted.

Q64 keeps all256 queued requests when the first-B target exceeds the pool;actual residency must be measured. A32_large requires102GiB of KV alone at FP32 and remains preflight-rejected at64GiB. It requires a larger declared pool/device in a newly frozen protocol. Neither is an observed OOM. Expected CUDA OOM is recorded as failure;no signals or forced termination are used.

The three public preparation scripts differ from their retained originals only by import ordering and literal-string line wrapping. Import bindings and non-import AST match exactly. The index records old/new SHA and initial failed checks. No production source,mathematics,tolerance,scorer,selected candidate or frozen quality denominator changes. The CLI and negative-path checks do not establish native correctness,quality,speed,Graph or HTTP E2E.

The current driver compares the actual hashed numerical gate GPU UUID with
both admitted UUID fields. Earlier format-only reviews remain historical;
the current two-line semantic binding change is recorded in
[actual-gate-uuid-binding-review.json](actual-gate-uuid-binding-review.json).
The complete positive native job still awaits the new device gate and real output.
