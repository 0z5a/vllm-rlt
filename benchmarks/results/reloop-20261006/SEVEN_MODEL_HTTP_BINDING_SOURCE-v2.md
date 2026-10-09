# Seven-model production string binding preparation

The complete source, tokenizer metadata and original request bindings passed
independent review. Actual production tokenizer execution, model loading,
HTTP serving and CUDA tests have not run.

| Model | Original traces | Requests | Actual string roundtrip |
|---|---:|---:|---|
| Ouro-1.4B | 6 | 6144 | Pending |
| Ouro-2.6B | 6 | 6144 | Pending |
| Huginn-0125 | 6 | 6144 | Pending |
| Nanbeige-4.2-3B | 6 | 6144 | Pending |
| HRM-Text-1B | 5 | 5120 | Pending |
| LoopFormer-3block-8iterations | 5 | 5120 | Pending |
| Parcae-370M | 5 | 5120 | Pending; loader compatibility unverified |

Review checked thirteen frozen source files, thirty-three metadata archive
files and eleven original provenance archive files. All 32 tokenizer metadata
pins match original receipts and reused bytes; all seven tokenizer JSON hashes
also match the original request manifests. Every model/source/tokenizer
revision is bound separately. Parcae uses its independently pinned tokenizer
repository; missing Hugging Face metadata is not fabricated.

All 39 original traces, 39,936 requests and seven arrival files remain unchanged.
The preparer uses production `load_tokenizer` and default `encode(text)`, with
an actual imported-loader source-path assertion. Three origin statements and
three receipt fields leave the remaining V1 syntax tree unchanged. Syntax, explicit
project Ruff configuration, CLI and seven inactive pre-import rejection
checks passed locally. These are source checks, not tokenizer qualification.

The actual CPU run must preserve complete original IDs, decoded strings,
production re-encoded IDs and mismatch records. A mismatch returns failure;
any canonical-text companion remains separate and unfrozen. Default BOS and
original native inputs are preserved. Nanbeige's earlier static prefix conflict
still requires actual production execution; it is not a passed roundtrip.

These 1024-request traces belong to the independent Loopquant supplement.
They do not replace Re:Loop's original 4C workloads or establish C128 serving,
steady-state performance, quantization quality or whole-task completion.
Whole-task pass contribution remains zero.

[Scientific bindings and immutable hashes](SEVEN_MODEL_HTTP_BINDING_SOURCE-v2.json).
