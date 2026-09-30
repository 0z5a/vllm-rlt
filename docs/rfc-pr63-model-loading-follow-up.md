# RFC: PR #63 Model Loading Follow-up

Date: 2026-09-30. Status: Follow-up implemented and CPU-validated; pending merge.

## Merge decision

PR #63 was merged as `17ac0d3`. This follow-up implements the agreed loading
fixes on that main baseline. The original behavior and acceptance criteria are
retained below for traceability. GPU parity remains unverified by this work.

Review snapshot: head `40670cee67b8ab2cd6a12ff07f8a50b870629f7f`,
base `4e916d1cd4318eb7e14a5842ffde0efa9726b8b7`.
Recheck these findings against the actual merged commit before implementation.

## Required follow-up

| Item | Current behavior at reviewed head | Expected behavior and acceptance check |
| --- | --- | --- |
| Caller revision | AutoModel maps the full `ByteDance/Ouro-1.4B` ID to a recipe and replaces `revision=None` with `574fa66...`. The tokenizer retains `None`. | Preserve the caller's source and revision, including `None`. Pinned recipe metadata must be selected explicitly and applied consistently. Test both `None` and explicit revisions through public loading entrypoints. |
| Local source identity | `resolve_local_model_path` matches `artifacts/models/<basename>` regardless of the requested Hub namespace or revision. | Accept explicit local paths; resolve Hub caches by complete repository identity and requested revision. A different namespace with the same basename must not silently load that directory. |
| Tokenizer source | Recipe aliases resolve for model weights, but LLM keeps the original alias for tokenizer loading; serving also loads the tokenizer from the unresolved argument. | Resolve source/revision once and use the result for both model and default tokenizer. Preserve explicit tokenizer overrides. Test text generation and serving with an alias and a local checkpoint. |
| PD download approval | PDWorker hardcodes `allow_download=True`; pd_serve has no corresponding approval flow. | Interactive entrypoints accept `y/yes` or an explicit approval flag. Non-interactive calls without approval fail clearly before downloading. Propagate authorization through PD rather than granting it inside workers. Test approval, refusal, and non-interactive use without requiring GPUs. |
| Serving model identity | `_is_ouro_model` guesses from repository names and defaults unknown remote names to Ouro. | Determine model type from the resolved `config.json`; surface config errors. A Nanbeige checkpoint under a custom name must not receive the Ouro-only tokenizer gate. |
| External parity evidence | The committed real-checkpoint test runs CPU BF16, while the author report describes GPU parity. | Label the existing evidence as CPU, or provide the actual reproducible GPU test and logs tied to a source SHA, checkpoint revision, and environment. CPU results do not establish GPU parity; the discrepancy is not proof of a GPU numerical defect. |

Preferred implementation: share source/revision resolution and authorization
across CLI, LLM, serving, and PD, then dispatch by configuration. Separate fixes
in each entrypoint are possible but would retain the inconsistency that caused
these failures. Keep model arithmetic unchanged in this follow-up.

## Already addressed

- Checkpoint constants moved to recipes, although caller semantics still need fixing.
- Both tiny configuration presets moved into test helpers.
- RMSNorm and RoPE extracted into shared layers with the precision behavior preserved.
- Dense reference implementations moved under tests/reference and kept independent.
- AutoModel config dispatch now surfaces fetch errors and unsupported model types.

## Validation record and closure

At the reviewed head, targeted CPU tests returned **138 passed, 1 skipped,
2 deselected**. The official tokenizer was not cached; the two external parity
tests were excluded. Python environment: PyTorch `2.13.0+cu130`, Transformers
`5.14.1`; GPU visibility and Hub network access were disabled.

Command:

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m pytest \
  tests/test_auto_model.py tests/test_recipes.py tests/test_normalization.py \
  tests/test_rotary_embedding.py tests/test_ouro.py tests/test_nanbeige.py \
  tests/test_serving.py tests/test_engine.py -q -rs \
  -k 'not parity_against_real_checkpoint and not parity_against_official_hf'
```

Mock reproductions confirmed revision replacement, unresolved tokenizer alias,
and cross-namespace basename matching. PD approval and serving identity findings
were established by static call-chain review. No GPU or real-checkpoint parity
was run by this review.

- [x] Implement the loading and authorization fixes with regression coverage.
- [x] Clarify parity evidence and its device scope: the committed real-checkpoint
  comparison is CPU BF16; no GPU parity claim is made by this follow-up.
- [ ] Merge the follow-up PR; GPU parity remains outside this validation.

## Implemented loading contract

- Pass explicit local directories directly, including `artifacts/models/...`.
  Hub IDs no longer search arbitrary artifact directories by basename.
- Recipe aliases expand repository IDs only. To use the recipe's pin, pass
  `revision=recipe.revision` explicitly. `revision=None` is sent unchanged to
  Hub resolution; subsequent consumers use that resolved snapshot commit so
  model configuration, weights, and the default tokenizer agree.
- Complete HF snapshots are found offline by repository and revision. Missing
  shards do not grant permission to download. Local parse/loading errors surface.
- Python LLM/AutoModel/PDEngine calls default to `allow_download=False`. CLI,
  serve, and pd-serve share `y/yes` confirmation and `-y/--yes`; PD workers inherit
  approval instead of granting it. Tokenizer loads are local-only without approval.
- Explicit tokenizer source/revision overrides remain independent of the resolved
  model snapshot. Serving checks model type from config before loading weights.

Example: `LLM("nanbeige", allow_download=True)` now resolves the canonical model
and tokenizer source. `vllm-rlt-pd-serve --model org/model --yes` explicitly
approves downloads. Offline callers can use complete HF caches or explicit paths.

## Follow-up validation

On branch `fix/model-loading-follow-up`, based on `17ac0d3`, the complete offline
CPU selection passed: **402 passed, 27 skipped, 113 deselected**. The command was
`python -m pytest -m 'not gpu' -k 'not parity_against_real_checkpoint and not parity_against_official_hf' -q`
with GPU visibility disabled and both Hub and Transformers offline. Skips cover
missing lm-eval and the uncached official tokenizer. Ruff lint and format checks
passed. The two profiler cleanup tests now create an actual local config fixture
so they continue to exercise cleanup after model-source validation.

New regressions cover exact caller revisions, cache identity and completeness,
alias text generation, config-based serving dispatch, interactive/non-interactive
approval, rejection before PD process creation, worker authorization propagation,
and resolved snapshot consistency. No GPU or real-checkpoint experiment was run.
