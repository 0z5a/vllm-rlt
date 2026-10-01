# Training rollout and full-weight publication

`LLM` and `LLMEngine` expose the same full-update operations. `LLM` also accepts
an existing `LLMEngine` or `PDEngine`, so framework adapters use the same generation
and publication calls with either backend.

```python
llm.start_weight_update(version=1)
for chunk in converted_hf_weights:
    llm.update_weights(chunk)  # Iterable[(physical_parameter_name, torch.Tensor)]
llm.finish_weight_update()
outputs = llm.generate(prompts, SamplingParams(
    max_tokens=128, temperature=1.0, seed=42, logprobs=0,
))
```

Each output contains aligned selected-token `log_probs`, `exit_depths`, the
committed `weight_version`, and its effective `sampling_params`. `logprobs=None`
keeps the ordinary inference path. Reported parameters resolve the model's
default `max_loops` and cap `top_k` at its vocabulary size. `logprobs=0` requests only the emitted token's
probability; top-N alternatives are not implemented.

`logprobs_mode="raw"` is the default: FP32 log-softmax of unscaled model logits.
`"processed"` returns probabilities after positive temperature and top-k/top-p
filtering. Greedy requests return the raw score in either mode. Speculative
outputs use the verified **target** policy's score, including correction/bonus
tokens; proposal or rejection-residual probabilities are never exposed as the
policy score. Explicit `stop_token_ids` apply even with `ignore_eos=True`.
`seed=None` creates an independent entropy-seeded request generator.

Publication requires an idle engine. Each chunk is checked for duplicate names,
unknown names and mismatched shapes before that chunk is copied. `finish_weight_update`
requires every physical parameter exactly once. Copies preserve parameter addresses
and captured graphs; a fresh transaction invalidates all cached prefix KV. If a
transfer fails, generation stays blocked until a fresh full transaction succeeds.
Starting again discards an unfinished transaction. `pause_generation` stops
admission while existing requests drain; `resume_generation` requires publication
to be complete. `abort_request` remains available to cancel requests before an update.

PDEngine waits for all transfers to release, broadcasts CPU-staged weight chunks
to every P/D worker, and commits a version only after every peer acknowledges it.
Recovery uses a version newer than **all attempted** PD versions because some
workers may already have committed. With `terminate_workers_on_failure=False`, fatal workers retain their registered
memory rather than receive forced process signals; normal `close()` remains
cooperative. No CUDA tensors change ownership over IPC;
NIXL's existing GPU KV path and registrations remain intact. Prefix invalidation
does not replace registered KV storage. PD worker control errors return replies
without terminating the worker.

Existing exit, async and speculative configuration rules still apply. In particular,
upstream self-speculation uses a full-depth target and synchronous execution;
upstream PD does not implement self-speculation. Publication adds no such restriction
to ordinary early-exit or async rollouts. Trainers must replay full-depth prompt
prefill and the returned decode depths rather than assume the entire sequence used K.

## Validation

```bash
python -m unittest tests.test_training_contract tests.test_pd_weight_control -v
CONTRACT_DEVICE=cuda python -m unittest tests.test_training_contract -v
python benchmarks/training_contract.py /models/Ouro-1.4B timings.json
UCX_TLS=tcp,cuda_copy python benchmarks/nixl_preflight.py
UCX_TLS=tcp,cuda_copy python benchmarks/pd_training_contract.py pd.json --cuda-graphs
UCX_TLS=tcp,cuda_copy python benchmarks/pd_training_contract.py full-pd.json \
  --model /models/Ouro-1.4B --cuda-graphs --max-num-seqs 1
```

CPU and Thor CUDA tests cover raw/processed probabilities against an independent
dense oracle, speculative correction/bonus alignment, publication, pointer stability,
prefix invalidation, early exit, async stop/abort and entropy seeds. The PD control
test drives two real CPU engines and checks recovery after one peer commits and
another fails. It uses in-process control delivery; it does **not** validate IPC or
NIXL transport. The separate two-A100 run uses real spawned workers and NIXL:
FP32 multi-request and full BF16 matched-batch checks cover three publications,
raw/processed scores, graph replay, prefix reuse, explicit stop/abort and recovery
to version five. See [PD evidence](validation/training-contract-a100-20261001/README.md).
No PD workers were launched on single-GPU Thor, where NIXL is absent.

This is a draft implementation of the
[Phase 1 contract proposed in issue 70](https://github.com/ThinkFlowLab/vllm-rlt/issues/70#issuecomment-5888101273)
on upstream `ecb1f8b505b7e831815b40aec3b4598619cca23a`; formal upstream agreement
remains outside the scope of this contributor draft.
