# Ouro terminal-output CUDA qualification

All 16 paired cases pass on official Ouro-1.4B BF16 weights: 32 arms, 1,360 requests and 10,880 generated tokens. Terminal-only and cumulative output return exactly equal final tokens, exit depths, selected scores, sampling metadata and policy versions. Every arm drains its KV allocations and scheduler queues, and request IDs are reused between paired arms.

This is a finite native-engine correctness campaign. It does not measure a speedup, reward quality, full RL, long-duration stability, public Safetensors loading on CUDA, or a GPU memory reduction.

| Execution | Actual resident requests | Paired cases | Final tokens/depths/scores | C64 process peak allocated | Speed ratio |
|---|---|---:|---|---:|---|
| Synchronous Triton | 1 / 4 / 16 / 64 | 4 | Exact | 3.842 GiB | Not measured |
| Async, one stream | 1 / 4 / 16 / 64 | 4 | Exact | 4.049 GiB | Not measured |
| Async, multiple streams | 1 / 4 / 16 / 64 | 4 | Exact | 4.299 GiB | Not measured |
| Synchronous CUDA Graph | 1 / 4 / 16 / 64 | 4 | Exact | 4.257 GiB | Not measured |

Each case enqueues `2C` requests, so it includes queued admission beyond the actual resident limit. Prompts contain 2–9 synthetic vocabulary IDs; each request produces eight sampled tokens at exactly four loops. Sampling uses temperature 0.8, request-specific seed `31800 + request_index`, processed selected scores and `ignore_eos=True`. The synthetic inputs test execution; they are not a language-quality dataset. There are 1,536 KV pages of four tokens, occupying 1.125 GiB. Peak used pages at C64 are 896, with zero left after every arm.

The cumulative arm emits lengths 1–8 for each request; the terminal arm emits only length 8. The same engine runs both arms using the same request IDs and seeds. The frozen score bound was absolute error 0.02; every observed score is exactly equal, so no tolerance relaxation was needed. The finite scheduler budget is 8,192 calls per arm; no timeout terminates a process.

Graph counters are cumulative across each pair: C1/C4/C16 each capture one graph and replay it 112 times; C64 captures three and replays 114 times. No graph fallback is reported. Sampling and host scheduling remain eager; this is recurrent-core Graph qualification, not whole-engine capture. Comparisons are within each execution mode; no cross-mode equivalence or acceleration claim is inferred.

## Runtime and evidence

- Native source: `7d6e08edf6327797337f361421245539ac0283e8`, the parent of this evidence-only change. All 54 deployed runtime files and five support files match their frozen hashes.
- Model: `ByteDance/Ouro-1.4B`, revision `574fa66cb8bf5abdc979642d01cf2b79b16bfab1`. The 2,869,336,434-byte weight file has SHA256 `58872a72616c736595b8b7662079c5b12c5a162ec16eae94f21c348dfa9885af`; config SHA256 is `ce9cc13da41591b8b4deca053d7dfee06424c0228628ee862ea86d725bc163f3`. Both are rehashed before loading.
- RTX 5090, Python 3.12.3, existing Torch 2.12.1+cu130 / CUDA 13.0, one CPU thread, Triton attention. No package installation or update.
- The private Torch-only input transport copies each parameter from the validated file into owned storage. Twelve local CPU comparisons against the real public Safetensors loader match every parameter and generation result across three source dtypes, two target dtypes and two KV-head layouts. Nine further CPU checks qualify output collection and reuse. Public-loader CUDA coverage remains pending.
- The other GPU ran a non-timing diagnostic. All 32 arms share one process and fixed order; CUDA allocator peaks are therefore observational process peaks, including retained allocations from earlier arms. The maximum allocated/reserved values are 4,616,098,304 / 4,863,295,488 bytes. They are not independent memory comparisons. A 12 GiB Torch allocator cap excludes CUDA context and other allocators.
- The controller and SSH session exit naturally with code 0. After archive collection, a fresh check finds no model readers, no GPU0 compute process, and acquires/releases the original GPU0 and heavy-IO locks together. The entire reserved window is explicitly returned. Model inputs remain needed for later experiments.

Raw observations and all final request objects: [`arms.jsonl`](../benchmarks/results/ouro-output-gpu/arms.jsonl). Independent readback: [`audit.json`](../benchmarks/results/ouro-output-gpu/audit.json). Per-case comparisons, actual environment and success receipt are in the same directory. The collected archive SHA256 is `13322b5d0b6e65deb05858294c1ec8c62a5571b52919ca75afb397bbef9318ab`.

## Reproduction

The exact private qualification package is retained under [`benchmarks/qualification/ouro-output`](../benchmarks/qualification/ouro-output). Its manifest records the executed files, not this later documentation commit. Construct a bundle with `runtime/vllm_rlt` from the pinned native commit and copy the retained `support` directory and manifest into it. No installation is required. With a separately authorized, idle GPU and the pinned local model files:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 \
PYTHONPATH="$BUNDLE/runtime:$BUNDLE/support" \
python "$BUNDLE/support/qualify.py" --bundle "$BUNDLE" \
  --model "$MODEL" --output "$BUNDLE/new-results"
```

The retained `support/run.sh` is the exact host-specific launcher used for this run. Its filesystem and lock paths are evidence, not portable defaults. The reader helpers are private qualification adapters; they do not replace or qualify the public checkpoint-loading API.
