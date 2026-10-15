# Evidence for the Triton packed-prefill + split-KV decode PR

Profiling figure for the single-request A/B: A = `main` 9d601ce, B = the PR head 43aa99e.
Both on one H100, Ouro-1.4B BF16, async scheduling + CUDA graphs, fixed 4 loops, a 2K-token
prompt and 64 output tokens, under torch.profiler. The profiled runs were separate from the
timed A/B.

- `figures/timeline-graphs-A-B.svg` — A above B. The top panels show the full request; the
  zoom shows one mid-decode engine step. A and B share one time scale per panel.
- The raw Chrome traces are attached to the release `pr-evidence-triton-kernels-20261015`
  in this fork (`graphs-main.trace.json.gz`, `graphs-pr-43aa99e.trace.json.gz`). Open them
  in Perfetto or chrome://tracing. The host name and output path in them are replaced with
  placeholders.
