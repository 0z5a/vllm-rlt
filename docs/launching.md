# Launching vllm-rlt

After completing the [installation](../README.md#getting-started), choose a
command-line batch, the Python API, or an HTTP server. Run the examples from
the repository root in the environment where the project is installed.

[Command-line inference](#command-line-inference) · [Python API](#python-api) ·
[HTTP serving](#http-serving)

## Command-line inference

```bash
vllm-rlt \
  --model ByteDance/Ouro-1.4B \
  --device cuda \
  --attention-backend triton \
  --prompt 'The capital of France is' \
  --prompt '2 + 2 =' \
  --max-tokens 32 \
  --exit-threshold 0.7
```

Repeat `--prompt` to submit a batch. `--model` also accepts a local checkpoint
directory. BF16 is the default; the official model and tokenizer are pinned to
revision `574fa66cb8bf5abdc979642d01cf2b79b16bfab1`. Model loading uses native
code and safetensors, without `trust_remote_code`.

To check the installation without downloading weights or using a GPU:

```bash
OMP_NUM_THREADS=1 vllm-rlt --toy --max-tokens 4 --exit-threshold 0.7
```

The toy model is tiny and randomly initialized; it checks engine execution,
not language quality. On shared GPU hosts, run GPU examples through your local
reservation system and set device visibility for the allocated GPUs.

## Python API

```python
from vllm_rlt import LLM, SamplingParams

llm = LLM(
    "ByteDance/Ouro-1.4B",
    device="cuda",
    attention_backend="triton",
)

outputs = llm.generate(
    ["The capital of France is", "2 + 2 ="],
    SamplingParams(
        max_tokens=32,
        temperature=0.0,
        min_loops=2,
        max_loops=4,
        exit_threshold=0.7,
    ),
)

for output in outputs:
    print(output.text)
    print("Exit depths:", output.exit_depths)
```

The API accepts text prompts or token-ID lists and returns results in input
order. Prompt prefill runs at full depth; the exit threshold controls subsequent
decode. Set `exit_threshold=1.0` for fixed-depth execution. The first generated
token comes from full-depth prefill, which is reflected in `exit_depths`.

For dynamic arrivals and step-by-step output, use `llm.engine.add_request(...)`,
`llm.engine.step()`, and `llm.engine.abort_request(...)`. See the
[engine design](design.md) for the execution model.

## HTTP serving

Start a resident model:

```bash
vllm-rlt-serve \
  --model ByteDance/Ouro-1.4B \
  --device cuda \
  --attention-backend triton \
  --host 127.0.0.1 \
  --port 8000
```

Send a streaming completion request:

```bash
curl -N http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "ByteDance/Ouro-1.4B",
    "prompt": "The capital of France is",
    "max_tokens": 32,
    "temperature": 0,
    "exit_threshold": 0.7,
    "stream": true
  }'
```

Concurrent requests share continuous engine batching. The server also exposes
`GET /health` and `GET /v1/models`. It implements a bounded subset of the
OpenAI completions API; chat completions are not implemented. See the
[serving guide](serving.md) for supported fields, request limits, and
benchmark client usage.
