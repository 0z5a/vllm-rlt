"""One reserved BF16 serving compatibility pass, not a performance comparison.

Prepare the model/tokenizer and client environment before running. The script
retains raw HTTP/client/server results, checks direct-engine output equivalence,
and terminates only its own server process. See docs/serving.md for the contract.
"""

import argparse
import asyncio
import gc
import hashlib
import importlib.metadata
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import aiohttp

MODEL = "ByteDance/Ouro-1.4B"
PROMPTS = [
    "The capital of France is",
    "2 + 2 =",
    "Write a short sentence about the ocean.",
    "你好，请介绍一下你自己。",
]
POLICIES = [1.0, 1.0, 0.7, 0.7]


def save(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def model_args(model_path):
    return SimpleNamespace(
        model=str(model_path),
        tokenizer=str(model_path),
        revision=None,
        tokenizer_revision=None,
        device="cuda",
        dtype="bfloat16",
        attention_backend="triton",
        num_blocks=512,
        block_size=16,
        max_num_seqs=8,
        max_num_batched_tokens=128,
        mode="refill",
    )


def params(threshold):
    return dict(
        max_tokens=8,
        temperature=0.0,
        seed=0,
        ignore_eos=True,
        min_loops=2,
        max_loops=4,
        exit_threshold=threshold,
    )


def references(model_path):
    import torch

    from vllm_lt.entrypoints.serve import load_engine
    from vllm_lt.sampling_params import SamplingParams

    engine, tokenizer = load_engine(model_args(model_path))
    results = []
    for i, (prompt, threshold) in enumerate(zip(PROMPTS, POLICIES)):
        engine.add_request(str(i), tokenizer.encode(prompt), SamplingParams(**params(threshold)))
        output = None
        for _ in range(1000):
            if not engine.has_unfinished_requests():
                break
            for output in engine.step():
                pass
        assert output is not None and output.finished
        results.append(
            {
                **asdict(output),
                "text": tokenizer.decode(
                    output.token_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                ),
            }
        )
    assert engine.cache_manager.num_used_blocks == 0
    del engine, tokenizer, output
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    return results


async def http_checks(port, process, started, expected, out):
    url = f"http://127.0.0.1:{port}"
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=300)) as session:
        observations = []
        for _ in range(600):
            if process.poll() is not None:
                raise RuntimeError("server exited before readiness")
            try:
                async with session.get(url + "/health") as response:
                    observations.append(response.status)
                    if response.status == 200:
                        break
            except aiohttp.ClientError:
                pass
            await asyncio.sleep(0.2)
        else:
            raise RuntimeError("readiness deadline exceeded")
        save(
            out / "readiness.json",
            {
                "process_to_ready_seconds": time.monotonic() - started,
                "health_statuses": observations,
            },
        )

        async def request(i):
            payload = {"model": MODEL, "prompt": PROMPTS[i], **params(POLICIES[i])}
            async with session.post(url + "/v1/completions", json=payload) as response:
                result = await response.json()
                save(
                    out / f"http-{i}.json",
                    {"status": response.status, "request": payload, "response": result},
                )
                assert response.status == 200, result
                assert result["choices"][0]["text"] == expected[i]["text"], (i, result, expected[i])
                assert result["choices"][0]["finish_reason"] == expected[i]["finish_reason"]
                assert result["usage"]["completion_tokens"] == len(expected[i]["token_ids"])
                assert result["usage"]["prompt_tokens"] == len(expected[i]["prompt_token_ids"])
                return result

        # No benchmark readiness probe or warmup may precede this actual request.
        await request(0)
        await asyncio.gather(*(request(i) for i in range(1, len(PROMPTS))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--client", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=18022)
    args = parser.parse_args()
    args.model_path = args.model_path.resolve()
    args.client = args.client.resolve()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    out = args.output
    # Check scheduler ownership before importing code that executes on a device.
    from vllm_lt.benchmarks.runner import environment

    ownership = environment()
    client_versions = json.loads(
        subprocess.check_output(
            [
                str(args.client.parent / "python"),
                "-c",
                "import importlib.metadata as m, json; "
                "print(json.dumps({n:m.version(n) for n in "
                "['vllm','torch','transformers','aiohttp']}))",
            ],
            text=True,
        )
    )
    if client_versions["vllm"] != "0.28.0":
        raise ValueError("this contract pins the unmodified vLLM 0.28.0 client")
    import torch

    torch.set_num_threads(1)
    repo = Path(__file__).resolve().parents[1]
    manifest = {
        "source_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True
        ).strip(),
        "source_status": subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=repo, text=True
        ),
        "ownership": ownership,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "cpu_threads": 1,
        "model": MODEL,
        "prepared_path": str(args.model_path),
        "model_files_sha256": {},
        "client": str(args.client),
        "client_versions": client_versions,
        "versions": {
            name: importlib.metadata.version(name)
            for name in ["torch", "triton", "transformers", "aiohttp"]
        },
        "dtype": "bfloat16",
        "attention_backend": "triton",
        "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "allow_bf16_reduced_precision_reduction": (
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
        ),
        "gpu_name": torch.cuda.get_device_name(),
        "contract": {
            "hypothesis": (
                "Serving preserves the direct-engine contract and accepts the unmodified client"
            ),
            "success": (
                "All four text/usage/finish comparisons pass; each client cell "
                "has 16 successful requests, each with 16 output tokens"
            ),
            "variable": "HTTP arrival pattern; reference uses serial direct calls",
            "prompts": PROMPTS,
            "policies": [params(t) for t in POLICIES],
            "budget": (
                "One first-request/concurrency feasibility pass, then one serial, "
                "concurrent and finite-rate compatibility invocation; no retries"
            ),
            "comparison": "functional qualification only; no speedup or tail-latency claim",
        },
    }
    hash_started = time.monotonic()
    model_files = [
        args.model_path / name
        for name in ["config.json", "tokenizer.json", "tokenizer_config.json"]
    ]
    model_files.extend(sorted(args.model_path.glob("*.safetensors")))
    for file in model_files:
        digest = hashlib.sha256()
        with file.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        manifest["model_files_sha256"][file.name] = digest.hexdigest()
    manifest["prepared_file_hash_seconds"] = time.monotonic() - hash_started
    save(out / "manifest.json", manifest)
    if manifest["source_status"]:
        raise ValueError("commit the source before running the frozen serving contract")
    prepare_started = time.monotonic()
    expected = references(args.model_path)
    save(out / "direct.json", expected)
    save(
        out / "preparation.json",
        {
            "direct_reference_seconds": time.monotonic() - prepare_started,
            "downloads": "none; model and environments prepared separately",
        },
    )
    server_command = [
        sys.executable,
        "-m",
        "vllm_lt.entrypoints.serve",
        "--model",
        str(args.model_path),
        "--tokenizer",
        str(args.model_path),
        "--port",
        str(args.port),
        "--num-blocks",
        "512",
    ]
    save(out / "server-command.json", server_command)
    with (out / "server.log").open("w") as log:
        started = time.monotonic()
        process = subprocess.Popen(server_command, cwd=repo, stdout=log, stderr=subprocess.STDOUT)
        save(out / "server-process.json", {"pid": process.pid})
        try:
            asyncio.run(http_checks(args.port, process, started, expected, out))
            cells = [("serial", 1, "inf"), ("concurrent", 8, "inf"), ("finite-rate", 4, "2")]
            for name, concurrency, rate in cells:
                command = [
                    str(args.client),
                    "bench",
                    "serve",
                    "--backend",
                    "openai",
                    "--base-url",
                    f"http://127.0.0.1:{args.port}",
                    "--endpoint",
                    "/v1/completions",
                    "--model",
                    MODEL,
                    "--tokenizer",
                    str(args.model_path),
                    "--dataset-name",
                    "random",
                    "--random-input-len",
                    "128",
                    "--random-output-len",
                    "16",
                    "--random-range-ratio",
                    "0",
                    "--num-prompts",
                    "16",
                    "--seed",
                    "20260912",
                    "--max-concurrency",
                    str(concurrency),
                    "--request-rate",
                    rate,
                    "--ignore-eos",
                    "--extra-body",
                    json.dumps(
                        {
                            "temperature": 0,
                            "seed": 0,
                            "min_loops": 2,
                            "max_loops": 4,
                            "exit_threshold": 1.0,
                        }
                    ),
                    "--percentile-metrics",
                    "ttft,tpot,itl,e2el",
                    "--metric-percentiles",
                    "50,90,99",
                    "--num-warmups",
                    "0",
                    "--save-result",
                    "--save-detailed",
                    "--disable-tqdm",
                    "--result-dir",
                    str(out),
                    "--result-filename",
                    name + ".json",
                ]
                save(out / f"{name}-command.json", command)
                client_env = dict(os.environ, VLLM_PLUGINS="")
                with (out / f"{name}.log").open("w") as client_log:
                    subprocess.run(
                        command,
                        cwd=repo,
                        env=client_env,
                        stdout=client_log,
                        stderr=subprocess.STDOUT,
                        timeout=300,
                        check=True,
                    )
                data = json.loads((out / f"{name}.json").read_text())
                assert data["completed"] == 16, data
                assert data["failed"] == 0, data
                assert data["total_output_tokens"] == 256, data
                assert data["output_lens"] == [16] * 16, data["output_lens"]
                assert [len(intervals) for intervals in data["itls"]] == [15] * 16
                assert not any(data.get("errors", [])), data.get("errors")
        finally:
            process.send_signal(signal.SIGTERM)
            try:
                process.wait(timeout=45)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            gc.collect()
            torch.cuda.empty_cache()
            save(
                out / "cleanup.json",
                {
                    "server_pid": process.pid,
                    "server_returncode": process.returncode,
                    "parent_cuda_allocated": torch.cuda.memory_allocated(),
                },
            )
    assert "engine cleanup: requests=0 kv_blocks=0" in (out / "server.log").read_text()
    save(out / "result.json", {"passed": True, "http_direct_matches": 4, "benchmark_cells": 3})


if __name__ == "__main__":
    main()
