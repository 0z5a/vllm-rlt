"""Paired old/new production classes, using the unchanged PR82 measurement functions."""

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["parity", "timing"], required=True)
    p.add_argument("--start", type=int, choices=(0, 1), default=0)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    a = p.parse_args()
    ROOT = Path(__file__).resolve().parents[1]
    MODEL = a.model
    sys.path[:0] = [str(ROOT), str(ROOT / "benchmarks")]
    import huginn_pr82_review as benchmark
    from huginn_pr82_review import read_json, sha_file, write_json

    HEAD = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    BASE_HEAD = subprocess.check_output(
        ["git", "-C", str(a.baseline), "rev-parse", "HEAD"], text=True
    ).strip()
    import torch
    from transformers import AutoTokenizer

    import vllm_rlt.engine.llm_engine as engine_module
    from vllm_rlt.models.huginn import HuginnForCausalLM
    from vllm_rlt.worker.model_runner import ModelRunner

    def load(name, relative):
        spec = importlib.util.spec_from_file_location(name, a.baseline / relative)
        m = importlib.util.module_from_spec(spec)
        sys.modules[name] = m
        spec.loader.exec_module(m)
        return m

    old_model = load("pr82_baseline_huginn", "vllm_rlt/models/huginn.py").HuginnForCausalLM
    old_runner = load("pr82_baseline_runner", "vllm_rlt/worker/model_runner.py").ModelRunner
    # Both imported baseline files use unchanged shared dependencies. FA3 is unused
    # by this explicit Triton workload; the new norm module is candidate-only.
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    a.output.mkdir(parents=True, exist_ok=False)
    assert torch.cuda.device_count() == 1, "Set CUDA_VISIBLE_DEVICES to one reserved GPU"
    manifest = read_json(ROOT / "benchmarks/fixtures/huginn-0125-manifest.json")
    for name, expected in manifest["files"].items():
        actual = sha_file(MODEL / name)
        assert all(actual[k] == expected[k] for k in ("size", "sha256")), name
    allowed = {"models/huginn.py", "worker/model_runner.py", "kernels/flash_attention.py"}
    base_files = {
        p.relative_to(a.baseline / "vllm_rlt").as_posix(): p
        for p in (a.baseline / "vllm_rlt").rglob("*.py")
    }
    new_files = {
        p.relative_to(ROOT / "vllm_rlt").as_posix(): p for p in (ROOT / "vllm_rlt").rglob("*.py")
    }
    assert set(base_files) - set(new_files) == set()
    assert set(new_files) - set(base_files) <= {"kernels/huginn_norm.py"}
    for name, path in base_files.items():
        if name not in allowed:
            assert path.read_bytes() == new_files[name].read_bytes(), name
    assert not subprocess.check_output(["git", "-C", str(a.baseline), "status", "--porcelain"]), (
        "Baseline checkout must be clean"
    )

    models = {}
    for arm, cls in (("baseline", old_model), ("candidate", HuginnForCausalLM)):
        models[arm] = cls.from_pretrained(MODEL, device="cuda", dtype=torch.bfloat16)
    # Prove both strict loaders loaded every parameter identically before measuring.
    with torch.inference_mode():
        first = dict(models["baseline"].named_parameters())
        second = dict(models["candidate"].named_parameters())
        assert first.keys() == second.keys()
        for name in first:
            assert torch.equal(first[name], second[name]), name
    original_make = benchmark.make_engine

    def make_engine(unused, arm):
        previous = engine_module.ModelRunner
        try:
            engine_module.ModelRunner = old_runner if arm == "baseline" else ModelRunner
            engine = original_make(models[arm], "core_coda")
        finally:
            engine_module.ModelRunner = previous
        assert type(engine.model_runner) is (old_runner if arm == "baseline" else ModelRunner)
        return engine

    benchmark.make_engine = make_engine
    benchmark.ARMS = ("baseline", "candidate")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    prompts = [tokenizer.encode(s, add_special_tokens=False) for s in benchmark.PROMPTS]
    header = {
        "base_head": BASE_HEAD,
        "head": HEAD,
        "source_dirty": bool(
            subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain"])
        ),
        "torch_version": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_uuid": str(torch.cuda.get_device_properties(0).uuid),
        "mode": a.mode,
        "start": a.start,
        "prompt_token_ids": prompts,
        "scope": (
            "both arms core+coda graphs; old/new production Huginn and ModelRunner classes; "
            "identical shared dependencies; separately loaded byte-equal parameters"
        ),
        "benchmark_function_source_unchanged": True,
    }
    write_json(a.output / "run-header.json", header)
    with torch.inference_mode():
        result = (
            benchmark.parity(None, prompts, a.output)
            if a.mode == "parity"
            else benchmark.timing(None, prompts, a.output, a.start)
        )
    write_json(a.output / "result.json", {**header, **result})


if __name__ == "__main__":
    main()
