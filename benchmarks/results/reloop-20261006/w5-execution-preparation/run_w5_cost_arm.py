"""One granted, preflighted W5 arm; reuse the frozen native cost observer."""

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import TypedDict, cast

from prepare_w5_workloads import Arm


class Scheduler(TypedDict):
    max_num_seqs: int
    max_num_batched_tokens: int
    mode: str
    min_coda_batch_size: int
    admission_scan_limit: int
    max_admission_bypasses: int
    prefill_chunk_size: int
    max_prefill_batches_before_decode: int
    policy: str
    enable_preemption: bool


class Case(TypedDict):
    case_id: str
    batch_limit_B: int
    outstanding_C: int
    prompt_ids: list[str]
    prompt_length_order: list[int]
    first_B_target_kv_bytes: int
    output_tokens: int
    ignore_eos: bool
    scheduler_config: Scheduler


class Protocol(TypedDict):
    status: str
    source_sha: str
    source_config_sha256: str
    model_revision: str
    trial_observer_sha256: str
    resolved_prompt_manifest_sha256: str
    common_pool_bytes: int
    history_planes: int
    block_size: int
    dtype: str
    attention_backend: str
    warmups: int
    measurements: int
    arms: list[Arm]
    cases: list[Case]


class Prompt(TypedDict):
    prompt_id: str
    case_id: str
    ordinal: int
    target_prompt_tokens: int
    prompt_token_ids: list[int]


class PromptFile(TypedDict):
    records: list[Prompt]


class Admission(TypedDict):
    status: str
    controller_pid: int
    protocol_sha256: str
    numerical_gate_sha256: str
    numerical_gate_gpu_uuid: str
    gpu_uuid: str
    resource_checks_verified: bool
    capacity_preflight_verified: bool
    model_files_verified: bool
    allowed_job_ids: list[str]
    archive_by_unix: float
    forecast_one_job_seconds: float
    offbox_reserve_seconds: float


class Gate(TypedDict):
    gpu_uuid: str
    status: str
    offbox_verified: bool
    source_sha: str
    model_revision: str
    dtype: str
    attention_backend: str


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "protocol",
        "prompts",
        "admission",
        "gate",
        "source",
        "model",
        "observer",
        "output",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument(
        "--arm", choices=("A_P4D4_off", "C_P4D3_off", "D_P4D3_on"), required=True
    )
    parser.add_argument("--fresh", type=int, choices=(0, 1), required=True)
    args = parser.parse_args()
    protocol = cast(Protocol, json.loads(args.protocol.read_text()))
    if protocol["status"] != "CANONICAL_PROMPTS_FROZEN_NOT_GRANTED_NOT_ADMITTED":
        raise ValueError(
            "cost requires actual canonical frozen prompts; templates are not admissible"
        )
    assert protocol["dtype"] == "float32" and protocol["attention_backend"] == "triton"
    assert protocol["history_planes"] == 4 and protocol["block_size"] == 16
    assert protocol["warmups"] == protocol["measurements"] == 5
    admission = cast(Admission, json.loads(args.admission.read_text()))
    gate = cast(Gate, json.loads(args.gate.read_text()))
    label = f"{args.case}-fresh{args.fresh}-{args.arm}"
    assert admission["status"] == "WHOLE_WINDOW_ADMITTED"
    assert admission["controller_pid"] == os.getppid()
    assert admission["protocol_sha256"] == digest(args.protocol)
    assert label in admission["allowed_job_ids"]
    assert (
        admission["resource_checks_verified"]
        and admission["capacity_preflight_verified"]
        and admission["model_files_verified"]
    )
    assert (
        time.time()
        + admission["forecast_one_job_seconds"]
        + admission["offbox_reserve_seconds"]
        < admission["archive_by_unix"]
    )
    assert digest(args.gate) == admission["numerical_gate_sha256"]
    assert gate["status"] == "PASS" and gate["offbox_verified"]
    assert (
        gate["source_sha"] == protocol["source_sha"]
        and gate["model_revision"] == protocol["model_revision"]
    )
    assert (
        gate["dtype"] == protocol["dtype"]
        and gate["attention_backend"] == protocol["attention_backend"]
    )
    assert (
        gate["gpu_uuid"]
        == admission["numerical_gate_gpu_uuid"]
        == admission["gpu_uuid"]
    )
    assert os.environ["CUDA_VISIBLE_DEVICES"] == admission["gpu_uuid"]
    # The owning whole-window controller verifies boot/locks/UUID/emptycompute,
    # the complete clean checkout and checkpoint bytes before admitting this job.
    assert (
        digest(args.source / "vllm_rlt/config.py") == protocol["source_config_sha256"]
    )
    assert digest(args.observer) == protocol["trial_observer_sha256"]
    assert digest(args.prompts) == protocol["resolved_prompt_manifest_sha256"]
    rows = cast(PromptFile, json.loads(args.prompts.read_text()))["records"]
    assert len(rows) == len({r["prompt_id"] for r in rows}) == 820
    case = next(c for c in protocol["cases"] if c["case_id"] == args.case)
    arm = next(a for a in protocol["arms"] if a["arm_id"] == args.arm)
    rows = [r for r in rows if r["case_id"] == args.case]
    assert len(rows) == 4 * case["outstanding_C"]
    assert [r["prompt_id"] for r in rows] == case["prompt_ids"]
    assert [len(r["prompt_token_ids"]) for r in rows] == case["prompt_length_order"]
    assert case["output_tokens"] == 128 and case["ignore_eos"]
    if (
        case["case_id"] != "Q64"
        and case["first_B_target_kv_bytes"] > protocol["common_pool_bytes"]
    ):
        raise ValueError(
            "PLANNED_CAPACITY_EXCEEDED: declared target residency does not fit"
        )
    assert (
        case["scheduler_config"]["max_num_seqs"] == case["batch_limit_B"]
        and case["scheduler_config"]["mode"] == "refill"
    )
    args.output.mkdir(exist_ok=False)
    sys.path[:0] = [str(args.source), str(args.observer.parent)]
    import fp32_candidate_cost_trial as observer
    import torch

    import vllm_rlt
    from vllm_rlt import (
        LLM,
        CacheConfig,
        ExecutionConfig,
        LoopCDParams,
        SamplingParams,
        SchedulerConfig,
    )

    assert Path(vllm_rlt.__file__).resolve().parents[1] == args.source.resolve()
    assert digest(Path(observer.__file__)) == protocol["trial_observer_sha256"]
    assert not torch.cuda.is_initialized()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    execution = ExecutionConfig(
        loopcd=arm["guidance"],
        prefill_depth=4,
        cuda_graphs=False,
        async_scheduling=False,
    )
    sampling = SamplingParams(
        max_tokens=128,
        temperature=0.0,
        min_loops=arm["depth"],
        max_loops=arm["depth"],
        ignore_eos=True,
        loopcd=LoopCDParams(
            reference_loop=1, strength=arm["strength"], implementation="two_head"
        )
        if arm["guidance"]
        else None,
    )
    cache_config = CacheConfig(
        kv_cache_memory_bytes=protocol["common_pool_bytes"],
        block_size=16,
        layout="last_exited",
        enable_prefix_caching=False,
    )
    scheduler = SchedulerConfig(**case["scheduler_config"])
    receipt = {
        "status": "RUNNING",
        "pid": os.getpid(),
        "job_id": label,
        "started_unix": time.time(),
        "protocol_sha256": digest(args.protocol),
        "admission_sha256": digest(args.admission),
        "gate_sha256": digest(args.gate),
        "observer_sha256": digest(args.observer),
        "driver_sha256": digest(Path(__file__)),
        "case": case,
        "arm": arm,
        "execution": asdict(execution),
        "sampling": asdict(sampling),
        "cache_config": asdict(cache_config),
        "scheduler_config": asdict(scheduler),
    }
    result = args.output / "case.json"
    result.write_text(json.dumps(receipt, indent=2) + "\n")
    llm: LLM | None = None
    try:
        llm = LLM(
            str(args.model),
            revision=protocol["model_revision"],
            device="cuda",
            dtype=torch.float32,
            attention_backend="triton",
            cache_config=cache_config,
            scheduler_config=scheduler,
            execution_config=execution,
        )
        engine = llm.engine
        parameter = next(engine.model_runner.model.parameters())
        assert parameter.dtype == torch.float32 and parameter.device.type == "cuda"
        cache = engine.cache_manager
        assert cache.storage_depths == 4 and cache.layout == "last_exited"
        largest = max((len(r["prompt_token_ids"]) + 127 + 15) // 16 * 4 for r in rows)
        assert largest <= cache.num_blocks
        receipt.update(
            capacity_status="ADMITTED_NATIVE_CACHE",
            actual_memory_plan=engine.memory_plan,
            kv_pool_bytes=cache.num_blocks * cache.bytes_per_block,
            actual_storage_planes=cache.storage_depths,
            GPU_free_total_bytes=list(torch.cuda.mem_get_info(parameter.device)),
        )
        result.write_text(json.dumps(receipt, indent=2) + "\n")
        observer.install_neutral_counters(engine)
        observer.run(
            engine,
            [r["prompt_token_ids"] for r in rows],
            sampling,
            case["outstanding_C"],
            args.output / "trials",
            warmups=5,
            repetitions=5,
        )
        receipt.update(status="COMPLETE", finished_unix=time.time())
    except torch.cuda.OutOfMemoryError as error:
        receipt.update(status="RUN_OOM", error=str(error), finished_unix=time.time())
        raise
    finally:
        if receipt["status"] == "RUNNING":
            receipt.update(status="INCOMPLETE", finished_unix=time.time())
        if llm is not None:
            llm.close()
        result.write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
