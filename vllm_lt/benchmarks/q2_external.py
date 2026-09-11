"""Finite cached-native/official FP32 comparison and CPU-only preparation CLI."""

import argparse
import hashlib
import json
import os
import signal
import subprocess
import time
import traceback
from pathlib import Path

from vllm_lt.benchmarks.ab_schema import RUNTIME_VARIABLES, affinity_snapshot
from vllm_lt.benchmarks.q2_external_driver import (
    check_deadline,
    execute_case,
    pool_descriptor,
    require_empty,
    shared_weight_proof,
)
from vllm_lt.benchmarks.q2_external_schema import (
    LIMITS,
    canonical_gpu_uuid,
    make_plan,
    source_probe,
    validate_plan,
    verify_plan,
)
from vllm_lt.benchmarks.runner import configure_process, environment, load_model, release_device
from vllm_lt.benchmarks.schema import read_json, write_json


def artifact_usage(root):
    """Bound ordinary task-owned outputs, including partial records and logs."""
    root = Path(root)
    total = 0
    case_bytes = {}
    for path in root.rglob("*"):
        if path.is_symlink() or not (path.is_file() or path.is_dir()):
            raise ValueError("Q2 artifacts must be ordinary files/directories")
        if not path.is_file():
            continue
        size = path.stat().st_size
        total += size
        relative = path.relative_to(root).parts
        if len(relative) >= 3 and relative[0] == "runs":
            case_bytes[relative[1]] = case_bytes.get(relative[1], 0) + size
    if total > LIMITS["artifact_bytes_max"] or any(
        size > LIMITS["case_bytes_max"] for size in case_bytes.values()
    ):
        raise RuntimeError("Q2 artifact byte budget exceeded")
    return {"total_bytes": total, "case_bytes": case_bytes}


def file_record(path):
    data = Path(path).read_bytes()
    return {"size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def failure(error):
    return {"type": type(error).__name__, "message": str(error)[:2000]}


def check_environment(plan, observed):
    controls = plan["contract"]["controls"]
    expected = str(controls["gpu_ids"][0])
    if observed["cuda_visible_devices"] != expected:
        raise ValueError("reserved GPU ID differs from the frozen Q2 plan")
    if canonical_gpu_uuid(observed["gpu_uuid"]) != controls["gpu_uuid"]:
        raise ValueError("reserved GPU UUID differs from frozen Q2 plan")
    if observed["actual_torch_threads"] != {
        "intraop": controls["cpu_threads"],
        "interop": controls["interop_threads"],
    }:
        raise ValueError("Q2 thread controls differ")
    if observed["cpu_affinity"] != controls["affinity"]["cpu_ids"]:
        raise ValueError("Q2 CPU affinity differs")


def run_worker(plan, output_dir, deadline_ns):
    """One weight load, one common resident native pool, and exactly eight rows."""
    import torch

    from vllm_lt.config import CacheConfig, SchedulerConfig
    from vllm_lt.engine.llm_engine import LLMEngine
    from vllm_lt.validation.official_cached import OfficialOuroCachedReference

    root = Path(output_dir)
    worker = {
        "schema_version": 1,
        "artifact_type": "q2_external_worker",
        "plan_sha256": plan["plan_sha256"],
        "started_ns": time.perf_counter_ns(),
        "deadline_ns": deadline_ns,
        "status": "running",
        "completed_runs": [],
        "failures": [],
    }
    write_json(root / "worker.json", worker)
    model = engine = official = weights = None
    active_result = None
    device_started = False
    try:
        check_deadline(deadline_ns)
        verify_plan(plan)
        worker["source_probe"] = source_probe()
        configure_process(plan["contract"])
        device_started = True
        worker["environment"] = environment()
        worker["environment"]["runtime_environment"] = {
            name: os.environ.get(name) for name in RUNTIME_VARIABLES
        }
        worker["environment"]["arithmetic"] = {
            "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "allow_bf16_reduced_precision_reduction": (
                torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
            ),
            "allow_fp16_reduced_precision_reduction": (
                torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
            ),
        }
        worker["environment"]["cudnn_allow_tf32"] = torch.backends.cudnn.allow_tf32
        worker["environment"]["active_affinity"] = affinity_snapshot()
        if (
            worker["environment"]["arithmetic"] != plan["contract"]["arithmetic"]
            or worker["environment"]["cudnn_allow_tf32"]
        ):
            raise ValueError("Q2 arithmetic flags differ after configuration")
        check_environment(plan, worker["environment"])
        check_deadline(deadline_ns)
        preparation = {"started_ns": time.perf_counter_ns()}
        worker["preparation"] = preparation
        write_json(root / "worker.json", worker)
        model = load_model(
            {
                **plan,
                "workload_stats": {
                    "W1": {"pool_bytes": plan["resource_estimates"]["native_pool_bytes"]}
                },
            },
            preparation,
        )
        settings = plan["contract"]["engine"]
        engine = LLMEngine(
            model,
            cache_config=CacheConfig(**settings["cache"]),
            scheduler_config=SchedulerConfig(**settings["scheduler"]),
            attention_backend="triton",
        )
        weights = model.state_dict()
        official = OfficialOuroCachedReference(plan["model_config"], weights)
        preparation["shared_weights"] = shared_weight_proof(model, official)
        preparation["native_pool"] = pool_descriptor(engine)
        torch.cuda.synchronize()
        require_empty(engine, official)
        preparation["ended_ns"] = time.perf_counter_ns()
        check_deadline(deadline_ns)
        write_json(root / "worker.json", worker)
        expected_ids = None
        feasibility_ids = {}
        for row in plan["execution_order"]:
            check_deadline(deadline_ns)
            run_dir = root / "runs" / row["run_id"]
            run_dir.mkdir(parents=True, exist_ok=False)
            started = time.perf_counter_ns()
            case_deadline = min(deadline_ns, started + LIMITS["case_timeout_s"] * 10**9)
            marker = {
                "schema_version": 1,
                "run": row,
                "plan_sha256": plan["plan_sha256"],
                "started_ns": started,
                "deadline_ns": case_deadline,
            }
            write_json(root / "active-case.json", marker)
            write_json(run_dir / "started.json", marker)
            active_result = None
            try:
                active_result = execute_case(
                    plan, row, engine, official, started_ns=started, deadline_ns=case_deadline
                )
                ids = active_result["requests"][0]["token_ids"]
                if row["phase"] == "feasibility":
                    feasibility_ids[row["implementation_id"]] = ids
                    if row["implementation_id"] == "official":
                        if feasibility_ids.get("native") != ids:
                            raise ValueError(
                                "native/official feasibility histories differ; timing blocked"
                            )
                        expected_ids = list(ids)
                        worker["equivalence"] = {
                            "passed": True,
                            "token_ids": expected_ids,
                            "exit_depths": [4] * len(ids),
                            "checked_ns": time.perf_counter_ns(),
                        }
                elif expected_ids is None or ids != expected_ids:
                    raise ValueError("Q2 warmup/measured greedy history differs from feasibility")
                write_json(run_dir / "result.json", active_result)
                artifact_usage(root)
                check_deadline(case_deadline)
                completion = {
                    **marker,
                    "status": "complete",
                    "result": file_record(run_dir / "result.json"),
                    "completed_ns": time.perf_counter_ns(),
                }
                write_json(run_dir / "completed.json", completion)
                worker["completed_runs"].append(row["run_id"])
                # Acknowledgment is part of the bounded case, before later work.
                write_json(root / "worker.json", worker)
                artifact_usage(root)
                check_deadline(case_deadline)
                acknowledgment = {
                    **completion,
                    "completion": file_record(run_dir / "completed.json"),
                    "acknowledged_ns": time.perf_counter_ns(),
                }
                write_json(run_dir / "acknowledged.json", acknowledgment)
                write_json(root / "active-case.json", acknowledgment)
                artifact_usage(root)
                check_deadline(case_deadline)
            except BaseException as error:
                partial = getattr(error, "q2_partial_result", active_result)
                if (run_dir / "completed.json").exists():
                    # Completion/ACK records already bind these result bytes.
                    # Preserve them and record the later lifetime failure apart.
                    write_json(
                        run_dir / "failure.json",
                        {
                            "run": row,
                            "plan_sha256": plan["plan_sha256"],
                            "failure": failure(error),
                            "occurred_ns": time.perf_counter_ns(),
                        },
                    )
                elif partial is not None:
                    partial["status"] = "failed"
                    partial["failures"] = [*partial.get("failures", []), failure(error)]
                    write_json(run_dir / "result.json", partial)
                raise
        worker["status"] = "complete"
    except BaseException as error:
        worker["status"] = "failed"
        worker["failures"].append(failure(error))
        traceback.print_exc()
    finally:
        # Settle the task's submitted work before releasing cache/weight owners.
        cleanup_errors = []
        if model is not None:
            try:
                torch.cuda.synchronize()
                if engine is not None:
                    for request_id in list(engine.scheduler.requests):
                        engine.abort_request(request_id)
                    engine.last_schedule = None
                if official is not None:
                    official.close(completion_confirmed=True)
            except BaseException as error:
                cleanup_errors.append(failure(error))
        active_result = weights = official = engine = model = None
        if device_started:
            try:
                release_device(worker)
            except BaseException as error:
                cleanup_errors.append(failure(error))
        else:
            worker["device_initialization"] = "not_started"
            worker["teardown_after_workspace_release"] = {"allocated_bytes": 0, "reserved_bytes": 0}
        if cleanup_errors:
            worker["status"] = "failed"
            worker["failures"].extend(cleanup_errors)
        worker["ended_ns"] = time.perf_counter_ns()
        if worker["ended_ns"] >= deadline_ns:
            worker["status"] = "failed"
            worker["failures"].append(
                {"type": "TimeoutError", "message": "global deadline includes final cleanup"}
            )
        write_json(root / "worker.json", worker)
    return worker


def _stop_child(child):
    if child.poll() is None:
        os.killpg(child.pid, signal.SIGTERM)
        try:
            child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait(timeout=5)


def run_plan(plan, output_dir):
    """Parent watchdog owns this one child process group and partial ACK ledger."""
    validate_plan(plan)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / "plan.json", plan)
    started = time.perf_counter_ns()
    deadline = started + LIMITS["total_timeout_s"] * 10**9
    manifest = {
        "schema_version": 1,
        "artifact_type": "q2_external_manifest",
        "plan_sha256": plan["plan_sha256"],
        "status": "running",
        "started_ns": started,
        "deadline_ns": deadline,
        "completed_runs": [],
        "failures": [],
    }
    write_json(root / "manifest.json", manifest)
    child = None
    old_handlers = {}

    def interrupted(signum, frame):
        raise RuntimeError(f"Q2 controller interrupted by signal {signum}")

    def retain_prefix():
        path = root / "worker.json"
        if not path.exists():
            return None
        worker = read_json(path)
        prefix = worker["completed_runs"]
        expected = [r["run_id"] for r in plan["execution_order"]]
        if (
            prefix != expected[: len(prefix)]
            or prefix[: len(manifest["completed_runs"])] != manifest["completed_runs"]
        ):
            raise ValueError("Q2 worker acknowledgment ledger is not a monotonic plan prefix")
        manifest["completed_runs"] = list(prefix)
        write_json(root / "manifest.json", manifest)
        return worker

    try:
        verify_plan(plan)
        check_deadline(deadline)
        for signum in (signal.SIGINT, signal.SIGTERM):
            old_handlers[signum] = signal.signal(signum, interrupted)
        command = [
            plan["interpreter"],
            "-m",
            "vllm_lt.benchmarks.q2_external",
            "worker",
            "--plan",
            str(root / "plan.json"),
            "--output",
            str(root),
            "--deadline-ns",
            str(deadline),
        ]
        manifest["launch"] = {
            "command": command,
            "started_ns": time.perf_counter_ns(),
            "ended_ns": None,
            "returncode": None,
        }
        write_json(root / "manifest.json", manifest)
        with (root / "worker.log").open("w") as log:
            child = subprocess.Popen(
                command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
            manifest["launch"]["pid"] = child.pid
            write_json(root / "manifest.json", manifest)
            while child.poll() is None:
                check_deadline(deadline)
                active_path = root / "active-case.json"
                if active_path.exists():
                    active = read_json(active_path)
                    if "acknowledged_ns" not in active:
                        check_deadline(active["deadline_ns"])
                artifact_usage(root)
                retain_prefix()
                time.sleep(0.2)
        worker = retain_prefix()
        manifest["launch"].update(ended_ns=time.perf_counter_ns(), returncode=child.returncode)
        if (
            child.returncode != 0
            or worker is None
            or worker["status"] != "complete"
            or len(manifest["completed_runs"]) != 8
        ):
            raise RuntimeError("Q2 worker did not complete the frozen eight-row protocol")
        if any(worker["teardown_after_workspace_release"].values()):
            raise RuntimeError("Q2 task-owned CUDA memory remains")
        check_deadline(deadline)
        manifest["status"] = "complete"
    except BaseException as error:
        manifest["status"] = "failed"
        manifest["failures"].append(failure(error))
        traceback.print_exc()
    finally:
        if child is not None:
            _stop_child(child)
            manifest["launch"].update(ended_ns=time.perf_counter_ns(), returncode=child.returncode)
        try:
            retain_prefix()
        except BaseException as error:
            manifest["failures"].append(failure(error))
            manifest["status"] = "failed"
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)
        manifest["ended_ns"] = time.perf_counter_ns()
        write_json(root / "manifest.json", manifest)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    probe = sub.add_parser("probe")
    probe.add_argument("--contract", required=True)
    probe.add_argument("--model", required=True)
    probe.add_argument("--gpu-ids", type=int, nargs=1, required=True)
    probe.add_argument("--gpu-uuid", required=True)
    probe.add_argument("--output", type=Path, required=True)
    for name in ("run", "worker"):
        child = sub.add_parser(name)
        child.add_argument("--plan", type=Path, required=True)
        child.add_argument("--output", type=Path, required=True)
        if name == "worker":
            child.add_argument("--deadline-ns", type=int, required=True)
    report = sub.add_parser("report")
    report.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "probe":
        plan = make_plan(args.contract, args.model, gpu_ids=args.gpu_ids, gpu_uuid=args.gpu_uuid)
        args.output.mkdir(parents=True, exist_ok=False)
        write_json(args.output / "plan.json", plan)
        print(
            json.dumps({"plan_sha256": plan["plan_sha256"], "executions": 8, "device_work": False})
        )
        return 0
    if args.command == "report":
        from vllm_lt.benchmarks.q2_external_report import write_report

        result = write_report(args.run_dir)
        print(
            json.dumps(
                {"evidence_status": result["evidence_status"], "decision": result["decision"]}
            )
        )
        return 0 if result["decision"] == "passed" else 1
    plan = read_json(args.plan)
    result = (
        run_plan(plan, args.output)
        if args.command == "run"
        else run_worker(plan, args.output, args.deadline_ns)
    )
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
