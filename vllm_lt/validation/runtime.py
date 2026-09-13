"""Shared scheduler verification and device accounting for validation tools."""

import getpass
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch


def memory():
    return {
        "allocated_bytes": torch.cuda.memory_allocated(),
        "reserved_bytes": torch.cuda.memory_reserved(),
    }


def environment():
    visibility = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not visibility or len(visibility.split(",")) != 1:
        raise ValueError("run requires exactly one caller-assigned CUDA_VISIBLE_DEVICES entry")
    if not shutil.which("gpu"):
        raise ValueError("device execution requires the verified gpu scheduler")
    rows = json.loads(subprocess.check_output(["gpu", "status", "--json"], text=True))
    scheduler = [row for row in rows if str(row["gpu_id"]) == visibility]
    if len(scheduler) != 1 or scheduler[0].get("user") != getpass.getuser():
        raise ValueError("visible GPU does not match this account's scheduler reservation")
    if scheduler[0].get("type") != "RUN":
        raise ValueError("device execution requires a gpu run reservation")
    if torch.cuda.device_count() != 1:
        raise ValueError("the benchmark supports one visible CUDA device")
    properties = torch.cuda.get_device_properties(0)
    return {
        "utc_started": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(),
        "account": getpass.getuser(),
        "pid": os.getpid(),
        "parent_pid": os.getppid(),
        "command": [sys.executable, *sys.argv],
        "python": sys.version,
        "torch_cuda_version": torch.version.cuda,
        "software": {
            name: importlib.metadata.version(name)
            for name in ("torch", "triton", "safetensors", "huggingface-hub")
        },
        "cuda_visible_devices": visibility,
        "logical_device": "cuda:0",
        "gpu_name": properties.name,
        "gpu_uuid": str(properties.uuid),
        "total_device_bytes": properties.total_memory,
        "compute_capability": [properties.major, properties.minor],
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "actual_torch_threads": {
            "intraop": torch.get_num_threads(),
            "interop": torch.get_num_interop_threads(),
        },
        "runtime_environment": {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "CUDA_MODULE_LOADING",
                "CUBLAS_WORKSPACE_CONFIG",
                "NVIDIA_TF32_OVERRIDE",
                "PYTORCH_ALLOC_CONF",
                "PYTORCH_CUDA_ALLOC_CONF",
            )
        },
        "numa_status": [
            line
            for line in Path("/proc/self/status").read_text().splitlines()
            if line.startswith(("Cpus_allowed_list:", "Mems_allowed_list:"))
        ],
        "scheduler": scheduler,
        "reservation_environment": {
            name: os.environ[name]
            for name in ("CANHAZGPU_TASK_ID", "CANHAZGPU_RUN_ID", "CANHAZGPU_RESERVATION_ID")
            if name in os.environ
        },
    }
