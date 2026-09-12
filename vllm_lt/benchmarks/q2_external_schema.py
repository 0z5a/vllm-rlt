"""BF16 single-pair protocol and readback of the frozen FP32 v1 experiment."""

import importlib
import os
import re
import sys
from copy import deepcopy
from pathlib import Path

from vllm_lt.models.config import OuroConfig
from vllm_lt.validation.schema import _file_records, _validate_dependencies, dependency_manifest

from .ab_schema import (
    RUNTIME_VARIABLES,
    affinity_snapshot,
    equal,
    require,
    validate_affinity,
    validate_model_config,
)
from .schema import (
    _digest,
    _file_record,
    _integer,
    _keys,
    _model_files,
    _source_manifest,
    _text,
    _validate_suite,
    read_json,
)

BASE_SHA = "630a8fdc0dd47b6da68a3d30db6b851fc08af5c5"
W1_SHA256 = "c0b63b23dc8334e1572ff3e1155a756520a4751c9304100fd1fc6a224c79c8c7"
INPUTS = {"suite": "benchmarks/fixtures/ouro-m1.json"}


def canonical_gpu_uuid(value):
    """Match NVIDIA's prefixed UUID and PyTorch's bare UUID without losing identity."""
    if not isinstance(value, str) or not re.fullmatch(
        r"(?:GPU-)?[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
        value,
    ):
        raise ValueError("malformed GPU UUID")
    return "GPU-" + value.removeprefix("GPU-")


IMPORT_MODULES = (
    "vllm_lt.benchmarks.q2_external_schema",
    "vllm_lt.benchmarks.q2_external",
    "vllm_lt.benchmarks.q2_external_driver",
    "vllm_lt.engine.llm_engine",
    "vllm_lt.worker.model_runner",
    "vllm_lt.models.ouro",
    "vllm_lt.validation.official",
    "vllm_lt.validation.official_cached",
)
LIMITS = {
    "total_timeout_s": 3600,
    "case_timeout_s": 600,
    "executions": 8,
    "workers": 1,
    "model_loads": 1,
    "feasibility_executions": 2,
    "warmup_executions": 2,
    "measured_executions": 4,
    "native_steps_max": 380,
    "official_calls_max": 64,
    "token_events_max": 64,
    "events_max": 128,
    "case_bytes_max": 2 * 1024**2,
    "artifact_bytes_max": 32 * 1024**2,
    "profile_executions": 0,
    "tensor_dump_bytes_max": 0,
}
ENGINE = {
    "dtype": "float32",
    "attention_backend": "triton",
    "cache": {"num_blocks": 1024, "block_size": 16},
    "scheduler": {
        "max_num_seqs": 8,
        "max_num_batched_tokens": 128,
        "min_coda_batch_size": 1,
        "mode": "refill",
    },
    "sampling": {
        "max_tokens": 64,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": -1,
        "seed": 0,
        "min_loops": 4,
        "max_loops": 4,
        "exit_threshold": 1.0,
        "ignore_eos": True,
    },
    "persistent_decode": False,
    "cuda_graphs": False,
}
ARITHMETIC = {
    "allow_tf32": False,
    "allow_bf16_reduced_precision_reduction": False,
    "allow_fp16_reduced_precision_reduction": False,
}
OFFICIAL = {
    "attention_implementation": "eager",
    "use_cache": True,
    "exit_at_step": 3,
    "logits_to_keep": 1,
    "use_weighted_exit": False,
    "cache_slots": 96,
    "final_cache_length": 191,
    "cache_policy": "distinct-depth-layer-published-append-with-versioned-mask-shim",
}
CONTROLS = {
    "cpu_threads": 1,
    "interop_threads": 1,
    "seed": 0,
    "numa_policy": "inherit-verified-parent-binding",
    "residency_policy": "one-native-model-shared-official-weights-native-pool-always-resident",
    "cache_policy": "reuse-native-engine-reset-request-preserve-allocator",
    "environment_policy": "one-prepared-environment-one-worker",
}
ACCEPTANCE = {
    "equivalence": "all-64-actual-greedy-ids-depth-four-before-warmup",
    "measured_outputs": "all-four-runs-match-feasibility",
    "timing": "two-complete-pairs-no-speed-minimum",
    "variation": "report-raw-values-and-ranges-no-winner-when-overlapping",
    "scope": "FP32-fixed-depth-W1-only-no-task-quality-or-adaptive-claim",
}
OBSERVER = {
    "timing": "common-host-token-delivery-no-stage-hooks-or-page-scans",
    "finite_checks": "feasibility-only",
    "cache_detail": "feasibility-or-outside-delivery-timing",
    "clock": "perf_counter_ns-after-actual-selection-readback",
}
STOP_CONDITIONS = [
    "total-or-complete-case-deadline-or-byte-budget",
    "native-step-or-official-call-bound",
    "nonfinite-feasibility-or-actual-output-equivalence-failure",
    "source-input-control-invariant-device-or-cleanup-failure",
    "no-retry-no-replacement-no-extra-profile-or-equivalence-execution",
]
PRODUCTION_PREFIXES = (
    "vllm_lt/core/",
    "vllm_lt/kernels/",
    "vllm_lt/engine/",
    "vllm_lt/models/",
    "vllm_lt/worker/",
    "vllm_lt/entrypoints/",
)
PRODUCTION_FILES = ("vllm_lt/config.py", "vllm_lt/request.py", "vllm_lt/sampling_params.py")
# Digest of all 22 production file records at accepted PR14, not the new harness commit.
BASE_PRODUCTION_SHA256 = "58b93e0c79f8c402d7ab2910bc7aeb75634b21ccc0cafe200e44233a27d91e3f"
BF16_BASE_SHA = "5aa3bf72044ed5de9ca8bc265ac74ef0b15b2c46"
BF16_PRODUCTION_SHA256 = "c7bc0b74fd15d0437e51c2f0e93381c82419ce28aada6f8353b306698f42d10d"
BF16_ENGINE = {
    **ENGINE,
    "dtype": "bfloat16",
    "scheduler": {**ENGINE["scheduler"], "max_num_seqs": 1},
}
BF16_LIMITS = {**LIMITS, "executions": 4, "warmup_executions": 0, "measured_executions": 2}
BF16_ACCEPTANCE = {
    "equivalence": "record-cross-backend-token-agreement-separately",
    "measured_outputs": "each-backend-matches-own-feasibility",
    "timing": "one-complete-pair-no-speed-minimum",
    "variation": "single-observation-no-repeatability-claim",
    "scope": "BF16-fixed-depth-W1-latency-spot-check-no-accuracy-or-adaptive-claim",
}
BF16_CONTROLS = {
    **CONTROLS,
    "cpu_control_limits": {"background_cores_max": 0.1, "runnable_delay_fraction_max": 0.05},
}


def _production(source):
    return [
        r
        for r in source["files"]
        if r["path"].startswith(PRODUCTION_PREFIXES) or r["path"] in PRODUCTION_FILES
    ]


def validate_contract(contract, *, resolved=False):
    _keys(
        contract,
        (
            "schema_version",
            "artifact_type",
            "contract_id",
            "hypothesis",
            "isolated_variable",
            "base_sha",
            "production_files",
            "engine",
            "official",
            "arithmetic",
            "controls",
            "limits",
            "acceptance",
            "observer_policy",
            "stop_conditions",
        ),
        name="Q2 external contract",
    )
    require(
        type(contract["schema_version"]) is int and contract["schema_version"] in (1, 2),
        "unsupported contract version",
    )
    equal(contract["artifact_type"], "q2_external_contract", "contract type")
    bf16 = contract["schema_version"] == 2
    for key in ("contract_id", "hypothesis", "isolated_variable"):
        _text(contract[key], key)
    equal(contract["base_sha"], BF16_BASE_SHA if bf16 else BASE_SHA, "PR14 base")
    _file_records(contract["production_files"], "PR14 production files")
    equal(
        _digest(contract["production_files"]),
        BF16_PRODUCTION_SHA256 if bf16 else BASE_PRODUCTION_SHA256,
        "PR14 production bytes",
    )
    for key, value in (
        ("engine", BF16_ENGINE if bf16 else ENGINE),
        ("official", OFFICIAL),
        ("arithmetic", ARITHMETIC),
        ("limits", BF16_LIMITS if bf16 else LIMITS),
        ("acceptance", BF16_ACCEPTANCE if bf16 else ACCEPTANCE),
        ("observer_policy", OBSERVER),
        ("stop_conditions", STOP_CONDITIONS),
    ):
        equal(contract[key], value, key)
    controls = contract["controls"]
    fixed_controls = BF16_CONTROLS if bf16 else CONTROLS
    _keys(controls, (*fixed_controls, "gpu_ids", "gpu_uuid", "affinity"), name="controls")
    equal({k: controls[k] for k in fixed_controls}, fixed_controls, "fixed controls")
    if resolved or controls["gpu_ids"] is not None:
        require(
            isinstance(controls["gpu_ids"], list) and len(controls["gpu_ids"]) == 1,
            "one exact physical GPU ID required",
        )
        _integer(controls["gpu_ids"][0], "physical GPU ID")
    if resolved or controls["gpu_uuid"] is not None:
        require(
            isinstance(controls["gpu_uuid"], str)
            and re.fullmatch(
                r"GPU-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
                controls["gpu_uuid"],
            )
            is not None,
            "exact physical GPU UUID required",
        )
    if resolved or controls["affinity"] is not None:
        validate_affinity(controls["affinity"])


def source_probe(dtype="float32"):
    source = _source_manifest()
    root = Path(source["root"]).resolve()
    imports = {}
    for name in IMPORT_MODULES:
        path = Path(importlib.import_module(name).__file__).resolve()
        require(path.is_relative_to(root), f"import outside selected source: {name}")
        imports[name] = str(path.relative_to(root))
    from vllm_lt.validation.official_cached import cached_official_provenance

    return {
        "source": source,
        "imports": imports,
        "dependencies": dependency_manifest(),
        "cached_official": cached_official_provenance(dtype=dtype),
    }


def execution_order(version=1):
    definitions = (
        ("N-feas", "native", "feasibility", 0, None),
        ("O-feas", "official", "feasibility", 0, None),
        ("N-warm", "native", "warmup", 0, None),
        ("O-warm", "official", "warmup", 0, None),
        ("N1", "native", "measured", 1, "pair-1"),
        ("O1", "official", "measured", 1, "pair-1"),
        ("O2", "official", "measured", 2, "pair-2"),
        ("N2", "native", "measured", 2, "pair-2"),
    )
    return [
        {
            "run_id": name,
            "implementation_id": impl,
            "phase": phase,
            "repetition": rep,
            "pair_id": pair,
            "max_steps": 380 if impl == "native" else 64,
        }
        for name, impl, phase, rep, pair in definitions
        if version == 1 or name in ("N-feas", "O-feas", "N1", "O1")
    ]


def _resources(config, version=1):
    element_bytes = 2 if version == 2 else 4
    page = (
        2
        * config["num_hidden_layers"]
        * 16
        * config["num_key_value_heads"]
        * config["head_dim"]
        * element_bytes
    )
    return {
        "native_page_bytes": page,
        "native_pool_bytes": page * 1024,
        "native_required_pages": 4 * ((128 + 63 + 15) // 16),
        "official_final_cache_bytes": 2
        * 96
        * config["num_key_value_heads"]
        * 191
        * config["head_dim"]
        * element_bytes,
        "case_artifacts_bytes_upper_bound": len(execution_order(version))
        * LIMITS["case_bytes_max"],
        "auxiliary_bytes_max": 16 * 1024**2,
        "artifact_bytes_upper_bound": LIMITS["artifact_bytes_max"],
        "weight_storage": "shared-once; actual unique bytes established before inference",
        "tensor_payload_bytes": 0,
    }


def _controls_hash(plan):
    return _digest(
        {
            key: plan[key]
            for key in (
                "contract",
                "workload",
                "model_files",
                "source",
                "imports",
                "dependencies",
                "cached_official",
                "interpreter",
                "runtime_environment",
            )
        }
    )


def make_plan(contract_path, model_path, *, gpu_ids, gpu_uuid, affinity=None):
    contract_path, model_path = Path(contract_path).resolve(), Path(model_path).resolve()
    original = read_json(contract_path)
    validate_contract(original)
    contract = deepcopy(original)
    contract["controls"].update(
        gpu_ids=gpu_ids,
        gpu_uuid=gpu_uuid,
        affinity=affinity if affinity is not None else affinity_snapshot(),
    )
    probe = source_probe(contract["engine"]["dtype"])
    root = Path(probe["source"]["root"])
    suite_path = root / INPUTS["suite"]
    suite = read_json(suite_path)
    config = OuroConfig.from_dict(read_json(model_path / "config.json")).to_dict()
    plan = {
        "schema_version": contract["schema_version"],
        "artifact_type": "q2_external_plan",
        "contract": contract,
        "inputs": {
            "contract": {"file": _file_record(contract_path), "contents": original},
            "suite": {"file": _file_record(suite_path), "contents": suite},
        },
        "workload": deepcopy(suite["workloads"][0]),
        "model_path": str(model_path),
        "model_config": config,
        "model_files": _model_files(model_path),
        **probe,
        "interpreter": sys.executable,
        "runtime_environment": {k: os.environ.get(k) for k in RUNTIME_VARIABLES},
        "execution_order": execution_order(contract["schema_version"]),
        "resource_estimates": _resources(config, contract["schema_version"]),
    }
    plan["controls_sha256"] = _controls_hash(plan)
    plan["plan_sha256"] = _digest(plan)
    validate_plan(plan)
    return plan


def validate_plan(plan):
    _keys(
        plan,
        (
            "schema_version",
            "artifact_type",
            "contract",
            "inputs",
            "workload",
            "model_path",
            "model_config",
            "model_files",
            "source",
            "imports",
            "dependencies",
            "cached_official",
            "interpreter",
            "runtime_environment",
            "execution_order",
            "resource_estimates",
            "controls_sha256",
            "plan_sha256",
        ),
        name="Q2 external plan",
    )
    equal(
        [plan["schema_version"], plan["artifact_type"]],
        [plan["contract"]["schema_version"], "q2_external_plan"],
        "plan version",
    )
    equal(
        plan["plan_sha256"],
        _digest({k: v for k, v in plan.items() if k != "plan_sha256"}),
        "plan content hash",
    )
    validate_contract(plan["contract"], resolved=True)
    _keys(plan["inputs"], ("contract", "suite"), name="inputs")
    for item in plan["inputs"].values():
        _keys(item, ("file", "contents"), name="input")
        _file_records([item["file"]], "input file")
        require(Path(item["file"]["path"]).is_absolute(), "absolute input path required")
    original = deepcopy(plan["inputs"]["contract"]["contents"])
    validate_contract(original)
    original["controls"].update(
        {k: plan["contract"]["controls"][k] for k in ("gpu_ids", "gpu_uuid", "affinity")}
    )
    equal(original, plan["contract"], "resolved input contract")
    suite = plan["inputs"]["suite"]["contents"]
    _validate_suite(suite)
    equal(plan["workload"], suite["workloads"][0], "embedded W1")
    equal(_digest(plan["workload"]), W1_SHA256, "unchanged M1 W1 actual-generation fixture")
    _keys(plan["model_config"], OuroConfig().to_dict(), name="model configuration")
    validate_model_config(plan["model_config"])
    equal(plan["model_config"]["eos_token_id"], 0, "pinned EOS ID")
    _file_records(plan["model_files"], "model files")
    files = {r["path"]: r for r in plan["model_files"]}
    require(
        {"config.json", "tokenizer.json", "tokenizer_config.json"} <= files.keys()
        and any(p.endswith(".safetensors") for p in files),
        "prepared checkpoint files required",
    )
    tokenizer = suite["provenance"]["tokenizer"]
    for key in ("sha256", "size_bytes"):
        equal(files["tokenizer.json"][key], tokenizer[key], "prepared tokenizer " + key)
    source = plan["source"]
    _keys(source, ("root", "commit", "status", "files"), name="source")
    require(
        isinstance(source["commit"], str)
        and len(source["commit"]) == 40
        and all(c in "0123456789abcdef" for c in source["commit"]),
        "source commit hash",
    )
    equal(source["status"], "", "clean frozen source")
    _file_records(source["files"], "source files")
    paths = [r["path"] for r in source["files"]]
    equal(paths, sorted(paths), "source file order")
    for p in paths:
        require(not Path(p).is_absolute() and ".." not in Path(p).parts, "safe source path")
    equal(_production(source), plan["contract"]["production_files"], "unchanged PR14 production")
    source_files = {r["path"]: r for r in source["files"]}
    for name, item in plan["inputs"].items():
        path = Path(item["file"]["path"])
        if path.is_relative_to(source["root"]):
            relative = str(path.relative_to(source["root"]))
            equal(
                source_files.get(relative),
                {**item["file"], "path": relative},
                "source-bound input " + name,
            )
    equal(plan["imports"], {k: k.replace(".", "/") + ".py" for k in IMPORT_MODULES}, "import paths")
    require(set(plan["imports"].values()) <= set(paths), "imports require frozen source files")
    _validate_dependencies(plan["dependencies"])
    official = plan["dependencies"]["official"]
    equal(official["dependencies"]["transformers"], "4.55.0", "official dependency")
    equal(official["optional_kernels_present"], False, "optional kernels absence")
    shim = next(r for r in source["files"] if r["path"] == "vllm_lt/validation/official_cached.py")
    expected_cached = {
        **official,
        "use_cache": True,
        "dtype": "torch." + plan["contract"]["engine"]["dtype"],
        "logits_to_keep": 1,
        "use_weighted_exit": False,
        "cache_slots": 96,
        "cache_class": "CompatibleUniversalTransformerCache",
        "cache_shim_sha256": shim["sha256"],
        "cache_update": "unchanged published append at depth * layers + layer",
    }
    equal(plan["cached_official"], expected_cached, "cached official source and semantics")
    for path in (source["root"], plan["model_path"], plan["interpreter"]):
        require(
            isinstance(path, str) and Path(path).is_absolute(), "absolute source/model/interpreter"
        )
    _keys(plan["runtime_environment"], RUNTIME_VARIABLES, name="runtime environment")
    for value in plan["runtime_environment"].values():
        require(value is None or isinstance(value, str), "environment values must be strings/null")
    equal(
        plan["execution_order"], execution_order(plan["schema_version"]), "declared execution order"
    )
    equal(
        plan["resource_estimates"],
        _resources(plan["model_config"], plan["schema_version"]),
        "derived resource budgets",
    )
    equal(plan["controls_sha256"], _controls_hash(plan), "controls digest")


def verify_plan(plan):
    """Reject actual source/input/model/environment drift before any device use."""
    validate_plan(plan)
    actual = source_probe(plan["contract"]["engine"]["dtype"])
    for key in ("source", "imports", "dependencies", "cached_official"):
        equal(actual[key], plan[key], "actual " + key)
    equal(_model_files(Path(plan["model_path"])), plan["model_files"], "checkpoint files")
    equal(
        OuroConfig.from_dict(read_json(Path(plan["model_path"]) / "config.json")).to_dict(),
        plan["model_config"],
        "checkpoint configuration",
    )
    for name, item in plan["inputs"].items():
        path = Path(item["file"]["path"])
        equal(_file_record(path), item["file"], "input file " + name)
        equal(read_json(path), item["contents"], "embedded input " + name)
    equal(sys.executable, plan["interpreter"], "prepared interpreter")
    equal(
        {k: os.environ.get(k) for k in RUNTIME_VARIABLES},
        plan["runtime_environment"],
        "runtime environment",
    )
    equal(affinity_snapshot(), plan["contract"]["controls"]["affinity"], "CPU/NUMA affinity")
