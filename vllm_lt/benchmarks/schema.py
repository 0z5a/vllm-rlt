"""CPU-only, versioned inputs and the deliberately fixed first benchmark plan."""

import hashlib
import json
import math
import subprocess
from pathlib import Path

from vllm_lt.models.config import OURO_MODEL_ID, OURO_REVISION, OuroConfig


def _keys(value, required, *, optional=(), name="object"):
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    missing, unknown = set(required) - value.keys(), value.keys() - set(required) - set(optional)
    if missing or unknown:
        raise ValueError(
            f"{name}: missing fields {sorted(missing)}, unknown fields {sorted(unknown)}"
        )


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")


def _reject_constant(value):
    raise ValueError(f"nonfinite JSON value: {value}")


def _finite_float(value):
    number = float(value)
    if not math.isfinite(number):
        _reject_constant(value)
    return number


def _object_pairs(pairs):
    result = {}
    for name, value in pairs:
        if name in result:
            raise ValueError(f"duplicate JSON field: {name}")
        result[name] = value
    return result


def read_json(path):
    return json.loads(
        Path(path).read_text(encoding="utf-8"),
        parse_constant=_reject_constant,
        parse_float=_finite_float,
        object_pairs_hook=_object_pairs,
    )


def write_json(path, value):
    """Write only finite UTF-8 JSON; directory ownership belongs to the caller."""
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _digest(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _file_record(path, *, relative_to=None):
    path = Path(path)
    checksum = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            checksum.update(chunk)
            size += len(chunk)
    return {
        "path": str(path.relative_to(relative_to) if relative_to else path.resolve()),
        "size_bytes": size,
        "sha256": checksum.hexdigest(),
    }


def _version(value, artifact_type):
    if type(value.get("schema_version")) is not int or value["schema_version"] != 1:
        raise ValueError("unsupported schema_version")
    if value.get("artifact_type") != artifact_type:
        raise ValueError(f"expected artifact_type={artifact_type}")


def _tokens(value, length, name):
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{name} must contain exactly {length} token IDs")
    for token in value:
        _integer(token, name)
        if token >= OuroConfig.vocab_size:
            raise ValueError(f"{name} token outside Ouro vocabulary")


def _validate_suite(suite):
    _keys(
        suite,
        (
            "schema_version",
            "artifact_type",
            "suite_id",
            "model_id",
            "model_revision",
            "tokenizer_revision",
            "provenance",
            "workloads",
            "workloads_sha256",
        ),
        name="suite",
    )
    _version(suite, "benchmark_suite")
    _text(suite["suite_id"], "suite_id")
    if (suite["model_id"], suite["model_revision"], suite["tokenizer_revision"]) != (
        OURO_MODEL_ID,
        OURO_REVISION,
        OURO_REVISION,
    ):
        raise ValueError("M1 requires the pinned Ouro model and tokenizer revision")
    provenance = suite["provenance"]
    _keys(
        provenance,
        (
            "kind",
            "text_sources",
            "prompt_construction",
            "output_construction",
            "depth_construction",
            "tokenizer",
        ),
        name="provenance",
    )
    if provenance["kind"] != "synthetic_cpu_tokenized":
        raise ValueError("M1 fixture provenance must be synthetic_cpu_tokenized")
    for name in ("prompt_construction", "output_construction", "depth_construction"):
        _text(provenance[name], name)
    if not isinstance(provenance["text_sources"], list) or not provenance["text_sources"]:
        raise ValueError("text_sources must be a nonempty list")
    for text in provenance["text_sources"]:
        _text(text, "text source")
    tokenizer = provenance["tokenizer"]
    _keys(
        tokenizer,
        ("filename", "sha256", "size_bytes", "library", "library_version", "add_special_tokens"),
        name="tokenizer provenance",
    )
    for name in ("filename", "sha256", "library", "library_version"):
        _text(tokenizer[name], name)
    _integer(tokenizer["size_bytes"], "tokenizer size_bytes", 1)
    if tokenizer["filename"] != "tokenizer.json" or tokenizer["add_special_tokens"] is not False:
        raise ValueError("fixtures require tokenizer.json without added special tokens")
    workloads = suite["workloads"]
    if (
        not isinstance(workloads, list)
        or any(not isinstance(w, dict) for w in workloads)
        or [w.get("workload_id") for w in workloads] != ["W1", "W2", "W3", "W4", "W5"]
    ):
        raise ValueError("M1 requires ordered workloads W1 through W5")
    dimensions = [[(128, 64)], [(128, 64)] * 8, [(512, 32)], [(64, 32), (128, 64)] * 4]
    dimensions.append(dimensions[-1])
    for number, (workload, expected) in enumerate(zip(workloads, dimensions), 1):
        replayed = number >= 4
        _keys(workload, ("workload_id", "kind", "requests"), optional=("replay",), name="workload")
        kind = "scheduler_replay" if replayed else "fixed_depth_generation"
        if workload["kind"] != kind or ("replay" in workload) != replayed:
            raise ValueError(f"W{number} has an invalid kind/replay combination")
        requests = workload["requests"]
        if not isinstance(requests, list) or len(requests) != len(expected):
            raise ValueError(f"W{number} has incorrect request count")
        ids = []
        for request, (prompt_length, output_length) in zip(requests, expected):
            _keys(
                request,
                ("request_id", "prompt_token_ids", "max_output_tokens", "arrival_offset_ns"),
                name="request",
            )
            _text(request["request_id"], "request_id")
            ids.append(request["request_id"])
            _tokens(request["prompt_token_ids"], prompt_length, "prompt_token_ids")
            _integer(request["max_output_tokens"], "max_output_tokens", 1)
            _integer(request["arrival_offset_ns"], "arrival_offset_ns")
            if request["max_output_tokens"] != output_length or request["arrival_offset_ns"] != 0:
                raise ValueError("request output length/arrival differs from the M1 workload")
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate request IDs")
        if replayed:
            _keys(workload["replay"], ids, name="replay request mapping")
            for request in requests:
                trace = workload["replay"][request["request_id"]]
                _keys(trace, ("output_token_ids", "exit_depths"), name="replay trace")
                count = request["max_output_tokens"]
                _tokens(trace["output_token_ids"], count, "replay output_token_ids")
                depths = trace["exit_depths"]
                if not isinstance(depths, list) or len(depths) != count:
                    raise ValueError("replay depths must match the exact output count")
                if any(type(depth) is not int or depth not in (2, 3, 4) for depth in depths):
                    raise ValueError("replay depths must be integers 2 through 4")
                if depths[0] != 4 or (number == 5 and any(depth != 4 for depth in depths)):
                    raise ValueError("first output and all W5 outputs require depth four")
    fourth, fifth = workloads[3:]
    if fourth["requests"] != fifth["requests"] or any(
        fourth["replay"][key]["output_token_ids"] != fifth["replay"][key]["output_token_ids"]
        for key in fourth["replay"]
    ):
        raise ValueError("W4 and W5 must share request and output-token histories")
    if {trace["exit_depths"][1] for trace in fourth["replay"].values()} != {2, 3, 4}:
        raise ValueError("W4 must have heterogeneous first decode depths 2/3/4")
    if suite["workloads_sha256"] != _digest(workloads):
        raise ValueError("workloads_sha256 mismatch")


def load_suite(path):
    suite = read_json(path)
    _validate_suite(suite)
    return suite


def _constants(actual, expected, name):
    _keys(actual, expected, name=name)
    for key, value in expected.items():
        found = actual[key]
        numeric = type(value) is float and type(found) in (int, float)
        if (not numeric and type(found) is not type(value)) or found != value:
            raise ValueError(f"{name}.{key} must equal {value!r}")


def _validate_contract(contract):
    _keys(
        contract,
        (
            "schema_version",
            "artifact_type",
            "contract_id",
            "hypothesis",
            "isolated_variable",
            "primary_metrics",
            "acceptance_criterion",
            "stop_conditions",
            "engine",
            "arithmetic",
            "controls",
            "limits",
            "observer_policy",
        ),
        name="contract",
    )
    _version(contract, "run_contract")
    for name in ("contract_id", "hypothesis", "isolated_variable", "acceptance_criterion"):
        _text(contract[name], name)
    for name in ("primary_metrics", "stop_conditions"):
        if not isinstance(contract[name], list) or not contract[name]:
            raise ValueError(f"{name} must be a nonempty list")
        for value in contract[name]:
            _text(value, name)
    engine = contract["engine"]
    _keys(engine, ("dtype", "attention_backend", "cache", "scheduler", "sampling"), name="engine")
    if engine["dtype"] != "float32" or engine["attention_backend"] != "triton":
        raise ValueError("M1 requires float32 and triton")
    _constants(engine["cache"], {"num_blocks": 1024, "block_size": 16}, "cache")
    _constants(
        engine["scheduler"],
        {"max_num_seqs": 8, "max_num_batched_tokens": 128, "min_coda_batch_size": 1},
        "scheduler",
    )
    _constants(
        engine["sampling"],
        {
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": -1,
            "seed": 0,
            "min_loops": 2,
            "max_loops": 4,
            "exit_threshold": 1.0,
            "ignore_eos": True,
        },
        "sampling",
    )
    _constants(
        contract["arithmetic"],
        {
            "allow_tf32": False,
            "allow_bf16_reduced_precision_reduction": False,
            "allow_fp16_reduced_precision_reduction": False,
        },
        "arithmetic",
    )
    _constants(
        contract["controls"],
        {
            "cpu_threads": 1,
            "interop_threads": 1,
            "seed": 0,
            "device_count": 1,
            "numa_policy": "inherit-and-record",
            "cache_policy": "fresh-engine-preserve-allocator",
            "residency_policy": "one-model-one-engine",
        },
        "controls",
    )
    _constants(
        contract["limits"],
        {"workload_timeout_s": 600, "total_timeout_s": 7200, "profile_decode_outputs": 16},
        "limits",
    )
    _constants(
        contract["observer_policy"],
        {
            "timing": "lightweight-host-events",
            "profile": "cpu-cuda-profiler",
            "finite_checks": "feasibility-only",
        },
        "observer_policy",
    )


def _source_manifest():
    root = Path(__file__).resolve().parents[2]

    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args])

    names = git("ls-files", "-z", "--cached", "--others", "--exclude-standard").decode().split("\0")
    return {
        "root": str(root),
        "commit": git("rev-parse", "HEAD").decode().strip(),
        "status": git("status", "--porcelain", "--untracked-files=all").decode(),
        "files": [
            _file_record(root / name, relative_to=root)
            for name in sorted(set(names))
            if name and (root / name).is_file()
        ],
    }


def _model_files(model_path):
    files = [
        path
        for path in sorted(model_path.iterdir())
        if path.is_file() and path.suffix in {".safetensors", ".json", ".txt", ".model"}
    ]
    names = {path.name for path in files}
    if not {"config.json", "tokenizer.json", "tokenizer_config.json"} <= names or not any(
        path.suffix == ".safetensors" for path in files
    ):
        raise ValueError("prepared model requires config, tokenizer files and safetensors weights")
    index = model_path / "model.safetensors.index.json"
    if index.exists():
        mapping = read_json(index).get("weight_map", {})
        if not mapping or not set(mapping.values()) <= names:
            raise ValueError("checkpoint index refers to missing weight shards")
    return [_file_record(path, relative_to=model_path) for path in files]


def _workload_stats(suite, config, contract):
    cache = contract["engine"]["cache"]
    page_bytes = (
        2
        * config.num_hidden_layers
        * cache["block_size"]
        * config.num_key_value_heads
        * config.head_dim
        * 4
    )
    stats = {}
    for workload in suite["workloads"]:
        prompts = outputs = steps = pages = decode_loops = 0
        for request in workload["requests"]:
            prompt, output = len(request["prompt_token_ids"]), request["max_output_tokens"]
            capacity = prompt + output - 1
            needed = 4 * math.ceil(capacity / cache["block_size"])
            if capacity > config.max_position_embeddings or needed > cache["num_blocks"]:
                raise ValueError("request exceeds context or KV capacity")
            prompts += prompt
            outputs += output
            pages += needed
            steps += prompt + 1 + (output - 1) * 6
            decode_loops += (
                sum(workload["replay"][request["request_id"]]["exit_depths"][1:])
                if "replay" in workload
                else (output - 1) * 4
            )
        if pages > cache["num_blocks"]:
            raise ValueError("workload cannot admit all intended requests into the fixed KV pool")
        stats[workload["workload_id"]] = {
            "prompt_tokens": prompts,
            "output_tokens": outputs,
            "reserved_pages": pages,
            "prefill_token_loops": prompts * 4,
            "decode_token_loops": decode_loops,
            "max_steps": steps,
            "max_events": 4 * steps + 2 * outputs + 8 * len(workload["requests"]) + 16,
            "pool_bytes": page_bytes * cache["num_blocks"],
            "bytes_per_block": page_bytes,
        }
    return stats


def _execution_order(suite, contract, stats):
    cells = [
        (w["workload_id"], mode)
        for w in suite["workloads"]
        for mode in (("refill", "no_refill") if "replay" in w else ("refill",))
    ]
    order = []
    workloads = {w["workload_id"]: w for w in suite["workloads"]}

    def add(workload_id, mode, phase, repetition, pair_id=None):
        cell_id = f"{workload_id}-{mode}"
        order.append(
            {
                "run_id": f"{phase}-{cell_id}-{repetition}",
                "cell_id": cell_id,
                "workload_id": workload_id,
                "mode": mode,
                "phase": phase,
                "repetition": repetition,
                "pair_id": pair_id,
                "instrumentation": {"profile": "profile", "feasibility": "validation"}.get(
                    phase, "timing"
                ),
                "max_steps": stats[workload_id]["max_steps"],
                "max_events": stats[workload_id]["max_events"],
                "workload_sha256": _digest(workloads[workload_id]),
                "controls_sha256": _digest(contract),
            }
        )

    for phase in ("feasibility", "warmup"):
        for workload_id, mode in cells:
            add(workload_id, mode, phase, 1)
    for workload_id in ("W1", "W2", "W3"):
        for repetition in (1, 2):
            add(workload_id, "refill", "measured", repetition)
    for workload_id in ("W4", "W5"):
        for mode, repetition in (("refill", 1), ("no_refill", 1), ("no_refill", 2), ("refill", 2)):
            add(workload_id, mode, "measured", repetition, f"{workload_id}-pair-{repetition}")
    for workload_id in ("W1", "W4"):
        add(workload_id, "refill", "profile", 1)
    return order


def make_plan(suite_path, contract_path, model_path):
    suite_path, contract_path, model_path = map(
        lambda p: Path(p).resolve(), (suite_path, contract_path, model_path)
    )
    suite, contract = load_suite(suite_path), read_json(contract_path)
    _validate_contract(contract)
    config = OuroConfig.from_dict(read_json(model_path / "config.json"))
    reference = OuroConfig()
    for name in (
        "vocab_size",
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "total_ut_steps",
    ):
        if getattr(config, name) != getattr(reference, name):
            raise ValueError(f"M1 model configuration differs from pinned Ouro: {name}")
    files = _model_files(model_path)
    tokenizer = next(record for record in files if record["path"] == "tokenizer.json")
    if any(
        tokenizer[name] != suite["provenance"]["tokenizer"][name]
        for name in ("sha256", "size_bytes")
    ):
        raise ValueError("prepared tokenizer does not match frozen fixture provenance")
    stats = _workload_stats(suite, config, contract)
    plan = {
        "schema_version": 1,
        "artifact_type": "execution_plan",
        "suite": suite,
        "contract": contract,
        "model_path": str(model_path),
        "model_config": config.to_dict(),
        "source": _source_manifest(),
        "model_files": files,
        "inputs": {"suite": _file_record(suite_path), "contract": _file_record(contract_path)},
        "workload_stats": stats,
        "execution_order": _execution_order(suite, contract, stats),
    }
    plan["plan_sha256"] = _digest(plan)
    return plan


def validate_plan_integrity(plan):
    """Check a saved plan without reading source, input files, weights, or a GPU."""
    _keys(
        plan,
        (
            "schema_version",
            "artifact_type",
            "suite",
            "contract",
            "model_path",
            "model_config",
            "source",
            "model_files",
            "inputs",
            "workload_stats",
            "execution_order",
            "plan_sha256",
        ),
        name="execution plan",
    )
    _version(plan, "execution_plan")
    if plan["plan_sha256"] != _digest({k: v for k, v in plan.items() if k != "plan_sha256"}):
        raise ValueError("execution plan content hash mismatch")


def verify_plan(plan):
    validate_plan_integrity(plan)
    current = make_plan(
        plan["inputs"]["suite"]["path"], plan["inputs"]["contract"]["path"], plan["model_path"]
    )
    if current != plan:
        changed = sorted(key for key in current if current[key] != plan.get(key))
        raise ValueError(f"frozen plan inputs/source/controls changed: {changed}")
