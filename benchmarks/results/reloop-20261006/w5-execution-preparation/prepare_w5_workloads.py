"""Prepare all original W5 traces; this CPU step grants no model admission."""

import argparse
import ast
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict, cast


class Arm(TypedDict):
    arm_id: str
    depth: int
    guidance: bool
    strength: float


class Selection(TypedDict):
    status: str
    candidate: str


class Plan(TypedDict):
    source_sha: str
    model_repo: str
    model_revision: str
    dtype: str
    attention_backend: str
    prefill_depth: int
    selection: Selection
    arms: list[Arm]


class ModelConfig(TypedDict):
    num_hidden_layers: int
    num_key_value_heads: int
    num_attention_heads: int
    hidden_size: int
    max_position_embeddings: int


@dataclass(frozen=True)
class Case:
    case_id: str
    batch: int
    concurrency: int
    lengths: tuple[int, ...]


CASES = (
    Case("L1", 1, 1, (512,)),
    Case("T32", 32, 32, (512,)),
    Case("Q64", 32, 64, (512, 2048)),
    Case("T64_short", 64, 64, (256,)),
    Case("C8_long", 8, 8, (2048,)),
    Case("C4_longer", 4, 4, (8192,)),
    Case("A32_large", 32, 32, (2048,)),
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scheduler_defaults(path: Path) -> dict[str, int | str | bool]:
    definition = next(
        node
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.ClassDef) and node.name == "SchedulerConfig"
    )
    defaults: dict[str, int | str | bool] = {}
    for node in definition.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            assert node.value is not None
            value = ast.literal_eval(node.value)
            assert type(value) in (int, str, bool)
            defaults[node.target.id] = value
    assert defaults["mode"] == "refill"
    return defaults


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirmation-plan", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--trial-observer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = cast(Plan, json.loads(args.confirmation_plan.read_text()))
    model = cast(ModelConfig, json.loads(args.model_config.read_text()))
    assert digest(args.confirmation_plan) == (
        "6677d34e3739999335aecbdfd83c00ea662b30f6f73ad276e36df3ea57ee54be"
    )
    assert digest(args.model_config) == (
        "ce9cc13da41591b8b4deca053d7dfee06424c0228628ee862ea86d725bc163f3"
    )
    assert plan["selection"]["status"] == "SELECTED"
    assert plan["selection"]["candidate"] == "D_P4D3_on"
    assert plan["dtype"] == "float32" and plan["attention_backend"] == "triton"
    assert plan["prefill_depth"] == 4
    config_source = args.source / "vllm_rlt/config.py"
    defaults = scheduler_defaults(config_source)
    assert model["hidden_size"] % model["num_attention_heads"] == 0
    head = model["hidden_size"] // model["num_attention_heads"]
    depth_bytes = (
        2 * model["num_hidden_layers"] * model["num_key_value_heads"] * head * 4
    )
    group_bytes = 4 * 16 * depth_bytes
    pool_bytes = 64 * 1024**3
    records, cases = [], []
    for case in CASES:
        rows = [
            {
                "prompt_id": f"w5/{case.case_id}/{index:04d}",
                "case_id": case.case_id,
                "ordinal": index,
                "target_prompt_tokens": case.lengths[index % len(case.lengths)],
            }
            for index in range(4 * case.concurrency)
        ]
        records.extend(rows)
        spans = [row["target_prompt_tokens"] + 127 for row in rows[: case.batch]]
        required = sum((span + 15) // 16 for span in spans) * group_bytes
        cases.append(
            {
                "case_id": case.case_id,
                "batch_limit_B": case.batch,
                "outstanding_C": case.concurrency,
                "target_resident_A": case.batch,
                "arrival": {
                    "mode": "closed_loop",
                    "initial_requests": case.concurrency,
                    "replacement": "one_per_completion_after_consuming_output_batch",
                    "total_requests": len(rows),
                },
                "prompt_ids": [row["prompt_id"] for row in rows],
                "prompt_length_order": [row["target_prompt_tokens"] for row in rows],
                "output_tokens": 128,
                "ignore_eos": True,
                "scheduler_config": {
                    **defaults,
                    "max_num_seqs": case.batch,
                    "max_num_batched_tokens": 512,
                },
                "first_B_target_kv_bytes": required,
                "position_range_supported": max(spans)
                <= model["max_position_embeddings"],
                "static_target_status": "PLANNED_CAPACITY_EXCEEDED"
                if required > pool_bytes
                else "STATIC_RESERVATION_FITS",
                "capacity_status": "NOT_ADMITTED",
                "native_e2e": "NOT_RUN",
                "scope": (
                    "First B target only; Q64 queued trace is not rejected by target excess. "
                    "Static fit excludes weights/activations/reference/scratch/free memory."
                ),
            }
        )
    args.output.mkdir(exist_ok=False)
    trace = args.output / "prompt-trace-template.json"
    trace.write_text(
        json.dumps(
            {
                "status": "TEMPLATE_NOT_CANONICALLY_TOKENIZED",
                "seed_text": "The scientist studied the stars and recorded the results. ",
                "resolution": (
                    "Canonical pinned tokenizer encode(add_special_tokens=False); repeat seed "
                    "token IDs and slice to exact target length; no chat template or "
                    "truncation of task prompts"
                ),
                "records": records,
            },
            indent=2,
        )
        + "\n"
    )
    manifest = {
        "status": "ALL_ORIGINAL_W5_INPUT_TEMPLATES_PREPARED_NOT_ADMITTED",
        "source_sha": plan["source_sha"],
        "full_source_checkout_verification": "REQUIRED_AT_GPU_ADMISSION",
        "source_config_sha256": digest(config_source),
        "model_repo": plan["model_repo"],
        "model_revision": plan["model_revision"],
        "model_config_sha256": digest(args.model_config),
        "selected_confirmation_input_sha256": digest(args.confirmation_plan),
        "trial_observer_sha256": digest(args.trial_observer),
        "dtype": "float32",
        "attention_backend": "triton",
        "prefill_depth": 4,
        "history_planes": 4,
        "block_size": 16,
        "common_pool_bytes": pool_bytes,
        "cuda_graphs": False,
        "arms": plan["arms"],
        "warmups": 5,
        "measurements": 5,
        "fresh_process_order": [
            [0, "A_P4D4_off"],
            [0, "C_P4D3_off"],
            [0, "D_P4D3_on"],
            [1, "D_P4D3_on"],
            [1, "C_P4D3_off"],
            [1, "A_P4D4_off"],
        ],
        "prompt_template_sha256": digest(trace),
        "resolved_prompt_manifest_sha256": None,
        "device_uuid": None,
        "new_whole_window_grant": "REQUIRED",
        "cases": cases,
        "requires_before_cost": [
            "Canonical exact token IDs and order frozen",
            "Fresh device/source/W2/model/resource checks",
            "Actual per-device memory plan and capacity admission",
            "Complete local baseline on any changed hardware",
            "Full seven-case scope retained; A32 requires larger memory than common64GiB pool",
        ],
    }
    target = args.output / "workload-templates.json"
    target.write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {
                "cases": len(cases),
                "template_requests": len(records),
                "manifest": str(target),
                "sha256": digest(target),
                "GPU_execution": "NOT_RUN",
            }
        )
    )


if __name__ == "__main__":
    main()
