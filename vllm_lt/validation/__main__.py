"""CPU-only Q1 probe/report and reservation-scoped numerical A/B execution."""

import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    probe = commands.add_parser("probe", help="freeze inputs, source, dependencies and exact cases")
    probe.add_argument("--suite", required=True, type=Path)
    probe.add_argument("--contract", required=True, type=Path)
    probe.add_argument("--model-path", required=True, type=Path)
    probe.add_argument(
        "--gpu-id", required=True, type=int, help="physical ID selected from scheduler status"
    )
    probe.add_argument("--output", required=True, type=Path)
    run = commands.add_parser("run", help="execute once under the frozen GPU reservation")
    run.add_argument("--plan", required=True, type=Path)
    run.add_argument("--output", required=True, type=Path)
    report = commands.add_parser(
        "report", help="reconcile saved evidence without CUDA or checkpoint access"
    )
    report.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    try:
        if args.command == "probe":
            from .official import _official_classes
            from .schema import make_plan, write_json

            if args.output.exists():
                raise FileExistsError("probe output directory must be new")
            _official_classes()  # CPU-only source/dependency and implementation checks.
            plan = make_plan(args.suite, args.contract, args.model_path, gpu_ids=[args.gpu_id])
            args.output.mkdir(parents=True, exist_ok=False)
            write_json(args.output / "plan.json", plan)
            print(
                json.dumps(
                    {
                        "plan": str(args.output / "plan.json"),
                        "plan_sha256": plan["plan_sha256"],
                        "executions": len(plan["execution_order"]),
                        "comparison_trajectories": len(plan["comparison_order"]),
                        "resource_estimates": plan["resource_estimates"],
                    }
                )
            )
            return 0
        if args.command == "run":
            from .runner import run
            from .schema import read_json

            manifest = run(read_json(args.plan), args.output)
            # Execution completion and numerical qualification are separate outcomes.
            return 0 if manifest["status"] == "complete" else 1
        from .report import write_report

        result = write_report(args.run_dir)
        print(
            json.dumps(
                {
                    "report": str(args.run_dir / "qualification.md"),
                    "evidence_status": result["evidence_status"],
                    "runtime_qualification": result["runtime_qualification"],
                    "runtime_by_dtype": result["runtime_by_dtype"],
                    "counts": result.get("counts"),
                },
                allow_nan=False,
            )
        )
        return (
            0
            if result.get("runtime_qualification") in ("passed", "qualified_on_declared_suite")
            else 1
        )
    except (ValueError, FileExistsError, FileNotFoundError, KeyError) as exc:
        print(f"invalid input: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"execution failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
