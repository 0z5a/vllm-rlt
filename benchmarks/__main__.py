"""CPU probe, reservation-scoped run, and offline report commands."""

import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    probe = commands.add_parser("probe")
    probe.add_argument("--suite", required=True, type=Path)
    probe.add_argument("--contract", required=True, type=Path)
    probe.add_argument("--model-path", required=True, type=Path)
    probe.add_argument("--output", required=True, type=Path)
    run = commands.add_parser("run")
    run.add_argument("--plan", required=True, type=Path)
    run.add_argument("--output", required=True, type=Path)
    report = commands.add_parser("report")
    report.add_argument("--run-dir", required=True, type=Path)
    report.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        if args.command == "probe":
            from benchmarks.schema import make_plan

            if args.output.exists():
                raise ValueError("probe output directory already exists")
            plan = make_plan(args.suite, args.contract, args.model_path)
            args.output.mkdir(parents=True, exist_ok=False)
            (args.output / "plan.json").write_text(
                json.dumps(plan, indent=2, sort_keys=True, allow_nan=False) + "\n"
            )
            print(
                json.dumps(
                    {"plan": str(args.output / "plan.json"), "plan_sha256": plan["plan_sha256"]}
                )
            )
            return 0
        if args.command == "run":
            from benchmarks.runner import run_plan
            from benchmarks.schema import read_json

            plan = read_json(args.plan)
            result = run_plan(plan, output_dir=args.output)
            return 0 if result["status"] == "complete" else 1
        from benchmarks.report import build_report

        result = build_report(args.run_dir, args.output)
        print(json.dumps({"report": str(args.output / "report.md")}))
        return 0 if result.get("complete", result.get("status") == "complete") else 1
    except (ValueError, FileExistsError, FileNotFoundError, KeyError) as exc:
        print(f"invalid input: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"execution failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
