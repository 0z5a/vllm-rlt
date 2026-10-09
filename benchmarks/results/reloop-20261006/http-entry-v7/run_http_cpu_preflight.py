"""Run the complete finite HTTP CPU preflight in a manager-admitted window."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cli(script, **paths):
    return [sys.executable, str(script)] + [
        argument
        for name, path in paths.items()
        for argument in ("--" + name.replace("_", "-"), str(path))
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in [
        "admission",
        "source",
        "source-manifest",
        "harness",
        "observer",
        "protocol",
        "prompts",
        "selection-input",
        "model",
        "output",
    ]:
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    admission = json.loads(args.admission.read_text())
    assert admission["status"] == "HTTP_CPU_PREFLIGHT_ADMITTED"
    assert sys.platform == "linux" and admission["controller_pid"] == os.getppid()
    assert Path(sys.executable).resolve() == Path(admission["python"]).resolve()
    assert (
        time.time() + admission["forecast_seconds"] + admission["offbox_seconds"]
        < admission["archive_by_unix"]
    )
    assert digest(Path(__file__)) == admission["preflight_driver_sha256"]
    for name in [
        "source_manifest",
        "protocol",
        "prompts",
        "selection_input",
        "observer",
    ]:
        assert digest(vars(args)[name]) == admission[name + "_sha256"]
    files = json.loads(args.source_manifest.read_text())
    assert len(files) == 133
    for name, row in files.items():
        assert not Path(name).is_absolute() and ".." not in Path(name).parts
        path = args.source / name
        assert path.stat().st_size == row["bytes"] and digest(path) == row["sha256"]
    assert set(admission["harness_files"]) == {
        "prepare_http_prompts.py",
        "http_fixed_work.py",
        "check_http_fixed_work.py",
        "audit_http_cost_arm.py",
        "run_http_cost_arm.py",
        "run_http_cpu_preflight.py",
    }
    for name, expected in admission["harness_files"].items():
        assert Path(name).name == name and digest(args.harness / name) == expected
    reference = {
        "fp32_candidate_cost_trial.py": (
            "8f581b7c1c2fd474a53c2126519964bb7cd3d7bb18cb5096a99437b72b14e5a9"
        ),
        "audit_w5_native_trials.py": (
            "35b94e8cdd1d69d700feeeafee7ed714d674727ae39f606ae63872fe6827d5d6"
        ),
        "run_w5_cost_arm.py": "63aafc4af99271702c147552171ddef834b554c3002fb14776e96ebf5a1ee5dc",
        "prepare_w5_workloads.py": (
            "c04400c1b3e344ac5afa06ae83200706c91b08a0e1d13daad18b850e5967603f"
        ),
    }
    for name, expected in reference.items():
        assert digest(args.observer.parent / name) == expected
    args.output.mkdir(exist_ok=False)
    inputs = args.output / "HTTP-inputs.json"
    finite = args.output / "finite-HTTP"
    junit = args.output / "socket-CLI.xml"
    commands = [
        (
            "socket-CLI",
            [
                sys.executable,
                "-m",
                "pytest",
                str(args.source / "tests/test_serving.py"),
                "-q",
                "-k",
                "loopcd",
                "--junitxml",
                str(junit),
            ],
        ),
        (
            "prompts",
            cli(
                args.harness / "prepare_http_prompts.py",
                source=args.source,
                model=args.model,
                protocol=args.protocol,
                prompts=args.prompts,
                selection_input=args.selection_input,
                output=inputs,
            ),
        ),
        (
            "finite-HTTP",
            cli(
                args.harness / "check_http_fixed_work.py",
                source=args.source,
                harness=args.harness,
                observer=args.observer,
                output=finite,
            ),
        ),
    ]
    environment = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES="",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        PYTHONDONTWRITEBYTECODE="1",
    )
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(args.source), str(args.harness), str(args.observer.parent)]
    )
    waits = []
    for name, command in commands:
        with (
            (args.output / (name + ".stdout")).open("wb") as stdout,
            (args.output / (name + ".stderr")).open("wb") as stderr,
        ):
            child = subprocess.Popen(
                command, cwd=args.source, env=environment, stdout=stdout, stderr=stderr
            )
            row = {
                "phase": name,
                "argv": command,
                "actual_pid": child.pid,
                "started_unix": time.time(),
            }
            (args.output / (name + "-start.json")).write_text(
                json.dumps(row, indent=2) + "\n"
            )
            row.update(natural_wait=child.wait(), finished_unix=time.time())
        waits.append(row)
        (args.output / (name + "-wait.json")).write_text(
            json.dumps(row, indent=2) + "\n"
        )
    passed = all(row["natural_wait"] == 0 for row in waits)
    prompt_check = json.loads(inputs.read_text()) if inputs.exists() else None
    finite_check = (
        json.loads((finite / "check-result.json").read_text())
        if (finite / "check-result.json").exists()
        else None
    )
    suites = ET.parse(junit).getroot().findall("testsuite") if junit.exists() else []
    tests = sum(int(suite.attrib["tests"]) for suite in suites)
    failed = sum(
        int(suite.attrib[k])
        for suite in suites
        for k in ["failures", "errors", "skipped"]
    )
    passed = (
        passed
        and tests == 14
        and failed == 0
        and prompt_check is not None
        and finite_check is not None
    )
    source_unchanged = all(
        digest(args.source / name) == row["sha256"] for name, row in files.items()
    )
    passed = passed and source_unchanged
    if passed:
        assert prompt_check["state"] == "EXACT_HTTP_STRING_PROMPTS_PASS"
        assert (
            not prompt_check["mismatched_prompt_ids"]
            and prompt_check["loader_origin_matches_bound_source"]
        )
        assert (
            prompt_check["loader_source_sha256"]
            == files["vllm_rlt/models/__init__.py"]["sha256"]
        )
        assert finite_check["state"] == "ACTUAL_CPU_FINITE_HTTP_INTEGRATION_PASS"
        assert (
            finite_check["healthy_raw_independent_auditor_pass"]
            and finite_check["stream_failure_raw_preservation_pass"]
        )
        assert (
            not finite_check["CUDA_initialized"]
            and not finite_check["checkpoint_weights_loaded"]
        )
        assert finite_check["http_raw_auditor_sha256"] == digest(
            args.harness / "audit_http_cost_arm.py"
        )
    receipt = {
        "state": "ACTUAL_CANONICAL_HTTP_SOURCE_PREFLIGHT_PASS"
        if passed
        else "HTTP_CPU_PREFLIGHT_FAILED_OR_INCOMPLETE",
        "source_manifest_sha256": digest(args.source_manifest),
        "source_files_unchanged_after_all_CPU_phases": source_unchanged,
        "HTTP_input_sha256": digest(inputs) if inputs.exists() else None,
        "http_helper_sha256": digest(args.harness / "http_fixed_work.py"),
        "http_raw_auditor_sha256": digest(args.harness / "audit_http_cost_arm.py"),
        "preflight_driver_sha256": digest(Path(__file__)),
        "admission_sha256": digest(args.admission),
        "actual_socket_CLI_tests_exit_code": waits[0]["natural_wait"],
        "actual_socket_CLI_tests_passed": tests if failed == 0 else None,
        "actual_tokenizer_roundtrip_exit_code": waits[1]["natural_wait"],
        "finite_HTTP_integration_exit_code": waits[2]["natural_wait"],
        "healthy_raw_independent_auditor_pass": bool(passed),
        "stream_failure_raw_preservation_pass": bool(passed),
        "finite_HTTP_check_sha256": digest(finite / "check-result.json")
        if finite_check
        else None,
        "phase_waits": [row["natural_wait"] for row in waits],
        "public_checkpoint_model_or_GPU_E2E_executed": False,
        "whole_task_n": 0,
    }
    (args.output / "CPU-preflight.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                k: receipt[k]
                for k in [
                    "state",
                    "phase_waits",
                    "actual_socket_CLI_tests_passed",
                    "whole_task_n",
                ]
            }
        )
    )
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
