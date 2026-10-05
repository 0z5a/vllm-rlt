"""Freeze performance-only prompt banks and unit-rate arrivals from calibration tokens."""

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random


@dataclass(frozen=True)
class Family:
    name: str
    context: int
    primary: int
    long: int
    loops: int
    mixed_depths: tuple[int, ...]


families = [
    Family("huginn-0125", 4096, 1024, 3968, 32, (8, 16, 32)),
    Family("nanbeige-4.2-3b", 262144, 1024, 4096, 2, (1, 2)),
    Family("hrm-text-1b", 4096, 1024, 3968, 2, ()),
    Family("loopformer-3block-8iterations", 1024, 512, 896, 8, ()),
    Family("parcae-370m", 2048, 1024, 1920, 8, ()),
]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--root", type=Path, required=True)
args = parser.parse_args()
base = args.root / "serving-families-v1"
base.mkdir(exist_ok=False)
summary = []
for family in families:
    data = args.root / "data-families-v1" / family.name
    manifest = json.loads((data / "manifest.json").read_text())
    calibration = data / "calibration.jsonl"
    digest = hashlib.sha256(calibration.read_bytes()).hexdigest()
    assert digest == manifest["splits"]["calibration"]["sha256"]
    bank = [
        token
        for line in calibration.read_text().splitlines()
        for token in json.loads(line)["token_ids"]
    ]
    output = base / family.name
    output.mkdir()
    rng = random.Random(17)
    shapes = {
        "smoke": (256, 32),
        "primary": (family.primary, 128),
        "long": (family.long, 128),
        "decode": (512, 512),
    }
    files, traces = {}, {}
    for label, (length, generated) in shapes.items():
        assert length + generated <= family.context
        rows = []
        for index, start in enumerate(rng.sample(range(len(bank) - length + 1), 512)):
            rows.append(
                dict(
                    request_id=f"{label}-{index}",
                    token_ids=bank[start : start + length],
                    output_tokens=generated,
                    loops=family.loops,
                    bank_start=start,
                )
            )
        assert len({tuple(row["token_ids"]) for row in rows}) == 512
        traces[label] = rows
    mixed_rng = random.Random(29)
    schedule = list(shapes) * 128
    mixed_rng.shuffle(schedule)
    mixed = []
    for index, label in enumerate(schedule):
        row = dict(traces[label][index])
        row.update(request_id=f"mixed-{index}", source_shape=label)
        mixed.append(row)
    traces["mixed-length"] = mixed
    if family.mixed_depths:
        depths = [family.mixed_depths[i % len(family.mixed_depths)] for i in range(512)]
        mixed_rng.shuffle(depths)
        traces["mixed-depth"] = [
            dict(row, request_id=f"mixed-depth-{i}", loops=depth)
            for i, (row, depth) in enumerate(zip(mixed, depths))
        ]
    for label, rows in traces.items():
        file = output / (label + ".json")
        file.write_text(json.dumps(rows) + "\n")
        files[file.name] = dict(
            requests=len(rows),
            prompt_lengths=sorted({len(row["token_ids"]) for row in rows}),
            output_lengths=sorted({row["output_tokens"] for row in rows}),
            decode_depths=sorted({row["loops"] for row in rows}),
            sha256=hashlib.sha256(file.read_bytes()).hexdigest(),
        )
    arrival_rng = random.Random(37)
    arrivals = [0.0]
    for _ in range(511):
        arrivals.append(arrivals[-1] + arrival_rng.expovariate(1.0))
    file = output / "unit-rate-arrivals.json"
    file.write_text(json.dumps(arrivals) + "\n")
    result = dict(
        model=family.name,
        model_revision=manifest["model_revision"],
        tokenizer_sha256=manifest["tokenizer_sha256"],
        calibration_sha256=digest,
        source="concatenated calibration token bank; synthetic performance input only",
        quality_use=False,
        context_limit=family.context,
        prefill_loops=family.loops,
        prefix_reuse=False,
        prompt_seed=17,
        mix_seed=29,
        files=files,
        mixed_depth_status="prepared; runtime qualification pending"
        if family.mixed_depths
        else "unsupported by the registered fixed-depth native path",
        open_arrivals=dict(
            file=file.name,
            sha256=hashlib.sha256(file.read_bytes()).hexdigest(),
            seed=37,
            protocol="divide unit-rate times by a separately frozen offered req/s",
            rate_status="not frozen; derive near-capacity points from independent BF16 only",
        ),
        formal_performance_status="unrun; SLO, rates, candidate and paired-trial schedule still required",
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    (output / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    summary.append(result)
    print(
        json.dumps(dict(model=family.name, files=len(files), status="prepared")),
        flush=True,
    )
(base / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
