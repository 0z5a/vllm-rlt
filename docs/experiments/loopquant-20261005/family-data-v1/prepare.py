"""Prepare immutable family tokenization without model evaluation or downloads."""

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
import re

import pyarrow.parquet as pq
from tokenizers import Tokenizer


@dataclass(frozen=True)
class Family:
    name: str
    revision: str
    tokenizer: Path
    tokenizer_revision: str
    tokenizer_sha256: str
    special_tokens: bool
    continuation: bool = False


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--root", type=Path, required=True)
parser.add_argument("--peer-root", type=Path, required=True)
args = parser.parse_args()
root, peer = args.root, args.peer_root
families = [
    Family(
        "huginn-0125",
        "bb6621b65e90b6a4b9b29ef88dc83866d450470c",
        root / "models/huginn-0125/tokenizer.json",
        "bb6621b65e90b6a4b9b29ef88dc83866d450470c",
        "9cc201a5061b70aba0d227ef9766fcfe21e0989e0c0ba442b4d7732c4de12308",
        True,
    ),
    Family(
        "nanbeige-4.2-3b",
        "b82e54bd609793562a75cbf9337970a93369eab5",
        root / "models/nanbeige-4.2-3b/tokenizer.json",
        "b82e54bd609793562a75cbf9337970a93369eab5",
        "1d858a0fc007f22af6ae18bfa1ae52d30e398aa9cd1ea06e7777176869346a3f",
        True,
    ),
    Family(
        "hrm-text-1b",
        "22097cbcecdd1301afe30a19a3ee61b96a9863e5",
        peer / "models/hrm-text-1b/tokenizer.json",
        "22097cbcecdd1301afe30a19a3ee61b96a9863e5",
        "ba520107dc9b27d25643e8f43fc1d9921038c86d3ada9dcaabf18526c5836d50",
        False,
        True,
    ),
    Family(
        "loopformer-3block-8iterations",
        "2b4fbaaf4e2510353ef7cfe07a07c671b5226739",
        peer / "models/loopformer-tokenizer/tokenizer.json",
        "2b4fbaaf4e2510353ef7cfe07a07c671b5226739",
        "1fe93b6152957cf9cfd6d89002467f789ce8b3f3e000b3a2edf27c808ddd0b9e",
        False,
    ),
    Family(
        "parcae-370m",
        "439284464ee4999bd1f762da7d044613a4828efe",
        peer / "models/parcae-tokenizer/tokenizer.json",
        "6247b5d0592876b73660f34c2dd16c9db9e9045c",
        "e0021e26057088d68047dbe6e77e2ba1c9fe9ae45bae3df9d67d48c82405ea77",
        False,
    ),
]
receipts = {
    row["file"]: row
    for row in json.loads((root / "data/download-manifest.json").read_text())
}
documents: dict[str, list[tuple[str, str]]] = {}
owners: dict[str, str] = {}
for source, split in [
    ("train", "calibration"),
    ("validation", "dev"),
    ("test", "locked_test"),
]:
    file = root / "data" / f"wiki-{source}.parquet"
    assert (
        hashlib.sha256(file.read_bytes()).hexdigest() == receipts[file.name]["sha256"]
    )
    lines = pq.ParquetFile(file).read(use_threads=False).column("text").to_pylist()
    texts, current = [], []
    for line in lines:
        if re.match(r"^\s*= [^=].*[^=] =\s*$", line) and current:
            texts.append("".join(current))
            current = []
        current.append(line)
    if current:
        texts.append("".join(current))
    documents[split] = []
    for text in texts:
        if not text.strip():
            continue
        digest = hashlib.sha256(text.encode()).hexdigest()
        assert digest not in owners or owners[digest] == split
        owners[digest] = split
        documents[split].append((digest, text))

base = root / "data-families-v1"
base.mkdir(exist_ok=False)
summary = []
for family in families:
    assert (
        hashlib.sha256(family.tokenizer.read_bytes()).hexdigest()
        == family.tokenizer_sha256
    )
    tokenizer = Tokenizer.from_file(str(family.tokenizer))
    output = base / family.name
    output.mkdir()
    counts = {}
    for split, docs in documents.items():
        windows = []
        for digest, text in docs:
            ids = tokenizer.encode(text, add_special_tokens=family.special_tokens).ids
            for start in range(0, len(ids) - 1, 512):
                tokens = ids[start : start + 512]
                if len(tokens) < (256 if family.continuation else 128):
                    continue
                row = dict(
                    document_id=digest,
                    content_sha256=digest,
                    source_revision="b08601e04326c79dfdd32d625aee71d232d685c3",
                    split=split,
                    start=start,
                    token_ids=tokens,
                )
                if family.continuation:
                    row["prefix_length"] = 128
                windows.append(row)
        random.Random(17).shuffle(windows)
        if split != "locked_test":
            windows = windows[:128]
            assert len(windows) == 128
        file = output / (split + ".jsonl")
        file.write_text("".join(json.dumps(row) + "\n" for row in windows))
        counts[split] = dict(
            windows=len(windows),
            documents=len({r["document_id"] for r in windows}),
            input_tokens=sum(len(r["token_ids"]) for r in windows),
            scored_targets=sum(
                len(r["token_ids"]) - (128 if family.continuation else 1)
                for r in windows
            ),
            sha256=hashlib.sha256(file.read_bytes()).hexdigest(),
        )
    manifest = dict(
        model=family.name,
        model_revision=family.revision,
        dataset_revision="b08601e04326c79dfdd32d625aee71d232d685c3",
        tokenizer_revision=family.tokenizer_revision,
        tokenizer_sha256=family.tokenizer_sha256,
        document_add_special_tokens=family.special_tokens,
        special_token_protocol="encode each complete document before token-window slicing; no reinsertion at later windows",
        scoring="bidirectional prefix128 then causal continuation"
        if family.continuation
        else "causal next-token full window",
        window=512,
        stride=512,
        minimum_tokens=256 if family.continuation else 128,
        seed=17,
        document_overlap_between_splits=0,
        splits=counts,
        locked_test_opened_for_model_evaluation=False,
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    summary.append(manifest)
    print(json.dumps(dict(family=family.name, splits=counts)), flush=True)
(base / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
