"""Receive a bounded stream of exactly the registered checkpoint files."""

from dataclasses import dataclass
import hashlib
from pathlib import Path
import tarfile
from typing import BinaryIO


@dataclass(frozen=True)
class TransferFile:
    name: str
    size: int
    sha256: str


def materialize(
    stream: BinaryIO, destination: Path, files: list[TransferFile]
) -> list[dict]:
    plan = {row.name: row for row in files}
    assert len(plan) == len(files)
    destination.mkdir(exist_ok=False)
    received = {}
    with tarfile.open(fileobj=stream, mode="r|") as archive:
        for member in archive:
            expected = plan[member.name]
            assert member.isfile() and Path(member.name).name == member.name
            assert member.size == expected.size and member.name not in received
            partial = destination / (member.name + ".partial")
            source = archive.extractfile(member)
            assert source is not None
            digest = hashlib.sha256()
            with source, partial.open("xb") as output:
                for chunk in iter(lambda: source.read(8 << 20), b""):
                    output.write(chunk)
                    digest.update(chunk)
            assert partial.stat().st_size == expected.size
            assert digest.hexdigest() == expected.sha256
            partial.rename(destination / member.name)
            received[member.name] = dict(
                name=member.name,
                bytes=expected.size,
                sha256=expected.sha256,
                independently_verified=True,
            )
    assert received.keys() == plan.keys()
    return [received[row.name] for row in files]
