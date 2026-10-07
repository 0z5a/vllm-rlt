"""Download pinned public artifacts with bounded memory and no duplicate shards."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
from http.client import IncompleteRead
import json
import os
from pathlib import Path
import shutil
import time
import urllib.request
from urllib.error import URLError


def digest(path: Path, size: int) -> tuple[str, str]:
    sha = hashlib.sha256()
    blob = hashlib.sha1(f"blob {size}\0".encode())
    with path.open("rb") as source:
        for data in iter(lambda: source.read(8 * 1024 * 1024), b""):
            sha.update(data)
            blob.update(data)
    return sha.hexdigest(), blob.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--endpoint",
        choices=("https://huggingface.co", "https://hf-mirror.com"),
        default="https://huggingface.co",
    )
    args = parser.parse_args()
    metadata = json.loads(args.metadata.read_text())
    folder = args.destination
    folder.mkdir(parents=True, exist_ok=True)
    started = time.time()
    files = [
        item
        for item in metadata["files"]
        if item["name"].endswith((".safetensors", ".py", ".json", ".txt", ".model"))
    ]
    needed = sum(item["size"] for item in files if not (folder / item["name"]).exists())
    if (
        not list(folder.glob("*.partial"))
        and shutil.disk_usage(folder).free < needed + 2 * 1024**3
    ):
        raise RuntimeError("insufficient space for model plus 2 GiB reserve")
    receipts = []
    for item in files:
        name, size = item["name"], item["size"]
        assert Path(name).name == name
        target = folder / name
        partial = folder / (name + ".partial")
        parts = folder / (name + ".completed-ranges")
        url = (
            f"{args.endpoint}/{metadata['model']}/resolve/{metadata['revision']}/{name}"
        )
        candidate = target
        if not target.exists():
            candidate = partial
            if size <= 8 * 1024 * 1024:
                with urllib.request.urlopen(url, timeout=60) as response:
                    data = response.read()
                assert len(data) == size
                partial.write_bytes(data)
            else:
                parts.mkdir(exist_ok=True)
                identity = parts / "identity.json"
                expected = dict(
                    model=metadata["model"], revision=metadata["revision"], file=item
                )
                if identity.exists():
                    assert json.loads(identity.read_text()) == expected
                else:
                    identity.write_text(json.dumps(expected))
                descriptor = os.open(partial, os.O_RDWR | os.O_CREAT, 0o600)
                os.ftruncate(descriptor, size)
                chunk = 8 * 1024 * 1024
                ranges = [
                    (start, min(start + chunk, size)) for start in range(0, size, chunk)
                ]

                def fetch(bounds: tuple[int, int]) -> int:
                    start, end = bounds
                    marker = parts / str(start)
                    if marker.exists():
                        assert marker.read_text() == str(end)
                        return end - start
                    request = urllib.request.Request(
                        url + f"?download=true&loopquant={started}-{start}",
                        headers={"Range": f"bytes={start}-{end - 1}"},
                    )
                    for attempt in range(3):
                        try:
                            with urllib.request.urlopen(
                                request, timeout=60
                            ) as response:
                                assert response.status == 206
                                assert (
                                    response.headers["Content-Range"]
                                    == f"bytes {start}-{end - 1}/{size}"
                                )
                                data = response.read()
                            break
                        except (TimeoutError, URLError, IncompleteRead):
                            if attempt == 2:
                                raise
                            print(
                                json.dumps(
                                    dict(file=name, retry=start, attempt=attempt + 1)
                                ),
                                flush=True,
                            )
                    assert len(data) == end - start
                    assert os.pwrite(descriptor, data, start) == len(data)
                    os.fsync(descriptor)
                    marker.write_text(str(end))
                    return len(data)

                received = 0
                with ThreadPoolExecutor(max_workers=args.workers) as executor:
                    for count in executor.map(fetch, ranges):
                        received += count
                        print(
                            json.dumps(dict(file=name, received=received, total=size)),
                            flush=True,
                        )
                os.close(descriptor)
        assert candidate.stat().st_size == size
        sha, blob = digest(candidate, size)
        assert sha == item["sha256"]
        if candidate == partial:
            partial.rename(target)
        receipt = dict(name=name, bytes=size, sha256=sha, independently_verified=True)
        receipts.append(receipt)
        print(json.dumps(receipt), flush=True)
        if parts.exists():
            for marker in parts.iterdir():
                marker.unlink()
            parts.rmdir()
    receipt = dict(
        model=metadata["model"],
        revision=metadata["revision"],
        download_endpoint=args.endpoint,
        files=receipts,
        independently_verified=True,
        start_unix=started,
        end_unix=time.time(),
    )
    (folder / "verified-manifest.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(dict(status="complete", files=len(receipts))), flush=True)


if __name__ == "__main__":
    main()
