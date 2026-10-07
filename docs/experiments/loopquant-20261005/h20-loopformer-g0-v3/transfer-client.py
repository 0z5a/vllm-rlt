"""Read verified peer assets into the admitted finite H20 stream controller."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--socket", type=Path, required=True)
args = parser.parse_args()
root = Path("/Users/0z5a/Documents/infra/loopquant-20261005")
label = "h20-loopformer-native-g0-v3"
packet = root / "snapshots" / label
authorization = json.loads((packet / "peer-authorization.json").read_text())
model = Path(authorization["model_path"])
owner_manifest = model / "verified-manifest.json"
assert (
    hashlib.file_digest(owner_manifest.open("rb"), "sha256").hexdigest()
    == authorization["manifest_sha256"]
)
assert args.socket.is_socket()
plan = json.loads((packet / "model-plan.json").read_text())
receipt = dict(
    reader_pid=os.getpid(),
    reader_start=subprocess.check_output(
        ["ps", "-p", str(os.getpid()), "-o", "lstart="], text=True
    ).strip(),
    start_unix=time.time(),
    source_model=str(model),
    reader_active=True,
    authorization_sha256=hashlib.sha256(
        (packet / "peer-authorization.json").read_bytes()
    ).hexdigest(),
    manifest_sha256=hashlib.sha256((packet / "manifest.json").read_bytes()).hexdigest(),
)
receipt_path = root / f"evidence/{label}-local-reader.json"
assert not receipt_path.exists()
receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
for row in plan["files"]:
    path = model / row["name"]
    assert path.stat().st_size == row["size"]
    with path.open("rb") as source:
        assert hashlib.file_digest(source, "sha256").hexdigest() == row["sha256"]
receipt["source_files_independently_verified"] = 8
environment = dict(os.environ, COPYFILE_DISABLE="1")
tar = subprocess.Popen(
    ["/usr/bin/tar", "--format=ustar", "-cf", "-", "-C", str(model)]
    + [row["name"] for row in plan["files"]],
    stdout=subprocess.PIPE,
    env=environment,
)
with (root / f"evidence/{label}-foreground.log").open("x") as log:
    ssh = subprocess.Popen(
        [
            "ssh",
            "-S",
            str(args.socket),
            "-p",
            "33885",
            "-o",
            "BatchMode=yes",
            "-o",
            "ProxyCommand=false",
            "root@region-42.seetacloud.com",
            f"/root/miniconda3/bin/python /root/autodl-tmp/0z5a/loopquant-h20-20261006/{label}/launch.py",
        ],
        stdin=tar.stdout,
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    assert tar.stdout is not None
    tar.stdout.close()
    receipt.update(
        tar_pid=tar.pid,
        ssh_pid=ssh.pid,
        tar_start=subprocess.check_output(
            ["ps", "-p", str(tar.pid), "-o", "lstart="], text=True
        ).strip(),
        ssh_start=subprocess.check_output(
            ["ps", "-p", str(ssh.pid), "-o", "lstart="], text=True
        ).strip(),
    )
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)
    ssh_code = ssh.wait()
tar_code = tar.wait()
receipt.update(
    ssh_returncode=ssh_code,
    tar_returncode=tar_code,
    end_unix=time.time(),
    reader_active=False,
)
receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(receipt), flush=True)
raise SystemExit(0 if ssh_code == tar_code == 0 else 1)
