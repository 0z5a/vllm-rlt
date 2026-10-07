"""Join the frozen finite controller and retain its natural process exit."""

import json
import os
from pathlib import Path
import subprocess
import sys
import time

root = Path("/root/autodl-tmp/0z5a/loopquant-h20-20261006")
label = "h20-loopformer-native-g0-v3"
started = time.time()
receipt = dict(
    launcher_pid=os.getpid(),
    launcher_start_ticks=int(
        Path(f"/proc/{os.getpid()}/stat").read_text().split(") ", 1)[1].split()[19]
    ),
    start_unix=started,
)
with (root / f"evidence/{label}-controller.log").open("x") as log:
    child = subprocess.Popen(
        [sys.executable, str(root / label / "run.py")],
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    receipt["controller_pid"] = child.pid
    (root / f"evidence/{label}-foreground-running.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    code = child.wait()
receipt.update(returncode=code, end_unix=time.time())
(root / f"evidence/{label}-foreground-exit.json").write_text(
    json.dumps(receipt, indent=2) + "\n"
)
print(json.dumps(receipt), flush=True)
raise SystemExit(code)
