"""Download approval shared by interactive inference entrypoints."""

import sys
from pathlib import Path

from vllm_rlt.models import resolve_local_model_path


def approve_download(args, parser):
    if getattr(args, "assume_yes", False):
        return True
    local = resolve_local_model_path(args.model, revision=args.revision)
    tokenizer = getattr(args, "tokenizer", None)
    if local is not None and (tokenizer is None or Path(tokenizer).expanduser().is_dir()):
        return False
    if not sys.stdin.isatty():
        parser.error("Download requires approval; pass -y/--yes in non-interactive environments.")
    response = input("Allow model/tokenizer downloads from HuggingFace? (y/yes): ").strip().lower()
    if response not in ("y", "yes"):
        parser.error("Download was not approved.")
    return True
