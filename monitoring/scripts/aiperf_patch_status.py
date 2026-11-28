#!/usr/bin/env python3
"""Judge whether a run's AIPerf processes used the reader patch (#31).

The hook in aiperf_hooks/ appends one JSON line per event to a status file.
aiperf_trace_replay.py and mlflow_log_run.py both read it through here.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional

HOOK_DIR = Path(__file__).resolve().parent / "aiperf_hooks"
HOOK_PATH = HOOK_DIR / "sitecustomize.py"
STATUS_NAME = "aiperf_patch_status.jsonl"
IDENTITY_KEYS = ("aiperf_version", "hook_sha256", "target_source_sha256")


def read_patch_status(path: Path) -> dict[str, Optional[str]]:
    """Summarize a status file.

    Args:
        path: The run's status file.

    Returns:
        "active" is "true" only if the patch was installed, every line agrees
        on one version, hook and target, that hook is the file in the checkout,
        and at least one conversation was read through the patch. It is None if
        there is no usable file. "problem" says why "active" is not "true".
    """
    summary: dict[str, Optional[str]] = {
        "active": None, "aiperf_version": None, "hook_sha256": None,
        "problem": None,
    }
    try:
        records = [json.loads(line)
                   for line in path.read_text().splitlines() if line.strip()]
        identities = {tuple(r[key] for key in IDENTITY_KEYS) for r in records}
        events = {r["event"] for r in records}
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        summary["problem"] = f"patch status unreadable: {type(exc).__name__}: {exc}"
        return summary
    summary["active"] = "false"
    if len(identities) != 1:
        summary["problem"] = (
            f"patch status lines name {len(identities)} different patches")
        return summary
    version, hook_sha, _ = identities.pop()
    summary["aiperf_version"], summary["hook_sha256"] = version, hook_sha
    if "installed" not in events:
        summary["problem"] = "no process reported installing the patch"
    elif hook_sha != hashlib.sha256(HOOK_PATH.read_bytes()).hexdigest():
        summary["problem"] = "the hook that ran differs from the file in the checkout"
    elif "patched_read" not in events:
        summary["problem"] = "no conversation was read through the patch"
    else:
        summary["active"] = "true"
    return summary
