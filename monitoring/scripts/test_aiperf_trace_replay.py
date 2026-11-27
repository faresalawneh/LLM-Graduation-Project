"""Tests for the client tokenizer lock on the way to AIPerf.

No Docker, GPU or network: aiperf is never run.
"""
import importlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

import start_run

SCRIPTS = Path(__file__).resolve().parent
REV = start_run.EXPECTED_MODEL_REVISION


@pytest.fixture
def load_replay(monkeypatch, tmp_path):
    """Import the replay module with the given tokenizer settings."""
    def load(**env):
        for name in ("TOKENIZER", "TOKENIZER_REVISION"):
            monkeypatch.delenv(name, raising=False)
        for name, value in env.items():
            monkeypatch.setenv(name, value)
        import aiperf_trace_replay
        return importlib.reload(aiperf_trace_replay)
    return load


def test_revision_is_passed_to_aiperf(load_replay, tmp_path):
    command = load_replay(TOKENIZER_REVISION=REV).build_aiperf_command(tmp_path)
    assert command[command.index("--tokenizer-revision") + 1] == REV
    assert command.count("--tokenizer-revision") == 1


def test_no_flag_without_a_revision(load_replay, tmp_path):
    command = load_replay().build_aiperf_command(tmp_path)
    assert "--tokenizer-revision" not in command


@pytest.mark.parametrize("env, expected", [({"TOKENIZER_REVISION": REV}, REV), ({}, "")])
def test_manifest_records_what_was_passed(load_replay, tmp_path, env, expected):
    replay = load_replay(**env)
    trace = tmp_path / "trace.csv"
    trace.write_text("Timestamp\n0\n")
    replay.TRACE_FILE = trace
    replay.write_run_manifest(tmp_path, "steady")
    manifest = json.loads((tmp_path / "run_manifest.json").read_text())
    assert manifest["tokenizer_revision"] == expected


def test_the_lock_reaches_the_replay_through_the_real_scripts(tmp_path, monkeypatch):
    """start_run's environment -> run_experiment.sh -> run_local_replay.sh -> python."""
    # "steady" keeps every output inside the sandbox; "smoke" writes to /tmp.
    monitoring = tmp_path / "repo" / "monitoring"
    scripts = monitoring / "scripts"
    scripts.mkdir(parents=True)
    shutil.copyfile(SCRIPTS.parent / "run_experiment.sh", monitoring / "run_experiment.sh")
    for name in ("run_local_replay.sh", "steady_window.csv"):
        shutil.copyfile(SCRIPTS / name, scripts / name)
    (scripts / "run_local_replay.sh").chmod(0o755)
    stubs = tmp_path / "stubs"
    stubs.mkdir()
    bodies = {
        "aiperf": "exit 0", "curl": "exit 0", "docker": "echo 0",
        "nvidia-smi": "exit 0",
        # Stands in for the replay: record its environment, make a run folder.
        "python": 'printf "%s %s" "$HF_HUB_OFFLINE" "$TOKENIZER_REVISION" '
                  '> "$CAPTURE"\nmkdir -p "$ARTIFACT_ROOT/steady_run"',
    }
    for name, body in bodies.items():
        (stubs / name).write_text(f"#!/bin/sh\n{body}\n")
        (stubs / name).chmod(0o755)
    capture = tmp_path / "capture"
    controller = start_run.RunController(
        start_run.Shell(),
        start_run.StackInspector(start_run.Shell(), tmp_path / "repo"))
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    env = {
        **controller.locked_env(),
        "PATH": f"{stubs}{os.pathsep}{os.environ['PATH']}",
        "CAPTURE": str(capture),
        "HOME": str(tmp_path),
    }
    env.pop("ARTIFACT_ROOT", None)
    subprocess.run(
        ["bash", str(monitoring / "run_experiment.sh"), "steady", "--no-mlflow"],
        env=env, check=True, capture_output=True, timeout=60)
    assert capture.read_text() == f"1 {REV}"
    (record_file,) = (tmp_path / "repo" / "artifacts").glob(".trace.*/trace_provenance.json")
    record = json.loads(record_file.read_text())
    assert record["client_hf_hub_offline"] == "1"
    assert record["client_tokenizer_revision"] == REV
