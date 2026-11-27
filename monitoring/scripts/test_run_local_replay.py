"""Tests for the trace step of run_local_replay.sh.

The script runs for real in a throwaway copy of the repo, with aiperf, curl and
python stubbed, so no Docker, GPU or network is used.
"""
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent
TRACES = {"steady": "steady_window.csv", "burst": "burst_window.csv"}
CRLF = b"timestamp,model\r\n0,a\r\n1,b\r\n"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_stub(directory, name, body):
    path = directory / name
    path.write_text("#!/bin/sh\n" + body)
    path.chmod(0o755)


@pytest.fixture
def sandbox(tmp_path):
    scripts = tmp_path / "repo" / "monitoring" / "scripts"
    scripts.mkdir(parents=True)
    for name in ("run_local_replay.sh", *TRACES.values()):
        shutil.copyfile(SCRIPTS / name, scripts / name)
    stubs = tmp_path / "stubs"
    stubs.mkdir()
    make_stub(stubs, "aiperf", "exit 0\n")
    make_stub(stubs, "curl", "exit 0\n")
    # Stands in for the replay: keep what the script handed over.
    make_stub(stubs, "python",
              'cp "$TRACE_FILE" "$CAPTURE/replayed.csv"\n'
              'printf "%s" "$TRACE_FILE" > "$CAPTURE/trace_file"\n')
    capture = tmp_path / "capture"
    capture.mkdir()
    env = {
        "PATH": f"{stubs}{os.pathsep}{os.environ['PATH']}",
        "CAPTURE": str(capture),
        "ARTIFACT_ROOT": str(tmp_path / "artifacts"),
        "HOME": str(tmp_path),
    }
    return scripts, capture, env


def replay(sandbox, scenario, **extra):
    scripts, capture, env = sandbox
    subprocess.run(
        ["bash", str(scripts / "run_local_replay.sh"), scenario],
        env={**env, **extra}, check=True, capture_output=True, timeout=60)
    used = Path((capture / "trace_file").read_text())
    record = json.loads((used.parent / "trace_provenance.json").read_text())
    return used, (capture / "replayed.csv").read_bytes(), record


@pytest.mark.parametrize("scenario", TRACES)
def test_tracked_traces_are_byte_identical_afterwards(sandbox, scenario):
    scripts = sandbox[0]
    before = {n: sha256(scripts / n) for n in TRACES.values()}
    real_before = {n: sha256(SCRIPTS / n) for n in TRACES.values()}
    original = scripts / TRACES[scenario]
    used, replayed, record = replay(sandbox, scenario)
    assert {n: sha256(scripts / n) for n in TRACES.values()} == before
    assert {n: sha256(SCRIPTS / n) for n in TRACES.values()} == real_before
    assert used != original
    assert used.name == original.name
    assert replayed == original.read_bytes().replace(b"\r\n", b"\n")
    assert record["trace_original_path"] == str(original)
    assert record["trace_original_sha256"] == before[original.name]
    assert record["trace_used"] == str(used)


@pytest.mark.parametrize("scenario", TRACES)
def test_crlf_trace_is_stripped_only_in_the_copy(sandbox, scenario):
    # CI checks out LFS pointers, so supply real CRLF content here.
    original = sandbox[0] / TRACES[scenario]
    original.write_bytes(CRLF)
    used, replayed, record = replay(sandbox, scenario)
    assert original.read_bytes() == CRLF
    assert replayed == CRLF.replace(b"\r\n", b"\n")
    assert record["trace_original_sha256"] == hashlib.sha256(CRLF).hexdigest()


def test_copy_lives_under_the_artifact_root_and_runs_do_not_collide(sandbox):
    first = replay(sandbox, "steady")[0]
    second = replay(sandbox, "steady")[0]
    root = Path(sandbox[2]["ARTIFACT_ROOT"])
    assert first != second
    for used in (first, second):
        assert used.parent.parent == root
        assert used.parent.name.startswith(".trace.")
    # run_experiment.sh finds run folders with "<scenario>_*": no match here.
    assert not list(root.glob("steady_*"))


@pytest.mark.parametrize("extra, expected", [({}, "20"), ({"MAX_CONCURRENCY": "7"}, "7")])
def test_records_the_concurrency_used(sandbox, extra, expected):
    assert replay(sandbox, "burst", **extra)[2]["max_concurrency"] == expected


def test_git_dirty_is_unavailable_outside_a_repository(sandbox):
    record = replay(sandbox, "burst", GIT_CEILING_DIRECTORIES="/")[2]
    assert record["git_dirty"] == "unavailable"


@pytest.mark.parametrize("dirty, expected", [(False, "false"), (True, "true")])
def test_records_git_dirty_at_run_time(sandbox, dirty, expected):
    repo = sandbox[0].parents[1]
    git = ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t"]
    subprocess.run([*git, "init", "-q"], check=True)
    subprocess.run([*git, "add", "-A"], check=True)
    subprocess.run([*git, "commit", "-qm", "x"], check=True)
    if dirty:
        (repo / "untracked.txt").write_text("x")
    assert replay(sandbox, "steady")[2]["git_dirty"] == expected
