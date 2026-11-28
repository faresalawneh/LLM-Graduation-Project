"""Tests for the AIPerf reader patch, its status record and the stall guard.

No Docker, GPU, network or AIPerf: a stand-in module takes AIPerf's place.
"""
import hashlib
import importlib
import importlib.util
import inspect
import json
import mmap
import os
import shutil
import signal
import subprocess
import sys
import types
from pathlib import Path
from unittest import mock

import pytest

import aiperf_patch_status as status
import aiperf_trace_replay as replay
import start_run

SCRIPTS = Path(__file__).resolve().parent
# Lines copied from the aiperf.log of the failed burst (#31).
NOTICE_LINE = (
    "2026-11-28 01:45:52.732 - records_manager - NOTICE - All requests have "
    "completed, please wait for the results to be processed (currently 9,477 of "
    "9,483 records processed)...\n")
ALL_RECEIVED = (
    "2026-11-28 14:45:49.642 - records_manager - INFO - Processed 2000 valid "
    "requests and 0 errors (2000 total).\n")


def load_hook():
    """Load the hook under another name, so it does not patch on import."""
    spec = importlib.util.spec_from_file_location(
        "reader_patch_under_test", status.HOOK_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


hook = load_hook()


class FakeSerializationError(Exception):
    pass


def fake_aiperf_module():
    """A stand-in for aiperf.dataset.memory_map_utils with the racy read."""
    class MemoryMapDatasetClient:
        data_file_path = "data"

        def get_conversation(self, conversation_id):
            info = self.index.offsets[conversation_id]
            self.data_mmap.seek(info.offset)
            return self._deserialize_conversation(self.data_mmap.read(info.size))

        def _deserialize_conversation(self, data):
            return data

    return types.SimpleNamespace(
        MemoryMapDatasetClient=MemoryMapDatasetClient,
        MemoryMapSerializationError=FakeSerializationError,
        _logger=mock.MagicMock())


def source_sha(module):
    source = inspect.getsource(module.MemoryMapDatasetClient.get_conversation)
    return hashlib.sha256(source.encode()).hexdigest()


def make_client(module, tmp_path, values):
    data = b"".join(values.values())
    (tmp_path / "data").write_bytes(data)
    client = module.MemoryMapDatasetClient()
    with (tmp_path / "data").open("rb") as handle:  # the map keeps its own handle
        client.data_mmap = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
    offsets, position = {}, 0
    for key, value in values.items():
        offsets[key] = types.SimpleNamespace(offset=position, size=len(value))
        position += len(value)
    client.index = types.SimpleNamespace(offsets=offsets)
    return client


VALUES = {"a": b'{"id":"a"}', "b": b'{"id":"b","more":1}', "c": b'{"id":"c"}'}


# --- the replacement read ----------------------------------------------------

def test_patched_read_returns_the_same_bytes_without_the_cursor(tmp_path):
    module = fake_aiperf_module()
    client = make_client(module, tmp_path, VALUES)
    expected = {key: client.get_conversation(key) for key in VALUES}
    assert expected == VALUES
    hook.install(module, "0.11.0", str(tmp_path / "status"), source_sha(module))
    for key in VALUES:
        client.data_mmap.seek(7)  # another thread's cursor must not matter
        assert client.get_conversation(key) == expected[key]
        assert client.data_mmap.tell() == 7


@pytest.mark.parametrize("offset, size", [(0, 10_000), (10_000, 1), (-4, 4), (0, -1)])
def test_a_range_outside_the_data_raises(tmp_path, offset, size):
    module = fake_aiperf_module()
    client = make_client(module, tmp_path, VALUES)
    hook.install(module, "0.11.0", str(tmp_path / "status"), source_sha(module))
    client.index.offsets["a"] = types.SimpleNamespace(offset=offset, size=size)
    with pytest.raises(FakeSerializationError, match="out of range"):
        client.get_conversation("a")


def test_an_unknown_conversation_is_still_a_key_error(tmp_path):
    module = fake_aiperf_module()
    client = make_client(module, tmp_path, VALUES)
    hook.install(module, "0.11.0", str(tmp_path / "status"), source_sha(module))
    with pytest.raises(KeyError):
        client.get_conversation("nope")


# --- fail-closed -------------------------------------------------------------

def assert_refused(module, version, path, sha, message):
    client = getattr(module, "MemoryMapDatasetClient", None)
    before = getattr(client, "get_conversation", None)
    with pytest.raises(hook.HookError, match=message):
        hook.install(module, version, path, sha)
    assert getattr(client, "get_conversation", None) is before


@pytest.mark.parametrize("version", ["0.12.0", "0.11.1", "0.10.0", ""])
def test_another_version_is_refused(tmp_path, version):
    module = fake_aiperf_module()
    assert_refused(module, version, str(tmp_path / "s"), source_sha(module),
                   "only for 0.11.0")


def test_changed_source_is_refused(tmp_path):
    module = fake_aiperf_module()
    # The recorded 0.11.0 hash does not match the stand-in's source.
    assert_refused(module, "0.11.0", str(tmp_path / "s"),
                   hook.TARGET_SOURCE_SHA256, "source sha256")


def test_a_missing_target_is_refused(tmp_path):
    module = fake_aiperf_module()
    del module.MemoryMapDatasetClient.get_conversation
    assert_refused(module, "0.11.0", str(tmp_path / "s"), "x", "was not found")


@pytest.mark.parametrize("path", [None, ""])
def test_a_missing_status_file_is_refused(path):
    module = fake_aiperf_module()
    assert_refused(module, "0.11.0", path, source_sha(module), "is not set")


def fake_package(root, version):
    """An importable fake aiperf distribution of the given version."""
    package = root / "aiperf" / "dataset"
    package.mkdir(parents=True)
    (root / "aiperf" / "__init__.py").write_text("")
    (package / "__init__.py").write_text("")
    (package / "memory_map_utils.py").write_text(
        "class MemoryMapDatasetClient:\n"
        "    def get_conversation(self, conversation_id):\n"
        "        return None\n")
    info = root / f"aiperf-{version}.dist-info"
    info.mkdir()
    (info / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: aiperf\nVersion: {version}\n")


@pytest.mark.parametrize("version, message", [
    ("0.12.0", "only for 0.11.0"), ("0.11.0", "source sha256")])
def test_a_refused_patch_stops_the_interpreter(tmp_path, version, message):
    """Python ignores exceptions from sitecustomize, so the hook must exit."""
    fake_package(tmp_path, version)
    env = {**os.environ,
           "PYTHONPATH": f"{status.HOOK_DIR}{os.pathsep}{tmp_path}",
           "AIPERF_PATCH_STATUS_FILE": str(tmp_path / "status"),
           "PYTHONDONTWRITEBYTECODE": "1"}
    result = subprocess.run(
        [sys.executable, "-c", "print('program ran')"],
        env=env, capture_output=True, text=True, timeout=60)
    assert result.returncode == hook.FAILURE_EXIT_CODE
    assert "program ran" not in result.stdout
    assert message in result.stderr
    assert not (tmp_path / "status").exists()


# --- status record -----------------------------------------------------------

def write_status(path, events, **changes):
    line = {"pid": 1, "aiperf_version": "0.11.0",
            "hook_sha256": hook.file_sha256(status.HOOK_PATH),
            "target_source_sha256": hook.TARGET_SOURCE_SHA256, **changes}
    path.write_text("".join(
        json.dumps({**line, "event": event}) + "\n" for event in events))
    return path


def test_status_is_active_only_with_a_patched_read(tmp_path):
    good = write_status(tmp_path / "s", ["installed", "forked", "patched_read"])
    assert status.read_patch_status(good)["active"] == "true"
    unread = write_status(tmp_path / "t", ["installed", "forked"])
    assert status.read_patch_status(unread)["active"] == "false"


def test_status_rejects_a_hook_other_than_the_checkout(tmp_path):
    path = write_status(tmp_path / "s", ["installed", "patched_read"],
                        hook_sha256="0" * 64)
    assert "differs from the file" in status.read_patch_status(path)["problem"]


def test_status_rejects_mixed_patches(tmp_path):
    path = write_status(tmp_path / "s", ["installed", "patched_read"])
    other = write_status(tmp_path / "o", ["installed"], aiperf_version="0.12.0")
    path.write_text(path.read_text() + other.read_text())
    assert status.read_patch_status(path)["active"] == "false"


@pytest.mark.parametrize("content", [None, "not json\n", '{"event": "installed"}\n'])
def test_missing_or_broken_status_is_not_a_guess(tmp_path, content):
    path = tmp_path / "s"
    if content is not None:
        path.write_text(content)
    summary = status.read_patch_status(path)
    assert summary["active"] is None and summary["problem"]


# --- stall guard -------------------------------------------------------------

class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def append(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(text)


@pytest.fixture
def guarded(tmp_path):
    clock = Clock()
    return (replay.StallGuard(tmp_path, clock=clock), clock,
            tmp_path / "logs" / "aiperf.log", tmp_path / "profile_export.jsonl")


def test_the_limit_is_120_seconds_and_not_configurable(monkeypatch, tmp_path):
    monkeypatch.setenv("STALL_GUARD_SECONDS", "1")
    importlib.reload(replay)
    assert replay.StallGuard(tmp_path).seconds == 120


def test_guard_stays_quiet_before_the_notice(guarded):
    guard, clock, log_file, _ = guarded
    append(log_file, "x - records_manager - INFO - working\n")
    guard.should_fire()
    clock.now += 10_000
    assert not guard.should_fire()


def test_guard_fires_once_after_the_limit(guarded):
    guard, clock, log_file, records = guarded
    append(records, "{}\n" * 3)
    append(log_file, NOTICE_LINE)
    assert not guard.should_fire()
    assert (guard.processed_at_notice, guard.expected_at_notice) == (9477, 9483)
    clock.now += 119.9
    assert not guard.should_fire()
    clock.now += 0.2
    assert guard.should_fire()
    clock.now += 500
    assert not guard.should_fire()
    assert guard.report()["guard_fired"] is True
    assert guard.report()["records_on_disk_at_fire"] == 3


def test_new_records_restart_the_limit(guarded):
    guard, clock, log_file, records = guarded
    append(log_file, NOTICE_LINE)
    guard.should_fire()
    clock.now += 100
    append(records, "{}\n")
    assert not guard.should_fire()
    clock.now += 100
    assert not guard.should_fire()
    clock.now += 21
    assert guard.should_fire()


def test_guard_disarms_when_all_records_arrived(guarded):
    guard, clock, log_file, _ = guarded
    append(log_file, NOTICE_LINE)
    guard.should_fire()
    append(log_file, ALL_RECEIVED)
    clock.now += 10_000
    assert not guard.should_fire()
    assert guard.report()["all_records_received_seen"] is True


def test_a_line_split_across_reads_is_still_seen(guarded):
    guard, _, log_file, _ = guarded
    append(log_file, NOTICE_LINE[:90])
    guard.observe()
    assert not guard.notice_seen
    append(log_file, NOTICE_LINE[90:])
    guard.observe()
    assert guard.notice_seen


@pytest.mark.parametrize("code, fired, received, active, expected", [
    (0, False, True, "true", 0),
    (0, True, True, "true", 1),
    (0, False, True, "false", 1),
    (0, False, True, None, 1),
    (0, False, False, "true", 1),
    (1, False, False, None, 0),  # AIPerf's own failure keeps its exit code
])
def test_invalid_reasons(tmp_path, code, fired, received, active, expected):
    guard = replay.StallGuard(tmp_path)
    guard.fired, guard.all_received_seen = fired, received
    patch = {"active": active, "problem": "p"}
    assert len(replay.invalid_reasons(code, guard, patch)) == expected


def test_only_aiperf_gets_the_hook(monkeypatch, tmp_path):
    monkeypatch.setenv("PYTHONPATH", "/somewhere")
    env = replay.aiperf_environment(tmp_path)
    assert env["PYTHONPATH"].split(os.pathsep) == [str(status.HOOK_DIR), "/somewhere"]
    assert env["AIPERF_PATCH_STATUS_FILE"] == str(tmp_path / status.STATUS_NAME)
    assert os.environ["PYTHONPATH"] == "/somewhere"


# A stand-in for AIPerf that stalls and, like AIPerf, saves records on SIGINT.
STALLING_AIPERF = """
import signal, sys, time
from pathlib import Path
out = Path(sys.argv[1]); (out / "logs").mkdir()
(out / "profile_export.jsonl").write_text("{}\\n" * 2)
interrupts = []
signal.signal(signal.SIGINT, lambda *_: interrupts.append(1))
(out / "logs" / "aiperf.log").write_text(sys.argv[2])
end = time.time() + 60
while not interrupts and time.time() < end:
    time.sleep(0.05)
time.sleep(0.5)
with (out / "profile_export.jsonl").open("a") as handle:
    handle.write("{}\\n")
(out / "interrupts").write_text(str(len(interrupts)))
"""


def test_a_stalled_run_gets_one_sigint_and_is_invalid(monkeypatch, tmp_path):
    out = tmp_path / "run"
    monkeypatch.setattr(replay, "build_aiperf_command", lambda artifact_dir: [
        sys.executable, "-S", "-c", STALLING_AIPERF, str(artifact_dir), NOTICE_LINE])
    monkeypatch.setattr(replay, "ENABLE_GPU_SCRAPE", False)
    monkeypatch.setattr(replay, "TRACE_FILE", status.HOOK_PATH)
    original = replay.StallGuard
    monkeypatch.setattr(replay, "StallGuard",
                        lambda directory: original(directory, seconds=1.0))
    assert replay.run_scenario("burst", out) == replay.INVALID_RUN_EXIT
    report = json.loads((out / replay.GUARD_REPORT_NAME).read_text())
    assert report["guard_fired"] is True
    assert report["aiperf_exit_code"] == 0
    assert "the stall guard fired" in report["invalid_reasons"]
    assert report["records_on_disk_at_exit"] == 3  # saved after the signal
    assert (out / "interrupts").read_text() == "1"


def test_the_guard_sends_sigint_only(tmp_path):
    process = mock.MagicMock()
    process.wait.side_effect = [subprocess.TimeoutExpired("a", 1)] * 3 + [0]
    guard = mock.MagicMock()
    guard.should_fire.side_effect = [False, True, False]
    assert replay.wait_with_guard(process, guard, 0) == 0
    process.send_signal.assert_called_once_with(signal.SIGINT)
    process.kill.assert_not_called()
    process.terminate.assert_not_called()


@pytest.mark.parametrize("code, logged", [(0, True), (86, True), (1, False)])
def test_run_experiment_logs_an_invalid_run_then_fails(tmp_path, code, logged):
    monitoring = tmp_path / "repo" / "monitoring"
    (monitoring / "scripts").mkdir(parents=True)
    shutil.copyfile(SCRIPTS.parent / "run_experiment.sh", monitoring / "run_experiment.sh")
    replay_script = monitoring / "scripts" / "run_local_replay.sh"
    replay_script.write_text(
        '#!/bin/sh\nmkdir -p "$ARTIFACT_ROOT/steady_run"\n' f"exit {code}\n")
    replay_script.chmod(0o755)
    stubs = tmp_path / "stubs"
    stubs.mkdir()
    for name, body in {"aiperf": "exit 0", "curl": "exit 0", "docker": "echo 0",
                       "nvidia-smi": "exit 0",
                       "python": 'echo "$@" >> "$CAPTURE"'}.items():
        (stubs / name).write_text(f"#!/bin/sh\n{body}\n")
        (stubs / name).chmod(0o755)
    capture = tmp_path / "capture"
    env = {**os.environ, "PATH": f"{stubs}{os.pathsep}{os.environ['PATH']}",
           "CAPTURE": str(capture), "HOME": str(tmp_path)}
    result = subprocess.run(
        ["bash", str(monitoring / "run_experiment.sh"), "steady"],
        env=env, capture_output=True, text=True, timeout=60)
    assert result.returncode == code
    assert capture.exists() is logged
    assert ("crash check" in result.stdout) is logged
    assert replay.INVALID_RUN_EXIT == 86


# --- preflight ---------------------------------------------------------------

def test_preflight_requires_the_hook_file(tmp_path):
    inspector = start_run.StackInspector(start_run.Shell(), tmp_path)
    with pytest.raises(start_run.RunError, match="reader patch missing"):
        inspector.aiperf_hook()


@pytest.mark.parametrize("version", ["0.12.0", "0.11.0"])
def test_preflight_fails_if_the_hook_refuses_to_load(tmp_path, version):
    fake_package(tmp_path / "site", version)
    target = tmp_path / "repo" / start_run.HOOK_RELATIVE
    target.parent.mkdir(parents=True)
    shutil.copyfile(status.HOOK_PATH, target)
    python = tmp_path / "python3"
    python.write_text(
        f'#!/bin/sh\nPYTHONPATH="$PYTHONPATH:{tmp_path / "site"}" '
        f'exec {sys.executable} "$@"\n')
    python.chmod(0o755)
    inspector = start_run.StackInspector(start_run.Shell(), tmp_path / "repo")
    inspector.aiperf_python = lambda: str(python)
    with pytest.raises(start_run.RunError, match="did not load"):
        inspector.aiperf_hook()


@pytest.mark.parametrize("first_line, ok", [
    ("#!/venv/bin/python3.12", True), ("#!/bin/sh", False), ("binary", False)])
def test_preflight_uses_aiperfs_own_interpreter(monkeypatch, tmp_path, first_line, ok):
    executable = tmp_path / "aiperf"
    executable.write_text(first_line + "\n")
    monkeypatch.setattr(start_run.shutil, "which", lambda tool: str(executable))
    inspector = start_run.StackInspector(start_run.Shell(), tmp_path)
    if ok:
        assert inspector.aiperf_python() == first_line[2:]
    else:
        with pytest.raises(start_run.RunError, match="which Python"):
            inspector.aiperf_python()


# --- MLflow params -----------------------------------------------------------

@pytest.fixture
def logger_module(monkeypatch):
    monkeypatch.setitem(sys.modules, "mlflow", mock.MagicMock())
    monkeypatch.delitem(sys.modules, "mlflow_log_run", raising=False)
    yield importlib.import_module("mlflow_log_run")
    sys.modules.pop("mlflow_log_run", None)


@pytest.mark.parametrize("fired", [False, True])
def test_mlflow_records_the_patch_and_guard(logger_module, tmp_path, fired):
    write_status(tmp_path / status.STATUS_NAME, ["installed", "patched_read"])
    (tmp_path / "stall_guard.json").write_text(
        json.dumps({"stall_guard_seconds": 120, "guard_fired": fired}))
    assert logger_module.client_provenance(tmp_path) == {
        "aiperf_version": "0.11.0",
        "aiperf_patch_active": "true",
        "aiperf_patch_sha256": hook.file_sha256(status.HOOK_PATH),
        "stall_guard_seconds": "120",
        "guard_fired": str(fired).lower(),
    }


def test_mlflow_says_unavailable_for_older_runs(logger_module, tmp_path):
    assert set(logger_module.client_provenance(tmp_path).values()) == {"unavailable"}
