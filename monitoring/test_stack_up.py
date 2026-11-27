"""Tests that stack_up.sh records the Qwen revision in stack_state.json.

The script runs for real in a throwaway directory. docker, curl and pkill are
stubbed, so nothing is started, stopped or downloaded.
"""
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent / "stack_up.sh"
REVISION = "a" * 40

DOCKER_STUB = r'''#!/bin/sh
echo "$*" >> "$CALLS"
case "$1 $2" in
  "compose config") echo "name: monitoring" ;;
  "exec vllm")
    [ "$FAKE_EXEC_FAILS" = 1 ] && exit 1
    printf '%s' "$FAKE_REVISION" ;;
  "inspect vllm") echo 0 ;;
  "inspect -f") exit 1 ;;
  inspect*) echo "vllm/vllm-openai@sha256:abc" ;;
esac
exit 0
'''


def run_stack_up(tmp_path, revision=REVISION, exec_fails=False):
    monitoring = tmp_path / "monitoring"
    monitoring.mkdir()
    shutil.copyfile(SCRIPT, monitoring / "stack_up.sh")
    (monitoring / ".env").write_text("VLLM_IMAGE=vllm/vllm-openai:v0.9.2\n")
    stubs = tmp_path / "stubs"
    stubs.mkdir()
    for name, body in (("docker", DOCKER_STUB), ("curl", "#!/bin/sh\nexit 0\n"),
                       ("pkill", "#!/bin/sh\nexit 0\n")):
        (stubs / name).write_text(body)
        (stubs / name).chmod(0o755)
    calls = tmp_path / "calls"
    env = {
        "PATH": f"{stubs}{os.pathsep}{os.environ['PATH']}",
        "CALLS": str(calls),
        "FAKE_REVISION": revision,
        "FAKE_EXEC_FAILS": "1" if exec_fails else "0",
        "HOME": str(tmp_path),
    }
    result = subprocess.run(
        ["bash", str(monitoring / "stack_up.sh")],
        env=env, capture_output=True, text=True, timeout=60, check=True)
    state = json.loads((monitoring / "logs" / "stack_state.json").read_text())
    return state, result.stderr, calls.read_text()


def test_records_the_revision_seen_in_the_container(tmp_path):
    state, stderr, calls = run_stack_up(tmp_path)
    assert state["qwen_revision"] == REVISION
    assert "Qwen revision unavailable" not in stderr
    assert ("exec vllm cat /root/.cache/huggingface/hub/"
            "models--Qwen--Qwen2.5-1.5B-Instruct/refs/main") in calls


@pytest.mark.parametrize("revision, exec_fails", [
    ("", False), ("not-a-revision", False), (REVISION, True),
])
def test_unreadable_revision_is_unavailable_with_a_warning(
        tmp_path, revision, exec_fails):
    state, stderr, _ = run_stack_up(tmp_path, revision, exec_fails)
    assert state["qwen_revision"] == "unavailable"
    assert "Qwen revision unavailable" in stderr


def test_other_state_keys_are_unchanged(tmp_path):
    state = run_stack_up(tmp_path)[0]
    assert set(state) == {
        "stack_env", "vllm_image", "vllm_digest", "qwen_revision",
        "restart_count_at_start", "stack_up_time",
    }
