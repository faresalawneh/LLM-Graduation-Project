"""Tests for the version lock handling in stack_up.sh.

The script runs for real in a throwaway directory. docker, curl and pkill are
stubbed, so nothing is started, stopped or downloaded.
"""
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent / "stack_up.sh"
REVISION = "a" * 40
LOCKED_CMD = f"--model\nm\n--revision\n{REVISION}\n--tokenizer-revision\n{REVISION}"
LOCKED_ENV = "VLLM_USE_V1=0\nHF_HUB_OFFLINE=1"
OVERRIDE_TEXT = "services: {}\n"

DOCKER_STUB = r'''#!/bin/sh
echo "$*" >> "$CALLS"
case "$*" in
  "inspect vllm --format {{range .Config.Cmd}}"*) printf '%s\n' "$FAKE_CMD" ;;
  "inspect vllm --format {{range .Config.Env}}"*) printf '%s\n' "$FAKE_ENV" ;;
  "inspect vllm"*) echo 0 ;;
  "inspect -f"*) exit 1 ;;
  inspect*) echo "vllm/vllm-openai@sha256:abc" ;;
  compose*" config") echo "name: monitoring" ;;
esac
exit 0
'''


def run_stack_up(tmp_path, override=True, cmd=LOCKED_CMD, env_lines=LOCKED_ENV,
                 stack_env="dev", create_override=True):
    monitoring = tmp_path / "monitoring"
    monitoring.mkdir()
    shutil.copyfile(SCRIPT, monitoring / "stack_up.sh")
    (monitoring / ".env").write_text("VLLM_IMAGE=vllm/vllm-openai:v0.9.2\n")
    override_path = monitoring / "docker-compose.local.yml"
    if create_override:
        override_path.write_text(OVERRIDE_TEXT)
    stubs = tmp_path / "stubs"
    stubs.mkdir()
    for name, body in (("docker", DOCKER_STUB), ("curl", "#!/bin/sh\nexit 0\n"),
                       ("pkill", "#!/bin/sh\nexit 0\n")):
        (stubs / name).write_text(body)
        (stubs / name).chmod(0o755)
    calls = tmp_path / "calls"
    calls.touch()
    env = {
        "PATH": f"{stubs}{os.pathsep}{os.environ['PATH']}",
        "CALLS": str(calls),
        "FAKE_CMD": cmd,
        "FAKE_ENV": env_lines,
        "STACK_ENV": stack_env,
        "HOME": str(tmp_path),
    }
    if override:
        env["INFERSCOPE_COMPOSE_OVERRIDE"] = str(override_path)
    result = subprocess.run(
        ["bash", str(monitoring / "stack_up.sh")],
        env=env, capture_output=True, text=True, timeout=60, check=False)
    state_file = monitoring / "logs" / "stack_state.json"
    state = json.loads(state_file.read_text()) if state_file.exists() else None
    return result, state, calls.read_text(), override_path


def up_lines(calls):
    return [line for line in calls.splitlines() if line.endswith(" up -d")]


def test_override_is_layered_and_recorded(tmp_path):
    result, state, calls, override = run_stack_up(tmp_path)
    assert result.returncode == 0
    assert up_lines(calls) == [
        f"compose -f docker-compose.yml -f {override} --profile local up -d"]
    assert "NOT locked" not in result.stderr
    assert state["override_active"] == "true"
    assert state["override_path"] == str(override)
    assert state["override_sha256"] == hashlib.sha256(
        OVERRIDE_TEXT.encode()).hexdigest()
    assert state["model_revision"] == REVISION
    assert state["tokenizer_revision"] == REVISION
    assert state["hf_hub_offline"] == "1"
    assert "qwen_revision" not in state


def test_without_override_it_warns_and_records_unlocked(tmp_path):
    result, state, calls, _ = run_stack_up(
        tmp_path, override=False, cmd="--model\nm", env_lines="VLLM_USE_V1=0")
    assert result.returncode == 0
    assert up_lines(calls) == ["compose --profile local up -d"]
    assert "version is NOT locked" in result.stderr
    assert state["override_active"] == "false"
    for key in ("override_path", "override_sha256", "model_revision",
                "tokenizer_revision", "hf_hub_offline"):
        assert state[key] == "unavailable", key
    assert "--revision unavailable" in result.stderr


@pytest.mark.parametrize("kwargs, message", [
    ({"create_override": False}, "override not found"),
    ({"stack_env": "demo"}, "only supported with STACK_ENV=dev"),
])
def test_bad_override_use_starts_nothing(tmp_path, kwargs, message):
    result, state, calls, _ = run_stack_up(tmp_path, **kwargs)
    assert result.returncode == 1
    assert message in result.stderr
    assert up_lines(calls) == []
    assert state is None


def test_a_branch_name_is_not_recorded_as_a_revision(tmp_path):
    state = run_stack_up(tmp_path, cmd="--revision\nmain")[1]
    assert state["model_revision"] == "unavailable"
