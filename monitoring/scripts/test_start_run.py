"""Tests for start_run.py. No Docker, GPU or network: subprocess is mocked."""
import hashlib
import json
import logging
import subprocess

import pytest

import start_run

IMAGE = "vllm/vllm-openai:v0.9.2"
DIGEST = "vllm/vllm-openai@" + start_run.EXPECTED_VLLM_DIGEST
REV = start_run.EXPECTED_MODEL_REVISION
BASE_COMMAND = ["--model", "Qwen/Qwen2.5-1.5B-Instruct", "--dtype", "float16"]
LOCK_FLAGS = ["--revision", REV, "--tokenizer-revision", REV]
BASE_ENV = ["NVIDIA_VISIBLE_DEVICES=all", "VLLM_USE_V1=0"]
OVERRIDE_TEXT = b"services: {vllm: {}}\n"
OVERRIDE_SHA = hashlib.sha256(OVERRIDE_TEXT).hexdigest()
SHA = "b" * 40
SECRET = "s3cr3t-grafana-value"


class FakeSystem:
    """Stands in for subprocess.run and records every command."""

    def __init__(self):
        self.calls = []
        self.dirty = False
        self.gpu = True
        self.pulled = True
        self.running = None
        self.volumes = [start_run.HF_VOLUME]
        self.digest = DIGEST
        self.base_command = list(BASE_COMMAND)
        self.merged_command = BASE_COMMAND + LOCK_FLAGS
        self.merged_env = BASE_ENV + ["HF_HUB_OFFLINE=1"]
        self.merged_extra = {}
        self.snapshot = True
        self.weights = start_run.EXPECTED_WEIGHTS_SHA256
        self.container_config = {
            "Cmd": BASE_COMMAND + LOCK_FLAGS,
            "Env": BASE_ENV + ["HF_HUB_OFFLINE=1"],
        }
        self.script_exit_codes = []

    def __call__(self, argv, **kwargs):
        tool = argv[0].rsplit("/", 1)[-1]
        args = argv[1:]
        self.calls.append((tool, args, kwargs))
        code, out, err = 0, "", ""
        if tool == "git":
            if "status" in args:
                out = " M file\n" if self.dirty else ""
            else:
                out = SHA + "\n"
        elif tool == "nvidia-smi":
            if self.gpu:
                out = "Test GPU\n"
            else:
                code, err = 9, "no devices"
        elif tool == "docker" and args[:2] == ["image", "inspect"]:
            if self.pulled:
                out = json.dumps([self.digest])
            else:
                code, err = 1, "Error: No such image: " + IMAGE
        elif tool == "docker" and args[0] == "ps":
            out = f"vllm|{self.running}\n" if self.running else ""
        elif tool == "docker" and args[:2] == ["volume", "ls"]:
            out = "\n".join(self.volumes)
        elif tool == "docker" and args[0] == "compose" and "config" in args:
            merged = any(a.endswith(start_run.OVERRIDE_NAME) for a in args)
            vllm = {
                "image": "${VLLM_IMAGE:-x}",
                "command": self.merged_command if merged else self.base_command,
                "environment": self.merged_env if merged else BASE_ENV,
            }
            if merged:
                vllm.update(self.merged_extra)
            out = json.dumps({"services": {"vllm": vllm, "grafana": {"image": "g"}}})
        elif tool == "docker" and args[0] == "inspect":
            out = json.dumps(self.container_config)
        elif tool == "docker" and args[0] in ("run", "exec"):
            if "sha256sum" in args:
                out = f"{self.weights}  {args[-1]}\n"
            elif not self.snapshot:
                code = 1
        elif tool == "bash" and self.script_exit_codes:
            code = self.script_exit_codes.pop(0)
        elif tool == "pkill":
            code = 1
        return subprocess.CompletedProcess(argv, code, out, err)

    def started(self):
        """Commands that would start, stop or change something."""
        return [
            (tool, args) for tool, args, _ in self.calls
            if tool in ("bash", "pkill")
            or (tool == "docker" and args[0] in ("run", "exec", "pull"))
            or (tool == "docker" and args[0] == "compose" and "config" not in args)
        ]


@pytest.fixture
def system(monkeypatch):
    fake = FakeSystem()
    monkeypatch.setattr(start_run.subprocess, "run", fake)
    monkeypatch.setattr(start_run.shutil, "which", lambda tool: f"/usr/bin/{tool}")
    monkeypatch.setattr(start_run.StackInspector, "api_answering", lambda self: False)
    for name in ("VLLM_IMAGE", "STACK_ENV", "MAX_CONCURRENCY",
                 start_run.OVERRIDE_ENV_VAR):
        monkeypatch.delenv(name, raising=False)
    return fake


@pytest.fixture
def repo(tmp_path):
    (tmp_path / "monitoring").mkdir()
    (tmp_path / "monitoring" / start_run.OVERRIDE_NAME).write_bytes(OVERRIDE_TEXT)
    write_env(tmp_path, f"VLLM_IMAGE={IMAGE}\nGRAFANA_ADMIN_PASSWORD={SECRET}\n")
    return tmp_path


def write_env(repo, text):
    (repo / "monitoring" / ".env").write_text(text)


def remove_override(repo):
    (repo / "monitoring" / start_run.OVERRIDE_NAME).unlink()


def make_controller(repo, answer="yes"):
    shell = start_run.Shell()
    prompts = []

    def ask(prompt):
        prompts.append(prompt)
        return answer

    controller = start_run.RunController(
        shell, start_run.StackInspector(shell, repo), ask=ask)
    return controller, prompts


# --- command line ------------------------------------------------------------

def test_no_action_is_rejected(capsys):
    with pytest.raises(SystemExit) as exc:
        start_run.parse_args([])
    assert exc.value.code == 2


@pytest.mark.parametrize("argv", [["--runs", "2"], ["--dry-run"], ["--no-mlflow"]])
def test_type_is_required_for_run_options(argv, capsys):
    with pytest.raises(SystemExit):
        start_run.parse_args(argv)
    with pytest.raises(SystemExit):
        start_run.parse_args(["--up", *argv])
    assert "require" in capsys.readouterr().err


@pytest.mark.parametrize("argv", [
    ["--type", "load"], ["--type"], ["--type", "smoke", "--up"],
    ["--type", "smoke", "--runs", "0"], ["--type", "smoke", "--runs", "x"],
])
def test_invalid_arguments_are_rejected(argv, capsys):
    with pytest.raises(SystemExit) as exc:
        start_run.parse_args(argv)
    assert exc.value.code == 2


def test_runs_defaults_to_one():
    assert start_run.parse_args(["--type", "steady"]).runs == 1


def test_non_dev_stack_env_is_refused(system, monkeypatch):
    monkeypatch.setenv("STACK_ENV", "demo")
    assert start_run.main(["--down"]) == 1
    assert system.calls == []


# --- gate --------------------------------------------------------------------

@pytest.mark.parametrize("answer", ["y", "YES ", "", "YES", "yes ", " yes"])
def test_gate_refuses_anything_but_yes(system, repo, answer):
    controller, prompts = make_controller(repo, answer)
    assert controller.run("steady", 1, True, False) == 1
    assert len(prompts) == 1
    assert not [c for c in system.calls if c[0] == "bash"]


def test_gate_refuses_closed_input(system, repo):
    controller, _ = make_controller(repo)

    def closed(prompt):
        raise EOFError

    controller.ask = closed
    assert controller.confirm() is False


def test_gate_accepts_yes_and_runs_each_time(system, repo):
    controller, _ = make_controller(repo)
    assert controller.run("steady", 3, True, False) == 0
    scripts = [args for tool, args, _ in system.calls if tool == "bash"]
    expected = [str(repo / "monitoring" / "run_experiment.sh"), "steady"]
    assert scripts == [expected] * 3


def test_no_mlflow_is_forwarded(system, repo):
    controller, _ = make_controller(repo)
    controller.run("smoke", 1, False, False)
    scripts = [args for tool, args, _ in system.calls if tool == "bash"]
    assert scripts[0][1:] == ["smoke", "--no-mlflow"]


def test_failed_run_stops_the_rest(system, repo):
    system.script_exit_codes = [0, 5]
    controller, _ = make_controller(repo)
    with pytest.raises(start_run.RunError, match="run 2 of 3 exited with code 5"):
        controller.run("steady", 3, True, False)
    assert len([c for c in system.calls if c[0] == "bash"]) == 2


UNLOCKED = {"Cmd": BASE_COMMAND, "Env": BASE_ENV}
BLOCKERS = {
    "git tree": ({"dirty": True}, None),
    "GPU": ({"gpu": False}, None),
    "vLLM variable": ({}, f"GRAFANA_ADMIN_PASSWORD={SECRET}\n"),
    "Grafana password": ({}, f"VLLM_IMAGE={IMAGE}\nGRAFANA_ADMIN_PASSWORD=\n"),
    "vLLM image": ({"pulled": False}, None),
    "vLLM digest": ({"digest": "vllm/vllm-openai@sha256:" + "0" * 64}, None),
    "version lock": ({"merged_command": BASE_COMMAND}, None),
    "compose drift": (
        {"base_command": BASE_COMMAND + ["--max-model-len", "2048"]}, None),
    "model snapshot": ({"snapshot": False}, None),
    "model weights": ({"weights": "0" * 64}, None),
    "running stack": ({"running": IMAGE, "container_config": UNLOCKED}, None),
}


@pytest.mark.parametrize("action", ["smoke", "steady", "burst", "up"])
@pytest.mark.parametrize("check", list(BLOCKERS))
def test_each_failed_check_blocks_before_the_gate(
        system, repo, check, action, caplog):
    changes, env = BLOCKERS[check]
    for key, value in changes.items():
        setattr(system, key, value)
    if env is not None:
        write_env(repo, env)
    controller, prompts = make_controller(repo, "yes")
    with pytest.raises(start_run.RunError, match="blocked by preflight.*" + check):
        controller.up() if action == "up" else controller.run(action, 1, True, False)
    assert prompts == []
    assert not [c for c in system.calls if c[0] == "bash"]
    assert SECRET not in caplog.text


def test_blockers_cover_every_preflight_check(system, repo):
    controller, _ = make_controller(repo)
    assert [r.name for r in controller.preflight_checks()] == list(BLOCKERS)


def test_preflight_is_repeated_before_each_later_run(system, repo):
    def ask(prompt):
        return "yes"

    controller, _ = make_controller(repo)
    controller.ask = ask
    real_stream = controller.shell.stream

    def stream_then_dirty(*args, **kwargs):
        code = real_stream(*args, **kwargs)
        system.dirty = True
        return code

    controller.shell.stream = stream_then_dirty
    with pytest.raises(start_run.RunError, match="blocked by preflight: git tree"):
        controller.run("steady", 3, True, False)
    assert len([c for c in system.calls if c[0] == "bash"]) == 1


def test_no_bypass_option_exists(capsys):
    for flag in ("--force", "--yes", "-y", "--skip-preflight", "--no-preflight"):
        with pytest.raises(SystemExit):
            start_run.parse_args(["--type", "smoke", flag])


def test_up_is_gated(system, repo):
    controller, prompts = make_controller(repo, "no")
    assert controller.up() == 1
    assert len(prompts) == 1
    assert not [c for c in system.calls if c[0] == "bash"]


def docker_calls(system, verb):
    return [a for t, a, _ in system.calls if t == "docker" and a[0] == verb]


@pytest.mark.parametrize("check", ["model_snapshot", "model_weights"])
def test_helper_containers_never_pull_and_mount_read_only(system, repo, check):
    controller, _ = make_controller(repo)
    getattr(controller.inspector, check)()
    (args,) = docker_calls(system, "run")
    assert args[args.index("--pull") + 1] == "never"
    assert args[args.index("--network") + 1] == "none"
    assert f"{start_run.HF_VOLUME}:/hf:ro" in args
    assert "--gpus" not in args
    assert start_run.EXPECTED_MODEL_REVISION in args[-1]


@pytest.mark.parametrize("check", ["model_snapshot", "model_weights"])
def test_helper_is_not_run_when_volume_is_missing(system, repo, check):
    system.volumes = []
    controller, _ = make_controller(repo)
    with pytest.raises(start_run.RunError, match="no volume"):
        getattr(controller.inspector, check)()
    assert system.started() == []


def test_running_container_is_read_with_exec(system, repo):
    system.running = IMAGE
    controller, _ = make_controller(repo)
    controller.inspector.model_weights()
    assert docker_calls(system, "run") == []
    (args,) = docker_calls(system, "exec")
    assert args[-1].startswith("/root/.cache/huggingface/hub/")


def test_weights_report_the_hash_time_and_use_a_long_timeout(system, repo):
    controller, _ = make_controller(repo)
    assert "hashed in " in controller.inspector.model_weights()
    kwargs = [k for t, a, k in system.calls if t == "docker" and a[0] == "run"][0]
    assert kwargs["timeout"] == start_run.HASH_TIMEOUT_S


# --- version lock, drift, running stack --------------------------------------

def test_missing_override_blocks_with_a_clear_message(system, repo, caplog):
    remove_override(repo)
    controller, prompts = make_controller(repo)
    with pytest.raises(start_run.RunError, match="version lock, compose drift"):
        controller.up()
    assert "override missing" in caplog.text
    assert prompts == []
    assert system.started() == [("docker", a) for a in docker_calls(system, "run")]


@pytest.mark.parametrize("change, message", [
    ({"merged_command": BASE_COMMAND + LOCK_FLAGS[2:]}, "must set --revision"),
    ({"merged_command": BASE_COMMAND + LOCK_FLAGS[:2]},
     "must set --tokenizer-revision"),
    ({"merged_command": BASE_COMMAND + ["--revision", "main"] + LOCK_FLAGS[2:]},
     "must set --revision"),
    ({"merged_command": BASE_COMMAND + LOCK_FLAGS + LOCK_FLAGS[:2]},
     "exactly once"),
    ({"merged_env": BASE_ENV}, "must set HF_HUB_OFFLINE=1"),
    ({"merged_env": BASE_ENV + ["HF_HUB_OFFLINE=0"]}, "must set HF_HUB_OFFLINE=1"),
])
def test_version_lock_failures(system, repo, change, message):
    for key, value in change.items():
        setattr(system, key, value)
    controller, _ = make_controller(repo)
    with pytest.raises(start_run.RunError, match=message):
        controller.inspector.version_lock()


def test_drift_after_a_compose_change_shows_the_diff(system, repo):
    # docker-compose.yml gains a flag that the copied command does not have.
    system.base_command = BASE_COMMAND + ["--enforce-eager"]
    controller, _ = make_controller(repo)
    with pytest.raises(start_run.RunError) as exc:
        controller.inspector.compose_drift()
    assert "differs from docker-compose.yml" in str(exc.value)
    assert "+++ docker-compose.local.yml" in str(exc.value)
    assert "\n---enforce-eager" in str(exc.value)


@pytest.mark.parametrize("change, message", [
    ({"base_command": BASE_COMMAND[:-1] + ["bfloat16"]}, "command differs"),
    ({"merged_command": BASE_COMMAND[:2] + LOCK_FLAGS}, "command differs"),
    ({"merged_env": BASE_ENV + ["HF_HUB_OFFLINE=1", "X=1"]}, "environment"),
    ({"merged_env": ["HF_HUB_OFFLINE=1"]}, "environment"),
    ({"merged_extra": {"image": "other:1"}}, "changes more than"),
])
def test_drift_failures(system, repo, change, message):
    for key, value in change.items():
        setattr(system, key, value)
    controller, _ = make_controller(repo)
    with pytest.raises(start_run.RunError, match=message):
        controller.inspector.compose_drift()


def test_compose_is_only_read_without_interpolation(system, repo, caplog):
    controller, _ = make_controller(repo)
    controller.inspector.compose_drift()
    controller.inspector.version_lock()
    for args in docker_calls(system, "compose"):
        assert "config" in args and "--no-interpolate" in args


def locked_state(sha=OVERRIDE_SHA):
    return {"override_sha256": sha}


def test_locked_running_container_passes(system, repo, monkeypatch):
    system.running = IMAGE
    monkeypatch.setattr(start_run.StackInspector, "stack_state",
                        lambda self: locked_state())
    controller, _ = make_controller(repo)
    assert "locked with the current override" in controller.inspector.running_stack()


@pytest.mark.parametrize("config, state, message", [
    ({"Cmd": BASE_COMMAND + LOCK_FLAGS[2:], "Env": ["HF_HUB_OFFLINE=1"]},
     locked_state(), "--revision is not"),
    ({"Cmd": BASE_COMMAND + LOCK_FLAGS[:2], "Env": ["HF_HUB_OFFLINE=1"]},
     locked_state(), "--tokenizer-revision is not"),
    ({"Cmd": BASE_COMMAND + LOCK_FLAGS, "Env": BASE_ENV},
     locked_state(), "HF_HUB_OFFLINE=1 is not set"),
    ({"Cmd": BASE_COMMAND + LOCK_FLAGS, "Env": ["HF_HUB_OFFLINE=1"]},
     locked_state("0" * 64), "override hash recorded at start differs"),
    ({"Cmd": BASE_COMMAND + LOCK_FLAGS, "Env": ["HF_HUB_OFFLINE=1"]},
     {}, "override hash recorded at start differs"),
])
def test_unlocked_running_container_is_blocked(
        system, repo, monkeypatch, config, state, message):
    system.running = IMAGE
    system.container_config = config
    monkeypatch.setattr(start_run.StackInspector, "stack_state", lambda self: state)
    controller, _ = make_controller(repo)
    with pytest.raises(start_run.RunError, match=message):
        controller.inspector.running_stack()


@pytest.mark.parametrize("action", ["up", "steady"])
def test_starts_always_pass_the_override(system, repo, action):
    controller, _ = make_controller(repo)
    controller.up() if action == "up" else controller.run(action, 2, True, False)
    envs = [k["env"] for t, _, k in system.calls if t == "bash"]
    expected = str(repo / "monitoring" / start_run.OVERRIDE_NAME)
    assert envs and all(e[start_run.OVERRIDE_ENV_VAR] == expected for e in envs)


# --- dry run -----------------------------------------------------------------

@pytest.mark.parametrize("pulled", [True, False])
def test_dry_run_starts_nothing_and_asks_nothing(system, repo, pulled, caplog):
    caplog.set_level(logging.INFO)
    system.pulled = pulled
    controller, prompts = make_controller(repo)
    assert controller.run("steady", 3, True, True) == 0
    assert prompts == []
    assert system.started() == []
    assert "inferscope-gp2" in caplog.text
    assert REV in caplog.text and OVERRIDE_SHA in caplog.text
    assert "not verified: a dry run starts no container" in caplog.text
    assert "nothing started" in caplog.text


def test_dry_run_reports_a_missing_override(system, repo, caplog):
    caplog.set_level(logging.INFO)
    remove_override(repo)
    controller, _ = make_controller(repo)
    assert controller.run("smoke", 1, True, True) == 0
    assert "unavailable: override missing" in caplog.text


def test_dry_run_shows_mlflow_disabled(system, repo, caplog):
    caplog.set_level(logging.INFO)
    controller, _ = make_controller(repo)
    controller.run("burst", 1, False, True)
    assert "disabled" in caplog.text


@pytest.mark.parametrize("running, state, message", [
    ("other:1", {}, "running vllm container uses"),
    (None, {}, "no vllm container"),
    (IMAGE, {"stack_env": "demo", "vllm_image": IMAGE}, "only dev"),
    (IMAGE, {"stack_env": "dev", "vllm_image": "old:1"}, "different vLLM image"),
])
def test_unaccounted_stack_is_refused(
        system, repo, monkeypatch, running, state, message):
    monkeypatch.setattr(start_run.StackInspector, "api_answering", lambda self: True)
    monkeypatch.setattr(start_run.StackInspector, "stack_state", lambda self: state)
    system.running = running
    controller, _ = make_controller(repo)
    with pytest.raises(start_run.RunError, match=message):
        controller.experiment_name(IMAGE)


# --- preflight ---------------------------------------------------------------

def run_preflight(repo, caplog):
    caplog.set_level(logging.INFO)
    controller, prompts = make_controller(repo)
    code = controller.preflight()
    assert prompts == []
    assert SECRET not in caplog.text
    return code


def test_preflight_passes(system, repo, caplog):
    assert run_preflight(repo, caplog) == 0
    assert "preflight passed" in caplog.text


@pytest.mark.parametrize("change, message", [
    ({"dirty": True}, "working tree is dirty"),
    ({"gpu": False}, "no devices"),
    ({"volumes": []}, "model cache missing"),
    ({"snapshot": False}, "snapshot " + REV + " missing"),
    ({"weights": "c" * 64}, "!= expected"),
    ({"digest": "vllm/vllm-openai@sha256:" + "c" * 64}, "vLLM image digest"),
    ({"merged_env": BASE_ENV}, "must set HF_HUB_OFFLINE=1"),
    ({"pulled": False}, "vLLM image not pulled"),
])
def test_preflight_failure_messages(system, repo, caplog, change, message):
    for key, value in change.items():
        setattr(system, key, value)
    assert run_preflight(repo, caplog) == 1
    assert message in caplog.text


@pytest.mark.parametrize("env, message", [
    (f"GRAFANA_ADMIN_PASSWORD={SECRET}\n", "has no VLLM_IMAGE variable"),
    (f"VLLM_IMAGE={IMAGE}\n", "has no GRAFANA_ADMIN_PASSWORD variable"),
    (f"VLLM_IMAGE={IMAGE}\nGRAFANA_ADMIN_PASSWORD=\n", "is empty"),
    (f'VLLM_IMAGE={IMAGE}\nGRAFANA_ADMIN_PASSWORD=""\n', "is empty"),
    (f"VLLM_IMAGE={IMAGE}\nGRAFANA_ADMIN_PASSWORD={SECRET}\n"
     f"GRAFANA_ADMIN_PASSWORD={SECRET}\n", "more than once"),
    (f"VLLM_IMAGE=x:latest\nGRAFANA_ADMIN_PASSWORD={SECRET}\n", "not :latest"),
])
def test_preflight_env_messages(system, repo, caplog, env, message):
    write_env(repo, env)
    assert run_preflight(repo, caplog) == 1
    assert message in caplog.text


def test_preflight_without_env_file(system, repo, caplog):
    (repo / "monitoring" / ".env").unlink()
    assert run_preflight(repo, caplog) == 1
    assert "does not exist" in caplog.text


@pytest.mark.parametrize("exists, message", [
    (True, "already exists"), (False, "does not exist: the .env password"),
])
def test_preflight_reports_grafana_volume(system, repo, caplog, exists, message):
    if exists:
        system.volumes.append(start_run.GRAFANA_VOLUME)
    assert run_preflight(repo, caplog) == 0
    assert message in caplog.text
    changing = [a for t, a, _ in system.calls
                if t == "docker" and a[0] == "volume" and a[1] != "ls"]
    assert changing == []


# --- down --------------------------------------------------------------------

@pytest.mark.parametrize("env", [
    None, f"VLLM_IMAGE={IMAGE}\n", f"VLLM_IMAGE={IMAGE}\nGRAFANA_ADMIN_PASSWORD=\n",
])
def test_down_works_without_grafana_password(system, repo, monkeypatch, env):
    monkeypatch.delenv("GRAFANA_ADMIN_PASSWORD", raising=False)
    monkeypatch.setenv("COMPOSE_FILE", str(repo / "monitoring/docker-compose.yml"))
    if env is None:
        (repo / "monitoring" / ".env").unlink()
    else:
        write_env(repo, env)
    controller, _ = make_controller(repo)
    assert controller.down() == 0
    tool, args, kwargs = [c for c in system.calls if c[0] == "docker"][0]
    assert args == ["compose", "-p", "monitoring", "down"]
    # Run away from the compose file, so it is never interpolated.
    assert not str(kwargs["cwd"]).startswith(str(repo))
    assert "COMPOSE_FILE" not in kwargs["env"]


def test_down_never_removes_volumes(system, repo):
    controller, _ = make_controller(repo)
    controller.down()
    for tool, args, _ in system.calls:
        if tool == "docker":
            assert not {"-v", "--volumes", "--rmi", "prune"} & set(args)


def test_down_failure_is_reported(system, repo, monkeypatch):
    controller, _ = make_controller(repo)
    monkeypatch.setattr(controller.shell, "stream", lambda *a, **k: 1)
    with pytest.raises(start_run.RunError, match="compose down exited with code 1"):
        controller.down()
