"""Tests for start_run.py. No Docker, GPU or network: subprocess is mocked."""
import json
import logging
import subprocess

import pytest

import start_run

IMAGE = "vllm/vllm-openai:v0.9.2"
DIGEST = "vllm/vllm-openai@sha256:" + "a" * 64
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
        self.revision = start_run.EXPECTED_QWEN_REVISION
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
                out = json.dumps([DIGEST])
            else:
                code, err = 1, "Error: No such image: " + IMAGE
        elif tool == "docker" and args[0] == "ps":
            out = f"vllm|{self.running}\n" if self.running else ""
        elif tool == "docker" and args[:2] == ["volume", "ls"]:
            out = "\n".join(self.volumes)
        elif tool == "docker" and args[0] in ("run", "exec"):
            out = self.revision + "\n"
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
            or (tool == "docker" and args[0] in ("run", "exec", "compose", "pull"))
        ]


@pytest.fixture
def system(monkeypatch):
    fake = FakeSystem()
    monkeypatch.setattr(start_run.subprocess, "run", fake)
    monkeypatch.setattr(start_run.shutil, "which", lambda tool: f"/usr/bin/{tool}")
    monkeypatch.setattr(start_run.StackInspector, "api_answering", lambda self: False)
    for name in ("VLLM_IMAGE", "STACK_ENV", "MAX_CONCURRENCY"):
        monkeypatch.delenv(name, raising=False)
    return fake


@pytest.fixture
def repo(tmp_path):
    (tmp_path / "monitoring").mkdir()
    write_env(tmp_path, f"VLLM_IMAGE={IMAGE}\nGRAFANA_ADMIN_PASSWORD={SECRET}\n")
    return tmp_path


def write_env(repo, text):
    (repo / "monitoring" / ".env").write_text(text)


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


@pytest.mark.parametrize("change, message", [
    ({"pulled": False}, "is not pulled"),
    ({"volumes": []}, "Qwen cache missing"),
])
def test_real_run_raises_before_the_gate(system, repo, change, message):
    for key, value in change.items():
        setattr(system, key, value)
    controller, prompts = make_controller(repo)
    with pytest.raises(start_run.RunError, match=message):
        controller.run("smoke", 1, True, False)
    assert prompts == []
    assert not [c for c in system.calls if c[0] == "bash"]


def test_up_is_gated(system, repo):
    controller, prompts = make_controller(repo, "no")
    assert controller.up() == 1
    assert len(prompts) == 1
    assert not [c for c in system.calls if c[0] == "bash"]


def test_helper_container_never_pulls_and_mounts_read_only(system, repo):
    controller, _ = make_controller(repo)
    controller.inspector.qwen_revision(IMAGE, allow_helper=True)
    args = [a for t, a, _ in system.calls if t == "docker" and a[0] == "run"][0]
    assert args[args.index("--pull") + 1] == "never"
    assert args[args.index("--network") + 1] == "none"
    assert f"{start_run.HF_VOLUME}:/hf:ro" in args


def test_helper_is_not_run_when_volume_is_missing(system, repo):
    system.volumes = []
    controller, _ = make_controller(repo)
    with pytest.raises(start_run.RunError, match="no volume"):
        controller.inspector.qwen_revision(IMAGE, allow_helper=True)
    assert system.started() == []


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
    assert "nothing started" in caplog.text


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
    ({"volumes": []}, "Qwen cache missing"),
    ({"revision": "c" * 40}, "!= expected"),
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
