#!/usr/bin/env python3
"""Gated command-line entry point for manual INFERSCOPE dev runs.

Wraps the existing shell scripts (run_experiment.sh, stack_up.sh, mlflow_ui.sh)
without changing them. Every action that can start the GPU stack prints a plan
and waits for the word "yes". Only the dev environment is supported.

Usage:
    python start_run.py --type {smoke,steady,burst} [--runs N] [--dry-run] [--no-mlflow]
    python start_run.py --preflight | --up | --down | --mlflow-ui
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import http.client
import json
import logging
import os
import re
import shutil
import subprocess  # nosec B404 - used only with list argv and no shell
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

log = logging.getLogger("start_run")

REPO_ROOT = Path(__file__).resolve().parents[2]

RUN_TYPES = ("smoke", "steady", "burst")
# The locked versions for the T5 runs (#31).
EXPECTED_MODEL_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
EXPECTED_WEIGHTS_SHA256 = (
    "dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee")
EXPECTED_VLLM_DIGEST = (
    "sha256:37cd5bd18d220a0f4c70401ce1d4a0cc588fbfe03cc210579428f2c47e6eac33")
SNAPSHOT_DIR = ("hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/"
                + EXPECTED_MODEL_REVISION)
WEIGHTS_FILE = "model.safetensors"
# Where compose mounts the cache volume in the vllm service.
CONTAINER_CACHE = "/root/.cache/huggingface"
REVISION_FLAGS = ("--revision", "--tokenizer-revision")
OFFLINE_SETTING = "HF_HUB_OFFLINE=1"
# AIPerf loads its tokenizer on the host, from the default Hugging Face cache.
HOST_MODEL_CACHE = (
    Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct")
TOKENIZER_FILES = (
    "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt")
# Any of these would make AIPerf load something other than what is checked.
CLIENT_CONFLICT_VARS = (
    "TOKENIZER", "HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE")
# The reader patch that aiperf_trace_replay.py loads into AIPerf.
HOOK_RELATIVE = Path("monitoring/scripts/aiperf_hooks/sitecustomize.py")
REQUIRED_AIPERF_VERSION = "0.11.0"
# Run by AIPerf's interpreter with the hook on PYTHONPATH.
HOOK_PROBE = (
    "import importlib.metadata as m; "
    "from aiperf.dataset import memory_map_utils as u; "
    "print(m.version('aiperf'), getattr("
    "u.MemoryMapDatasetClient.get_conversation, '_inferscope_patch', False))")
# run_experiment.sh activates this environment if aiperf is not on PATH.
DEFAULT_AIPERF_VENV = Path.home() / "venvs/aiperf-env"
OVERRIDE_NAME = "docker-compose.local.yml"
# stack_up.sh layers the override only when this variable names it.
OVERRIDE_ENV_VAR = "INFERSCOPE_COMPOSE_OVERRIDE"
HASH_TIMEOUT_S = 900
# Compose names the dev project after the monitoring/ directory.
DEV_PROJECT = "monitoring"
HF_VOLUME = f"{DEV_PROJECT}_hf_cache"
GRAFANA_VOLUME = f"{DEV_PROJECT}_grafana_data"
# mlflow_log_run.py uses this experiment for a dev stack.
DEV_EXPERIMENT = "inferscope-gp2"
# Port 5000, the mlflow_ui.sh default, belongs to the API (README, ports table).
MLFLOW_UI_PORT = "5001"
WATCHER_PATTERN = "docker events --filter container=vllm"
VLLM_HOST = "localhost"
VLLM_PORT = 8000


class RunError(RuntimeError):
    """A condition that stops the requested action, with a readable message."""


@dataclass(frozen=True)
class CheckResult:
    """Outcome of one preflight check."""

    name: str
    ok: bool
    detail: str


class Shell:
    """Runs external tools by full path, without a shell."""

    def which(self, tool: str) -> str:
        """Resolve a tool to its full path.

        Args:
            tool: Executable name.

        Returns:
            The absolute path of the executable.

        Raises:
            RunError: If the tool is not on PATH.
        """
        path = shutil.which(tool)
        if path is None:
            raise RunError(f"{tool} not found on PATH")
        return path

    def capture(
        self, tool: str, *args: str, timeout: int = 60,
        env: Optional[dict[str, str]] = None,
    ) -> subprocess.CompletedProcess:
        """Run a short command and capture its output.

        Args:
            tool: Executable name.
            *args: Arguments passed as a list.
            timeout: Seconds to wait.
            env: Environment to use instead of the current one.

        Returns:
            The completed process, whatever its exit code.

        Raises:
            RunError: If the command cannot be started or times out.
        """
        argv = [self.which(tool), *args]
        try:
            return subprocess.run(  # nosec B603 - list argv, no shell
                argv, capture_output=True, text=True, timeout=timeout,
                check=False, env=env,
            )
        except (subprocess.SubprocessError, OSError) as exc:
            raise RunError(f"could not run {tool}: {exc}") from exc

    def output(self, tool: str, *args: str) -> str:
        """Return a command's stdout.

        Raises:
            RunError: If the command exits non-zero or cannot run.
        """
        result = self.capture(tool, *args)
        if result.returncode != 0:
            raise RunError(
                f"{tool} {' '.join(args)} failed ({result.returncode}): "
                f"{result.stderr.strip()}")
        return result.stdout.strip()

    def stream(
        self,
        tool: str,
        *args: str,
        cwd: Path,
        env: Optional[dict[str, str]] = None,
    ) -> int:
        """Run a long command with its output going to the terminal.

        Args:
            tool: Executable name.
            *args: Arguments passed as a list.
            cwd: Working directory for the command.
            env: Environment to use instead of the current one.

        Returns:
            The command's exit code.

        Raises:
            RunError: If the command cannot be started.
        """
        argv = [self.which(tool), *args]
        try:
            return subprocess.run(  # nosec B603 - list argv, no shell
                argv, cwd=str(cwd), env=env, check=False,
            ).returncode
        except OSError as exc:
            raise RunError(f"could not run {tool}: {exc}") from exc


class StackInspector:
    """Facts about the repository, the .env file and Docker."""

    def __init__(self, shell: Shell, repo_root: Path) -> None:
        self.shell = shell
        self.repo_root = repo_root
        self.monitoring_dir = repo_root / "monitoring"
        self.override_path = self.monitoring_dir / OVERRIDE_NAME
        self.host_model_cache = HOST_MODEL_CACHE

    def env_value(self, key: str) -> str:
        """Read one variable from monitoring/.env.

        Args:
            key: Variable name.

        Returns:
            The value, which is empty if the variable is set to nothing.

        Raises:
            RunError: If the file or variable is missing, or defined twice.
                The message never contains a value.
        """
        path = self.monitoring_dir / ".env"
        if not path.is_file():
            raise RunError("monitoring/.env does not exist")
        found = [
            line[len(key) + 1:].strip()
            for line in path.read_text().splitlines()
            if line.startswith(f"{key}=")
        ]
        if not found:
            raise RunError(f"monitoring/.env has no {key} variable")
        if len(found) > 1:
            raise RunError(f"monitoring/.env defines {key} more than once")
        value = found[0]
        if value.startswith("#"):
            return ""
        return value.strip("\"'")

    def vllm_image(self) -> str:
        """Return the vLLM image that compose will use.

        Raises:
            RunError: If it is unset, floating, or overridden by the shell.
        """
        image = self.env_value("VLLM_IMAGE")
        if not image:
            raise RunError("VLLM_IMAGE is empty in monitoring/.env")
        if image.endswith(":latest"):
            raise RunError("VLLM_IMAGE must be pinned, not :latest")
        override = os.environ.get("VLLM_IMAGE")
        if override is not None and override != image:
            raise RunError(
                "VLLM_IMAGE in the shell differs from monitoring/.env; compose "
                "would use the shell value. Unset it first.")
        return image

    def require_grafana_password(self) -> None:
        """Raise RunError unless GRAFANA_ADMIN_PASSWORD has a value."""
        if not self.env_value("GRAFANA_ADMIN_PASSWORD"):
            raise RunError("GRAFANA_ADMIN_PASSWORD is empty in monitoring/.env")

    def image_digest(self, image: str) -> Optional[str]:
        """Return the local repo digest of an image, or None if not pulled.

        Raises:
            RunError: If Docker fails for another reason, or the image has
                no repo digest.
        """
        result = self.shell.capture(
            "docker", "image", "inspect", image,
            "--format", "{{json .RepoDigests}}")
        if result.returncode != 0:
            if "no such image" in result.stderr.lower():
                return None
            raise RunError(f"docker image inspect failed: {result.stderr.strip()}")
        digests = json.loads(result.stdout)
        if not digests:
            raise RunError(f"image {image} has no repo digest")
        return digests[0]

    def running_vllm_image(self) -> Optional[str]:
        """Return the image of the running vllm container, or None."""
        listing = self.shell.output("docker", "ps", "--format", "{{.Names}}|{{.Image}}")
        for line in listing.splitlines():
            name, _, image = line.partition("|")
            if name == "vllm":
                return image
        return None

    def volume_exists(self, name: str) -> bool:
        """Return True if a Docker volume with exactly this name exists."""
        listing = self.shell.output("docker", "volume", "ls", "--format", "{{.Name}}")
        return name in listing.splitlines()

    def api_answering(self) -> bool:
        """Return True if /v1/models answers, as run_experiment.sh checks it."""
        connection = http.client.HTTPConnection(VLLM_HOST, VLLM_PORT, timeout=5)
        try:
            connection.request("GET", "/v1/models")
            return connection.getresponse().status < 400
        except (OSError, http.client.HTTPException):
            return False
        finally:
            connection.close()

    def stack_state(self) -> dict:
        """Load the state file written by stack_up.sh.

        Raises:
            RunError: If the file is missing or not valid JSON.
        """
        path = self.monitoring_dir / "logs" / "stack_state.json"
        try:
            return json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise RunError(
                f"the stack is up but {path.name} cannot be read: {exc}") from exc

    def override_sha256(self) -> str:
        """Return the override file's SHA-256.

        Raises:
            RunError: If the override does not exist.
        """
        if not self.override_path.is_file():
            raise RunError(f"override missing: monitoring/{OVERRIDE_NAME}")
        return hashlib.sha256(self.override_path.read_bytes()).hexdigest()

    def compose_model(self, with_override: bool) -> dict:
        """Return the compose model without substituting any variable.

        --no-interpolate keeps secrets as literal placeholders, so none is read.

        Args:
            with_override: Whether to layer the local override.

        Raises:
            RunError: If compose cannot parse the files.
        """
        files = ["-f", str(self.monitoring_dir / "docker-compose.yml")]
        if with_override:
            files += ["-f", str(self.override_path)]
        text = self.shell.output(
            "docker", "compose", *files, "--profile", "local",
            "config", "--no-interpolate", "--format", "json")
        try:
            model = json.loads(text)
            model["services"]["vllm"]["command"]
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise RunError(f"unexpected compose config output: {exc!r}") from exc
        return model

    def version_lock(self) -> str:
        """Check that the override pins the revision and forces offline mode.

        Returns:
            A short description of the lock.

        Raises:
            RunError: If the override is missing or does not lock all three.
        """
        digest = self.override_sha256()
        vllm = self.compose_model(with_override=True)["services"]["vllm"]
        command = vllm["command"]
        for flag in REVISION_FLAGS:
            values = [command[i + 1] for i, item in enumerate(command[:-1])
                      if item == flag]
            if values != [EXPECTED_MODEL_REVISION]:
                raise RunError(
                    f"override must set {flag} {EXPECTED_MODEL_REVISION} "
                    f"exactly once (found {values or 'none'})")
        offline = [item for item in vllm.get("environment") or []
                   if item.startswith("HF_HUB_OFFLINE=")]
        if offline != [OFFLINE_SETTING]:
            raise RunError(f"override must set {OFFLINE_SETTING}")
        return f"revision and offline mode locked, sha256 {digest}"

    def compose_drift(self) -> str:
        """Check that the override changes only what it is meant to change.

        The merged vLLM command, without the two added flags, must equal the
        command in docker-compose.yml. The only other change allowed is the
        added HF_HUB_OFFLINE setting.

        Raises:
            RunError: With a diff, if anything else differs.
        """
        self.override_sha256()
        base = self.compose_model(with_override=False)
        merged = self.compose_model(with_override=True)
        base_vllm = base["services"]["vllm"]
        merged_vllm = merged["services"]["vllm"]

        stripped: list[str] = []
        items = iter(merged_vllm["command"])
        for item in items:
            if item in REVISION_FLAGS:
                next(items, None)
            else:
                stripped.append(item)
        if stripped != base_vllm["command"]:
            diff = "\n".join(difflib.unified_diff(
                base_vllm["command"], stripped,
                "docker-compose.yml", OVERRIDE_NAME, lineterm="", n=2))
            raise RunError(
                "the override's vLLM command differs from docker-compose.yml:\n"
                + diff)

        base_env = base_vllm.get("environment") or []
        merged_env = merged_vllm.get("environment") or []
        if sorted(merged_env) != sorted(set(base_env) | {OFFLINE_SETTING}):
            raise RunError(
                "the override changes the vLLM environment beyond "
                f"{OFFLINE_SETTING}: {sorted(set(merged_env) ^ set(base_env))}")

        for model in (base, merged):
            for key in ("command", "environment"):
                model["services"]["vllm"].pop(key, None)
        if base != merged:
            changed = sorted(
                name for name in set(base["services"]) | set(merged["services"])
                if base["services"].get(name) != merged["services"].get(name))
            raise RunError(
                "the override changes more than the vLLM command and "
                f"environment (services: {changed or 'none; top-level keys'})")
        return "base command and all other settings match docker-compose.yml"

    def read_model_cache(
        self, image: str, program: str, args: Sequence[str], path: str,
        timeout: int = 60,
    ) -> subprocess.CompletedProcess:
        """Run a read-only program against a path in the model cache.

        Uses the running vllm container if there is one. Otherwise mounts the
        cache volume read-only in a short-lived container made from the vLLM
        image itself, with no network and no pull.

        Args:
            image: The configured vLLM image.
            program: Program to run.
            args: Its arguments, before the path.
            path: Path relative to the cache root.
            timeout: Seconds to wait.

        Returns:
            The completed process.

        Raises:
            RunError: If the cache volume or the image is not there.
        """
        if self.running_vllm_image() is not None:
            command = ["exec", "vllm", program, *args,
                       f"{CONTAINER_CACHE}/{path}"]
        else:
            # Checked first: "docker run -v" would create a missing volume.
            if not self.volume_exists(HF_VOLUME):
                raise RunError(f"model cache missing: no volume {HF_VOLUME}")
            if self.image_digest(image) is None:
                raise RunError("model cache not read: the vLLM image is not pulled")
            command = ["run", "--rm", "--pull", "never", "--network", "none",
                       "--entrypoint", program, "-v", f"{HF_VOLUME}:/hf:ro",
                       image, *args, f"/hf/{path}"]
        return self.shell.capture("docker", *command, timeout=timeout)

    def model_snapshot(self) -> str:
        """Check that the locked snapshot directory is in the cache volume."""
        result = self.read_model_cache(
            self.vllm_image(), "test", ["-d"], SNAPSHOT_DIR)
        if result.returncode != 0:
            raise RunError(
                f"snapshot {EXPECTED_MODEL_REVISION} missing from {HF_VOLUME} "
                f"{result.stderr.strip()}".strip())
        return f"snapshot {EXPECTED_MODEL_REVISION} present"

    def model_weights(self) -> str:
        """Hash the weights and compare them with the locked SHA-256."""
        image = self.vllm_image()
        started = time.monotonic()
        result = self.read_model_cache(
            image, "sha256sum", [], f"{SNAPSHOT_DIR}/{WEIGHTS_FILE}",
            timeout=HASH_TIMEOUT_S)
        elapsed = time.monotonic() - started
        if result.returncode != 0:
            raise RunError(f"cannot hash {WEIGHTS_FILE}: {result.stderr.strip()}")
        found = (result.stdout.split() or [""])[0]
        if found != EXPECTED_WEIGHTS_SHA256:
            raise RunError(
                f"{WEIGHTS_FILE} sha256 {found!r} != expected "
                f"{EXPECTED_WEIGHTS_SHA256}")
        return f"{WEIGHTS_FILE} sha256 matches (hashed in {elapsed:.1f} s)"

    def client_tokenizer(self) -> str:
        """Check the tokenizer AIPerf will load on the host.

        The host's refs/main must name the locked revision, and its tokenizer
        files must be byte-identical to those in the model cache volume.

        Returns:
            A short description of what matched.

        Raises:
            RunError: Listing every problem found.
        """
        conflicts = [name for name in CLIENT_CONFLICT_VARS if os.environ.get(name)]
        if conflicts:
            raise RunError(
                f"unset {', '.join(conflicts)}: it would change which tokenizer "
                "AIPerf loads")
        problems: list[str] = []
        refs = self.host_model_cache / "refs" / "main"
        if not refs.is_file():
            problems.append("host refs/main is missing")
        else:
            found = refs.read_text().strip()
            if found != EXPECTED_MODEL_REVISION:
                problems.append(
                    f"host refs/main is {found}, expected {EXPECTED_MODEL_REVISION}")

        image = self.vllm_image()
        snapshot = self.host_model_cache / "snapshots" / EXPECTED_MODEL_REVISION
        for name in TOKENIZER_FILES:
            local = snapshot / name
            if not local.is_file():
                problems.append(f"host file missing: {name}")
                continue
            result = self.read_model_cache(
                image, "sha256sum", [], f"{SNAPSHOT_DIR}/{name}")
            if result.returncode != 0:
                problems.append(
                    f"volume file unreadable: {name} {result.stderr.strip()}")
                continue
            in_volume = (result.stdout.split() or [""])[0]
            on_host = hashlib.sha256(local.read_bytes()).hexdigest()
            if on_host != in_volume:
                problems.append(
                    f"{name} differs: host {on_host}, volume {in_volume or 'none'}")
        if problems:
            raise RunError("; ".join(problems))
        return (f"host refs/main is {EXPECTED_MODEL_REVISION}; "
                f"{len(TOKENIZER_FILES)} tokenizer files match the volume")

    def aiperf_python(self) -> str:
        """Return the interpreter of the aiperf that a run would start.

        Raises:
            RunError: If aiperf or its interpreter cannot be determined.
        """
        executable = shutil.which("aiperf")
        if executable is None:
            venv = Path(os.environ.get("AIPERF_VENV") or DEFAULT_AIPERF_VENV)
            executable = str(venv / "bin" / "aiperf")
        try:
            with open(executable, "rb") as handle:
                first = handle.readline().decode("utf-8", errors="replace").strip()
        except OSError as exc:
            raise RunError(f"aiperf not found at {executable}: {exc}") from exc
        if not first.startswith("#!") or "python" not in first:
            raise RunError(
                f"cannot tell which Python runs {executable} (first line {first!r})")
        return first[2:].strip()

    def aiperf_hook(self) -> str:
        """Check that the reader patch loads into AIPerf's own interpreter.

        Raises:
            RunError: If the hook is missing, refuses to load, or AIPerf is not
                the version it is written for.
        """
        hook = self.repo_root / HOOK_RELATIVE
        if not hook.is_file():
            raise RunError(f"reader patch missing: {HOOK_RELATIVE}")
        python = self.aiperf_python()
        with tempfile.TemporaryDirectory() as scratch:
            env = {**os.environ, "PYTHONPATH": str(hook.parent),
                   "AIPERF_PATCH_STATUS_FILE": str(Path(scratch) / "status.jsonl"),
                   "PYTHONDONTWRITEBYTECODE": "1"}
            result = self.shell.capture(
                python, "-c", HOOK_PROBE, env=env, timeout=120)
        found = result.stdout.split()
        if result.returncode != 0 or found != [REQUIRED_AIPERF_VERSION, "True"]:
            raise RunError(
                f"reader patch did not load into {python} (exit "
                f"{result.returncode}, reported {found or 'nothing'}) "
                f"{result.stderr.strip()[-300:]}".strip())
        digest = hashlib.sha256(hook.read_bytes()).hexdigest()
        return (f"aiperf {REQUIRED_AIPERF_VERSION}; reader patch loads, "
                f"sha256 {digest}")

    def running_stack(self) -> str:
        """Check that a vllm container that is already running is locked.

        run_experiment.sh reuses a stack that is up, so the override file
        alone says nothing about that container.

        Raises:
            RunError: If it was not started with the current override.
        """
        if self.running_vllm_image() is None:
            return "no vllm container running"
        text = self.shell.output(
            "docker", "inspect", "vllm", "--format", "{{json .Config}}")
        config = json.loads(text)
        command = config.get("Cmd") or []
        problems = [
            f"{flag} is not {EXPECTED_MODEL_REVISION}"
            for flag in REVISION_FLAGS
            if not any(command[i] == flag and command[i + 1] == EXPECTED_MODEL_REVISION
                       for i in range(len(command) - 1))
        ]
        if OFFLINE_SETTING not in (config.get("Env") or []):
            problems.append(f"{OFFLINE_SETTING} is not set")
        if not problems:
            state = self.stack_state()
            if state.get("override_sha256") != self.override_sha256():
                problems.append(
                    "the override hash recorded at start differs from the file")
        if problems:
            raise RunError(
                "the running vllm container is not locked: "
                + "; ".join(problems) + ". Run --down, then start again")
        return "running container is locked with the current override"

    def git_sha(self) -> str:
        """Return the full commit SHA of the checkout."""
        return self.shell.output("git", "-C", str(self.repo_root), "rev-parse", "HEAD")

    def git_dirty(self) -> bool:
        """Return True if there are uncommitted or untracked changes."""
        return bool(self.shell.output(
            "git", "-C", str(self.repo_root), "status", "--porcelain"))

    def gpu_name(self) -> str:
        """Return the name of the GPU reported by nvidia-smi."""
        name = self.shell.output(
            "nvidia-smi", "--query-gpu=name", "--format=csv,noheader")
        if not name:
            raise RunError("nvidia-smi reports no GPU")
        return name


class RunController:
    """Implements each command-line action."""

    def __init__(
        self,
        shell: Shell,
        inspector: StackInspector,
        ask: Callable[[str], str] = input,
    ) -> None:
        self.shell = shell
        self.inspector = inspector
        self.ask = ask

    def confirm(self) -> bool:
        """Return True only if the user types exactly "yes"."""
        try:
            return self.ask('Type "yes" to continue: ') == "yes"
        except EOFError:
            return False

    def experiment_name(self, image: str) -> str:
        """Return the MLflow experiment the logger will write to.

        Args:
            image: The configured vLLM image.

        Raises:
            RunError: If a stack is already up that this tool cannot account
                for: unknown, demo, or running a different image.
        """
        running = self.inspector.running_vllm_image()
        if running is not None and running != image:
            raise RunError(
                f"the running vllm container uses {running}, but "
                f"monitoring/.env says {image}")
        if not self.inspector.api_answering():
            return DEV_EXPERIMENT
        if running is None:
            raise RunError(
                f"port {VLLM_PORT} answers but no vllm container is running")
        state = self.inspector.stack_state()
        if state.get("stack_env") != "dev":
            raise RunError(
                f"the running stack records stack_env="
                f"{state.get('stack_env')!r}; only dev is supported")
        if state.get("vllm_image") != image:
            raise RunError(
                "stack_state.json records a different vLLM image than "
                "monitoring/.env, so MLflow would log the wrong one")
        return DEV_EXPERIMENT

    def build_plan(
        self,
        action: str,
        runs: Optional[int],
        use_mlflow: bool,
        dry_run: bool,
    ) -> list[tuple[str, str]]:
        """Collect the facts shown before the gate.

        A dry run reports facts it cannot read as unavailable. A real action
        raises instead.

        Args:
            action: A run type, or "up".
            runs: Number of runs, or None for --up.
            use_mlflow: Whether the runs are logged to MLflow.
            dry_run: Whether this is a preview only.

        Returns:
            Ordered (label, value) pairs.

        Raises:
            RunError: If a required fact cannot be established.
        """
        image = self.inspector.vllm_image()
        experiment = self.experiment_name(image)

        digest = self.inspector.image_digest(image)
        if digest is None:
            if not dry_run:
                raise RunError(
                    f"vLLM image {image} is not pulled; this tool does not "
                    "start anything that would pull it")
            digest = "unavailable: image not pulled"
        elif not digest.endswith("@" + EXPECTED_VLLM_DIGEST):
            digest += f"  MISMATCH, expected {EXPECTED_VLLM_DIGEST}"

        try:
            self.inspector.version_lock()
            self.inspector.compose_drift()
            lock = [
                ("override", f"monitoring/{OVERRIDE_NAME}"),
                ("override sha256", self.inspector.override_sha256()),
                ("model revision", EXPECTED_MODEL_REVISION),
                ("offline", OFFLINE_SETTING),
                ("client tokenizer",
                 f"{OFFLINE_SETTING}, revision {EXPECTED_MODEL_REVISION}"),
            ]
        except RunError as exc:
            if not dry_run:
                raise
            lock = [("version lock", f"unavailable: {exc}")]

        plan = [
            ("vLLM image", image),
            ("vLLM digest", digest),
            *lock,
            ("model files",
             "not verified: a dry run starts no container" if dry_run
             else "snapshot and weights verified by preflight"),
            ("git SHA", self.inspector.git_sha()),
            ("tree dirty", "yes" if self.inspector.git_dirty() else "no"),
            ("type", action),
        ]
        if runs is not None:
            plan += [
                ("runs", str(runs)),
                ("max concurrency", os.environ.get("MAX_CONCURRENCY") or "20"),
                ("MLflow experiment", experiment if use_mlflow else "disabled"),
            ]
        plan.append((
            "note", "starting the stack pulls any other compose image that is "
                    "missing and builds nvml-exporter"))
        return plan

    def locked_env(self) -> dict[str, str]:
        """Environment that locks both the server and the client.

        stack_up.sh loads the local override. The replay, and so AIPerf, runs
        offline with the locked tokenizer revision. run_experiment.sh passes
        both on to the scripts it calls.
        """
        return {**os.environ,
                OVERRIDE_ENV_VAR: str(self.inspector.override_path),
                "HF_HUB_OFFLINE": "1",
                "TOKENIZER_REVISION": EXPECTED_MODEL_REVISION}

    @staticmethod
    def show_plan(plan: Sequence[tuple[str, str]]) -> None:
        """Log the plan, one fact per line."""
        log.info("plan:")
        for label, value in plan:
            log.info("  %-18s %s", label, value)

    def require_preflight(self) -> None:
        """Block the action unless every preflight check passes.

        There is no bypass, and the gate cannot override a failure.

        Raises:
            RunError: Naming each failed check.
        """
        failed = [r for r in self.preflight_checks() if not r.ok]
        for result in failed:
            log.error("FAIL  %-16s %s", result.name, result.detail)
        if failed:
            raise RunError(
                "blocked by preflight: "
                + ", ".join(result.name for result in failed))

    def run(self, run_type: str, runs: int, use_mlflow: bool, dry_run: bool) -> int:
        """Pass preflight and the gate, then call run_experiment.sh.

        Preflight is repeated before each later run of the same invocation.

        Returns:
            0 on success or after a dry run, 1 if the gate is refused.

        Raises:
            RunError: If preflight fails, or a run exits non-zero; later runs
                are not started.
        """
        if not dry_run:
            self.require_preflight()
        self.show_plan(self.build_plan(run_type, runs, use_mlflow, dry_run))
        if dry_run:
            log.info("dry run: nothing started")
            return 0
        if not self.confirm():
            log.error("not confirmed; nothing started")
            return 1
        script = self.inspector.monitoring_dir / "run_experiment.sh"
        env = self.locked_env()
        extra = () if use_mlflow else ("--no-mlflow",)
        for number in range(1, runs + 1):
            if number > 1:
                self.require_preflight()
            log.info("run %d of %d (%s)", number, runs, run_type)
            code = self.shell.stream(
                "bash", str(script), run_type, *extra,
                cwd=self.inspector.monitoring_dir, env=env)
            if code != 0:
                raise RunError(
                    f"run {number} of {runs} exited with code {code}; "
                    "remaining runs not started")
        return 0

    def up(self) -> int:
        """Pass preflight and the gate, then start the dev stack."""
        self.require_preflight()
        self.show_plan(self.build_plan("up", None, False, False))
        if not self.confirm():
            log.error("not confirmed; nothing started")
            return 1
        script = self.inspector.monitoring_dir / "stack_up.sh"
        code = self.shell.stream(
            "bash", str(script), cwd=self.inspector.monitoring_dir,
            env=self.locked_env())
        if code != 0:
            raise RunError(f"stack_up.sh exited with code {code}")
        return 0

    def down(self) -> int:
        """Stop the dev stack, keeping its volumes.

        Compose is addressed by project name from an empty directory. Reading
        the compose file would fail when GRAFANA_ADMIN_PASSWORD is missing.

        Returns:
            0 on success.

        Raises:
            RunError: If compose or pkill fails.
        """
        env = {k: v for k, v in os.environ.items() if k != "COMPOSE_FILE"}
        with tempfile.TemporaryDirectory() as empty_dir:
            code = self.shell.stream(
                "docker", "compose", "-p", DEV_PROJECT, "down",
                cwd=Path(empty_dir), env=env)
        if code != 0:
            raise RunError(f"docker compose down exited with code {code}")
        watcher = self.shell.capture("pkill", "-f", WATCHER_PATTERN)
        # pkill exits 1 when nothing matched.
        if watcher.returncode not in (0, 1):
            raise RunError(f"pkill failed: {watcher.stderr.strip()}")
        return 0

    def mlflow_ui(self) -> int:
        """Open the MLflow UI through mlflow_ui.sh."""
        script = self.inspector.monitoring_dir / "mlflow_ui.sh"
        return self.shell.stream(
            "bash", str(script), MLFLOW_UI_PORT,
            cwd=self.inspector.monitoring_dir)

    def preflight_checks(self) -> list[CheckResult]:
        """Run every check, so one failure does not hide the others."""
        inspector = self.inspector
        results: list[CheckResult] = []

        def check(name: str, probe: Callable[[], str]) -> None:
            try:
                results.append(CheckResult(name, True, probe()))
            except RunError as exc:
                results.append(CheckResult(name, False, str(exc)))

        def clean_tree() -> str:
            if inspector.git_dirty():
                raise RunError("the working tree is dirty (see git status)")
            return f"clean at {inspector.git_sha()}"

        def password() -> str:
            inspector.require_grafana_password()
            return "set (value not shown)"

        def pulled() -> str:
            image = inspector.vllm_image()
            digest = inspector.image_digest(image)
            if digest is None:
                raise RunError(f"vLLM image not pulled: {image}")
            return digest

        def locked_digest() -> str:
            digest = pulled()
            if not digest.endswith("@" + EXPECTED_VLLM_DIGEST):
                raise RunError(
                    f"vLLM image digest {digest} != expected {EXPECTED_VLLM_DIGEST}")
            return EXPECTED_VLLM_DIGEST

        check("git tree", clean_tree)
        check("GPU", inspector.gpu_name)
        check("vLLM variable", inspector.vllm_image)
        check("Grafana password", password)
        check("vLLM image", pulled)
        check("vLLM digest", locked_digest)
        check("version lock", inspector.version_lock)
        check("compose drift", inspector.compose_drift)
        check("model snapshot", inspector.model_snapshot)
        check("model weights", inspector.model_weights)
        check("client tokenizer", inspector.client_tokenizer)
        check("AIPerf patch", inspector.aiperf_hook)
        check("running stack", inspector.running_stack)
        return results

    def preflight(self) -> int:
        """Report each check and whether the Grafana volume exists.

        Returns:
            0 if every check passes, otherwise 1.
        """
        results = self.preflight_checks()
        for result in results:
            log.log(
                logging.INFO if result.ok else logging.ERROR,
                "%s  %-16s %s",
                "PASS" if result.ok else "FAIL", result.name, result.detail)
        try:
            exists = self.inspector.volume_exists(GRAFANA_VOLUME)
        except RunError as exc:
            log.error("FAIL  %-16s %s", "Grafana volume", exc)
            return 1
        log.info(
            "INFO  %-16s %s %s", "Grafana volume", GRAFANA_VOLUME,
            "already exists: it keeps its old admin password"
            if exists else
            "does not exist: the .env password applies on first start")
        failed = sum(not result.ok for result in results)
        if failed:
            log.error("preflight failed: %d check(s)", failed)
            return 1
        log.info("preflight passed")
        return 0


def positive_int(text: str) -> int:
    """Argparse type for a whole number of at least 1."""
    try:
        value = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"not a whole number: {text!r}") from exc
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse the command line; exactly one action is required."""
    parser = argparse.ArgumentParser(
        description="Gated entry point for manual INFERSCOPE dev runs.")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--type", choices=RUN_TYPES, help="run this workload")
    action.add_argument("--preflight", action="store_true", help="check readiness")
    action.add_argument("--up", action="store_true", help="start the dev stack")
    action.add_argument("--down", action="store_true",
                        help="stop the dev stack, keeping volumes")
    action.add_argument("--mlflow-ui", action="store_true",
                        help=f"MLflow UI on port {MLFLOW_UI_PORT}")
    parser.add_argument("--runs", type=positive_int, help="with --type (default 1)")
    parser.add_argument("--dry-run", action="store_true",
                        help="with --type: print the plan only")
    parser.add_argument("--no-mlflow", action="store_true",
                        help="with --type: skip MLflow logging")
    args = parser.parse_args(argv)
    if args.type is None and (args.runs is not None or args.dry_run or args.no_mlflow):
        parser.error("--runs, --dry-run and --no-mlflow require --type")
    if args.runs is None:
        args.runs = 1
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the requested action and return the process exit code."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args(argv)
    shell = Shell()
    controller = RunController(shell, StackInspector(shell, REPO_ROOT))
    try:
        stack_env = os.environ.get("STACK_ENV", "dev")
        if stack_env != "dev":
            raise RunError(f"STACK_ENV={stack_env!r}: only dev is supported")
        if args.preflight:
            return controller.preflight()
        if args.up:
            return controller.up()
        if args.down:
            return controller.down()
        if args.mlflow_ui:
            return controller.mlflow_ui()
        return controller.run(
            args.type, args.runs, not args.no_mlflow, args.dry_run)
    except RunError as exc:
        log.error("%s", exc)
        return 1
    except KeyboardInterrupt:
        log.error("interrupted")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
