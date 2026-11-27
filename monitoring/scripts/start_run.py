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
import http.client
import json
import logging
import os
import re
import shutil
import subprocess  # nosec B404 - used only with list argv and no shell
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

log = logging.getLogger("start_run")

REPO_ROOT = Path(__file__).resolve().parents[2]

RUN_TYPES = ("smoke", "steady", "burst")
EXPECTED_QWEN_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
QWEN_REFS_FILE = "hub/models--Qwen--Qwen2.5-1.5B-Instruct/refs/main"
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

    def capture(self, tool: str, *args: str) -> subprocess.CompletedProcess:
        """Run a short command and capture its output.

        Args:
            tool: Executable name.
            *args: Arguments passed as a list.

        Returns:
            The completed process, whatever its exit code.

        Raises:
            RunError: If the command cannot be started or times out.
        """
        argv = [self.which(tool), *args]
        try:
            return subprocess.run(  # nosec B603 - list argv, no shell
                argv, capture_output=True, text=True, timeout=60, check=False,
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

    def qwen_revision(self, image: str, allow_helper: bool) -> str:
        """Read the cached Qwen revision that the vLLM container will see.

        Uses the running vllm container if there is one. Otherwise mounts the
        cache volume read-only in a short-lived container made from the vLLM
        image itself, with no network and no pull.

        Args:
            image: The configured vLLM image.
            allow_helper: False forbids starting the helper container.

        Returns:
            The 40-character revision in refs/main.

        Raises:
            RunError: If the revision cannot be read.
        """
        if self.running_vllm_image() is not None:
            args = ("exec", "vllm", "cat",
                    f"/root/.cache/huggingface/{QWEN_REFS_FILE}")
        else:
            # Checked first: "docker run -v" would create a missing volume.
            if not self.volume_exists(HF_VOLUME):
                raise RunError(f"Qwen cache missing: no volume {HF_VOLUME}")
            if not allow_helper:
                raise RunError("not read: a dry run starts no container")
            if self.image_digest(image) is None:
                raise RunError("Qwen cache not read: the vLLM image is not pulled")
            args = ("run", "--rm", "--pull", "never", "--network", "none",
                    "--entrypoint", "cat", "-v", f"{HF_VOLUME}:/hf:ro",
                    image, f"/hf/{QWEN_REFS_FILE}")
        result = self.shell.capture("docker", *args)
        if result.returncode != 0:
            raise RunError(f"Qwen cache missing: {result.stderr.strip()}")
        revision = result.stdout.strip()
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise RunError("Qwen refs/main does not hold a 40-character commit ID")
        return revision

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

        try:
            revision = self.inspector.qwen_revision(image, allow_helper=not dry_run)
            if revision != EXPECTED_QWEN_REVISION:
                revision += f"  MISMATCH, expected {EXPECTED_QWEN_REVISION}"
        except RunError as exc:
            if not dry_run:
                raise
            revision = f"unavailable: {exc}"

        plan = [
            ("vLLM image", image),
            ("vLLM digest", digest),
            ("Qwen revision", revision),
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
        extra = () if use_mlflow else ("--no-mlflow",)
        for number in range(1, runs + 1):
            if number > 1:
                self.require_preflight()
            log.info("run %d of %d (%s)", number, runs, run_type)
            code = self.shell.stream(
                "bash", str(script), run_type, *extra,
                cwd=self.inspector.monitoring_dir)
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
            "bash", str(script), cwd=self.inspector.monitoring_dir)
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

        def cache() -> str:
            return inspector.qwen_revision(inspector.vllm_image(), allow_helper=True)

        def revision() -> str:
            found = cache()
            if found != EXPECTED_QWEN_REVISION:
                raise RunError(
                    f"Qwen revision {found} != expected {EXPECTED_QWEN_REVISION}")
            return found

        check("git tree", clean_tree)
        check("GPU", inspector.gpu_name)
        check("vLLM variable", inspector.vllm_image)
        check("Grafana password", password)
        check("vLLM image", pulled)
        check("Qwen cache", cache)
        check("Qwen revision", revision)
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
