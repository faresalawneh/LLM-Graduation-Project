#!/usr/bin/env python3
"""Log a completed INFERSCOPE replay run to MLflow.

Reads artifacts written by aiperf_trace_replay.py and registers them as an
MLflow run. Runs strictly after the benchmark completes, so the measurement
path stays untouched. Safe to backfill on historical artifact directories.

Usage:
    python mlflow_log_run.py <artifact_dir> [--stack-state PATH] [--experiment NAME]
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import shutil
import subprocess  # nosec B404 - used only with list argv and no shell
from pathlib import Path
from typing import Any

try:
    import mlflow
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "mlflow is not installed. Run:\n"
        "    pip install -r requirements-experiments.txt"
    ) from exc

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("mlflow_log_run")

_UNSAFE = re.compile(r"[^0-9a-zA-Z_\-. :/]+")

# Every file in the run directory is logged, so each MLflow run is
# self-contained. mlruns/ is gitignored, so this never reaches the repo.
# Add filenames here to exclude them.
SKIP_ARTIFACTS: frozenset[str] = frozenset()



def sanitize(name: str) -> str:
    """Convert an AIPerf metric label into an MLflow-safe key."""
    cleaned = _UNSAFE.sub("_", name).strip("_ ")
    return re.sub(r"[ _]+", "_", cleaned)


def read_json(path: Path) -> dict[str, Any]:
    """Load JSON, returning an empty dict if the file is absent or malformed."""
    try:
        with path.open() as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("could not read %s: %s", path.name, exc)
        return {}


def parse_aiperf_csv(path: Path) -> dict[str, float]:
    """Flatten the AIPerf summary CSV into {metric_stat: value}.

    Keeps avg/min/max/p50/p95/p99 per metric; drops the rest to limit cardinality.
    """
    keep = {"avg", "min", "max", "p50", "p95", "p99"}
    metrics: dict[str, float] = {}
    if not path.exists():
        log.warning("missing %s", path.name)
        return metrics
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            label = row.get("Metric")
            if not label:
                continue
            base = sanitize(label)
            for stat in keep:
                raw = row.get(stat)
                if raw in (None, "", "N/A"):
                    continue
                try:
                    metrics[f"{base}.{stat}"] = float(raw)
                except ValueError:
                    continue
    return metrics


_TOOL_PATHS: dict[str, str] = {}


def _require(tool: str) -> str:
    """Resolve a tool to its full path once, so no partial path is executed."""
    path = _TOOL_PATHS.get(tool)
    if path is None:
        path = shutil.which(tool)
        if path is None:
            raise RuntimeError(f"{tool} not found on PATH")
        _TOOL_PATHS[tool] = path
    return path


def git_sha(repo_root: Path) -> str:
    """Return the current commit SHA, or 'unknown' outside a git tree."""
    try:
        out = subprocess.run(
            [_require("git"), "-C", str(repo_root), "rev-parse", "--short", "HEAD"],  # nosec B603 - fixed argv, no shell, no user input
            capture_output=True, text=True, timeout=10, check=True,
        )
        return out.stdout.strip()
    except (subprocess.SubprocessError, OSError, RuntimeError):
        return "unknown"


def docker_state(container: str = "vllm") -> dict[str, Any]:
    """Return current restart count / exit code / OOM flag for the container."""
    fmt = "{{.RestartCount}}|{{.State.ExitCode}}|{{.State.OOMKilled}}|{{.State.Status}}"
    try:
        out = subprocess.run(
            [_require("docker"), "inspect", container, "--format", fmt],  # nosec B603 - fixed argv, no shell, no user input
            capture_output=True, text=True, timeout=15, check=True,
        )
        restarts, exit_code, oom, status = out.stdout.strip().split("|")
        return {
            "restart_count": int(restarts),
            "exit_code": int(exit_code),
            "oom_killed": oom.lower() == "true",
            "status": status,
        }
    except (subprocess.SubprocessError, OSError, ValueError, RuntimeError) as exc:
        log.warning("docker inspect unavailable: %s", exc)
        return {}


def count_jsonl(path: Path) -> int:
    """Count non-blank lines in a JSONL file; 0 if absent."""
    if not path.exists():
        return 0
    with path.open() as handle:
        return sum(1 for line in handle if line.strip())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact_dir", type=Path)
    parser.add_argument("--stack-state", type=Path, default=None)
    parser.add_argument(
        "--experiment",
        default=None,
        help="MLflow experiment. Default: 'inferscope-gp2-demo' when the "
             "stack state says stack_env=demo, else 'inferscope-gp2'.",
    )
    parser.add_argument(
        "--run-type",
        default=None,
        help="Tag the run (e.g. smoke, historical). Auto-set to 'smoke' "
             "when trace_row_count < 1000.",
    )
    parser.add_argument("--tracking-uri", default=None)
    args = parser.parse_args()

    run_dir: Path = args.artifact_dir.expanduser().resolve()
    if not run_dir.is_dir():
        log.error("not a directory: %s", run_dir)
        return 2

    repo_root = Path(__file__).resolve().parents[2]
    mlruns = repo_root / "mlruns"
    mlruns.mkdir(exist_ok=True)
    stack_path = args.stack_state or (repo_root / "monitoring/logs/stack_state.json")
    stack = read_json(stack_path)
    stack_env = stack.get("stack_env") or "unknown"
    # Demo (assessed) runs go to their own experiment so they never mix with
    # dev runs; dev and stacks without a recorded env keep the old default.
    experiment = args.experiment or (
        "inferscope-gp2-demo" if stack_env == "demo" else "inferscope-gp2")

    tracking = args.tracking_uri or f"sqlite:///{mlruns / 'mlflow.db'}"
    mlflow.set_tracking_uri(tracking)
    if mlflow.get_experiment_by_name(experiment) is None:
        mlflow.create_experiment(
            experiment,
            artifact_location=f"file://{mlruns / 'artifacts'}",
        )
    mlflow.set_experiment(experiment)
    log.info("stack_env=%s -> experiment %s", stack_env, experiment)

    manifest = read_json(run_dir / "run_manifest.json")
    state = docker_state()

    scenario = manifest.get("scenario", "unknown")
    with mlflow.start_run(run_name=f"{scenario}-{run_dir.name}"):
        params = {
            "scenario": scenario,
            "model": manifest.get("model"),
            "endpoint_type": manifest.get("endpoint_type"),
            "trace_file": manifest.get("trace_file"),
            "trace_row_count": manifest.get("trace_row_count"),
            "slice_duration": manifest.get("slice_duration"),
            "enable_gpu_scrape": manifest.get("enable_gpu_scrape"),
            "start_time": manifest.get("start_time"),
            "artifact_dir": str(run_dir),
            "vllm_image": stack.get("vllm_image"),
            "vllm_digest": stack.get("vllm_digest"),
            "git_sha": git_sha(repo_root),
            "stack_env": stack_env,
        }
        mlflow.log_params({k: v for k, v in params.items() if v is not None})

        metrics = parse_aiperf_csv(run_dir / "profile_export_aiperf.csv")
        metrics["gpu_telemetry_samples"] = float(
            count_jsonl(run_dir / "gpu_telemetry_export.jsonl"))
        metrics["server_metric_scrapes"] = float(
            count_jsonl(run_dir / "server_metrics_export.jsonl"))

        baseline = stack.get("restart_count_at_start")
        if state:
            metrics["vllm_restart_count"] = float(state["restart_count"])
            metrics["vllm_oom_killed"] = float(state["oom_killed"])
            if baseline is not None:
                metrics["vllm_restarts_during_run"] = float(
                    state["restart_count"] - baseline)
        if metrics:
            mlflow.log_metrics(metrics)

        logged_bytes = 0
        for candidate in sorted(run_dir.iterdir()):
            if not candidate.is_file() or candidate.name in SKIP_ARTIFACTS:
                continue
            mlflow.log_artifact(str(candidate))
            logged_bytes += candidate.stat().st_size
        log.info("logged %d artifact file(s), %.1f MB",
                 sum(1 for f in run_dir.iterdir() if f.is_file()),
                 logged_bytes / 1e6)

        trace = str(manifest.get("trace_file") or "")
        run_type = args.run_type or ("smoke" if "smoke" in trace else "full")
        mlflow.set_tags({
            "run_type": run_type,
            "scenario": scenario,
            "source": "aiperf_trace_replay",
            "stack_env": stack_env,
            "host": "msi-rtx5060-wsl2",
        })

        crashes = metrics.get("vllm_restarts_during_run")
        log.info("logged %d metrics from %s", len(metrics), run_dir.name)
        if crashes:
            log.warning("vLLM restarted %.0f time(s) during this run", crashes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
