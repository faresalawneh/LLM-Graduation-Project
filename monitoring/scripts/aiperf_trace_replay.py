#!/usr/bin/env python3
"""
AIPerf BurstGPT trace replay with live OTel metrics streaming.

Alternative benchmark path to burstgpt_real_replay.py: AIPerf (>= 0.8.0)
replays the BurstGPT trace natively via --custom-dataset-type burst_gpt_trace
--fixed-schedule, and streams per-batch metrics over OTLP/HTTP DURING the run
to the otel-collector (monitoring/docker-compose.yml). The collector maps them
to ttft_ms / request_latency_ms / throughput_tokens_per_sec with a scenario
label (see monitoring/otel/otel-collector-config.yml), so the existing Grafana
dashboards update live and work unchanged.

Requires the otel extra: pip install "aiperf[otel]"

AIPerf also auto-collects vLLM server metrics (333ms interval) into
server_metrics_export.{json,csv,jsonl} inside the artifact dir.

Note: --endpoint-type chat works with burst_gpt_trace (verified with
aiperf 0.11.0 against Qwen2.5-0.5B-Instruct); completions works as well
and is the default.

The old jsonl-tail + Pushgateway path was removed: aiperf 0.11 only flushes
profile_export.jsonl at end of run, so tailing gave no live data. The
reference implementation lives on in aiperf_with_push.py.
"""

import json
import logging
import os
import shutil
import subprocess
from contextlib import suppress
import threading
import time
from pathlib import Path
from typing import List

import requests

# ── Config ────────────────────────────────────────────────────────────────────
INFERENCE_URL   = os.getenv("INFERENCE_URL", "http://localhost:8000")
MODEL           = os.getenv("MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
TRACE_FILE      = Path(os.getenv("TRACE_FILE", "test_trace.csv"))
SCENARIO        = os.getenv("SCENARIO", "steady")   # "steady" | "burst"
ARTIFACT_ROOT   = Path(os.getenv("ARTIFACT_ROOT", "aiperf_results"))
ENDPOINT_TYPE   = os.getenv("ENDPOINT_TYPE", "completions")
TOKENIZER       = os.getenv("TOKENIZER", "")   # optional local tokenizer path/id
SLICE_DURATION  = os.getenv("SLICE_DURATION", "5")

# Live metrics endpoint. IMPORTANT: aiperf 0.11 does NOT auto-append the OTLP
# path — --otel-url must be the FULL http://host:4318/v1/metrics URL. With a
# bare host:port URL aiperf silently exports nothing.
# Set OTEL_URL="" to disable streaming.
OTEL_URL        = os.getenv("OTEL_URL", "http://localhost:4318/v1/metrics")

PROMETHEUS_URL  = os.getenv("PROMETHEUS_URL", "http://localhost:9090") + "/api/v1/query"

# Laptop has no DCGM exporter, so GPU scraping is off unless explicitly enabled.
ENABLE_GPU_SCRAPE = os.getenv("ENABLE_GPU_SCRAPE", "0") == "1"
GPU_QUERIES = [
    'DCGM_FI_DEV_SM_CLOCK{gpu="0"}',
    'DCGM_FI_DEV_FB_USED{gpu="0"}',
    'DCGM_FI_DEV_GPU_TEMP{gpu="0"}',
    'DCGM_FI_DEV_POWER_USAGE{gpu="0"}',
    'DCGM_FI_DEV_MEM_CLOCK{gpu="0"}',
    'DCGM_FI_DEV_FB_FREE{gpu="0"}',
    'DCGM_FI_DEV_FB_TOTAL{gpu="0"}',
    'NVML_DEV_PROC_COUNT{gpu="0"}',
    'NVML_DEV_PROC_FB_USED{gpu="0"}',
    'NVML_DEV_CLOCK_THROTTLE_REASONS{gpu="0"}',
    "vllm:num_requests_waiting",
    "vllm:num_requests_running",
    "vllm:gpu_cache_usage_perc",
]
GPU_SCRAPE_INTERVAL = float(os.getenv("GPU_SCRAPE_INTERVAL", "5"))
CONCURRENCY = os.environ.get("MAX_CONCURRENCY", "20")
VLLM_CONTAINER = os.getenv("VLLM_CONTAINER", "vllm")
VLLM_LOG_TAIL = int(os.getenv("VLLM_LOG_TAIL", "2000"))
AIPERF_FINALIZE_TIMEOUT = float(os.getenv("AIPERF_FINALIZE_TIMEOUT", "300"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("aiperf_trace_replay")


def build_aiperf_command(artifact_dir: Path) -> List[str]:
    executable = shutil.which("aiperf") or "aiperf"
    command = [
        executable,
        "profile",
        "--model",
        MODEL,
        "--url",
        INFERENCE_URL,
        "--endpoint-type",
        ENDPOINT_TYPE,
        "--streaming",
        "--input-file",
        str(TRACE_FILE),
        "--custom-dataset-type",
        "burst_gpt_trace",
        "--fixed-schedule",
        "--concurrency", CONCURRENCY,
        "--server-metrics-formats",
        "json",
        "csv",
        "jsonl",
        "--slice-duration",
        SLICE_DURATION,
        "--artifact-dir",
        str(artifact_dir),
    ]
    if TOKENIZER:
        command += ["--tokenizer", TOKENIZER]
    if OTEL_URL:
        command += [
            "--otel-url", OTEL_URL,
            "--stream", "metrics",
            "--otel-resource-attributes", "scenario={}".format(SCENARIO),
        ]
    return command


# ── Metric helpers ────────────────────────────────────────────────────────────

def query_prometheus(query: str):
    try:
        r = requests.get(PROMETHEUS_URL, params={"query": query}, timeout=5)
        r.raise_for_status()
        result = r.json().get("data", {}).get("result", [])
        if result:
            return float(result[0]["value"][1])
    except Exception:
        pass
    return None


# ── GPU scrape thread ─────────────────────────────────────────────────────────

def gpu_scrape_thread(stop, artifact_dir):
    artifact_dir.mkdir(parents=True, exist_ok=True)
    path = artifact_dir / "gpu_telemetry.jsonl"
    _missing_warned = set()
    with path.open("a") as f:
        while not stop.is_set():
            t0 = time.monotonic()
            snap = {"timestamp": int(time.time())}
            for q in GPU_QUERIES:
                v = query_prometheus(q)
                if v is None:
                    if q not in _missing_warned:
                        log.warning("GPU scrape missing (omitting field): %s", q)
                        _missing_warned.add(q)
                    continue
                snap[q] = v
            if len(snap) > 1:
                f.write(json.dumps(snap) + "\n")
                f.flush()
            stop.wait(max(0.0, GPU_SCRAPE_INTERVAL - (time.monotonic() - t0)))


# ── Scenario runner ───────────────────────────────────────────────────────────

def count_trace_rows(trace_file: Path) -> int:
    with trace_file.open() as f:
        return max(sum(1 for line in f if line.strip()) - 1, 0)  # minus header


def _git_rev() -> str:
    """Return the current harness commit, or "unknown" outside a git tree."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent, text=True, timeout=5
        ).strip()
    except Exception:
        return "unknown"


def write_run_manifest(artifact_dir: Path, scenario: str) -> None:
    manifest = {
        "start_time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "scenario": scenario,
        "inference_url": INFERENCE_URL,
        "model": MODEL,
        "endpoint_type": ENDPOINT_TYPE,
        "trace_file": str(TRACE_FILE),
        "trace_row_count": count_trace_rows(TRACE_FILE),
        "artifact_dir": str(artifact_dir),
        "slice_duration": SLICE_DURATION,
        "tokenizer": TOKENIZER,
        "enable_gpu_scrape": ENABLE_GPU_SCRAPE,
        "otel_url": OTEL_URL,
        "concurrency": CONCURRENCY,
        "gpu_scrape_interval": GPU_SCRAPE_INTERVAL,
        "gpu_queries": GPU_QUERIES,
        "harness_commit": _git_rev(),
    }
    with (artifact_dir / "run_manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)


def capture_vllm_logs(artifact_dir: Path, phase: str, since: str = None) -> None:
    """Snapshot the vLLM container log into the artifact directory.

    The startup banner carries the KV block allocation ("# GPU blocks: N"),
    which is unrecoverable after the fact. With restart:unless-stopped the
    stream spans restarts, so a faulted run yields one banner per generation
    in the post-run capture.

    Non-following and timeout-bounded: `docker logs -f` re-streams the whole
    history and has previously hung the harness.
    """
    dest = artifact_dir / "logs"
    dest.mkdir(parents=True, exist_ok=True)
    cmd = ["docker", "logs"]
    if since:
        cmd += ["--since", since]
    else:
        # unbounded `docker logs` returns the container's entire history
        # (hundreds of MB on a long-lived container); the startup banner is
        # within the last few hundred lines of the most recent generation.
        cmd += ["--tail", str(VLLM_LOG_TAIL)]
    cmd.append(VLLM_CONTAINER)
    try:
        out = subprocess.run(  # nosec B603 - fixed argv, no shell
            cmd, capture_output=True, text=True, timeout=30, check=False,
        )
        text = out.stdout + out.stderr
        (dest / "vllm_{}.log".format(phase)).write_text(text)
        banners = [ln for ln in text.splitlines() if "GPU blocks" in ln]
        if banners:
            log.info("vLLM allocation banner(s) captured (%s): %d", phase, len(banners))
        elif phase == "startup":
            log.warning(
                "no allocation banner in last %d lines: container predates the "
                "tail window; block count for this generation is unrecorded",
                VLLM_LOG_TAIL,
            )
    except (subprocess.SubprocessError, OSError) as exc:
        log.warning("vLLM log capture (%s) failed: %s", phase, exc)


def _finalize_watchdog(process, artifact_dir: Path) -> int:
    """Wait for AIPerf, bounding the post-replay finalisation phase.

    After a mid-replay engine kill AIPerf completes the replay but stalls
    indefinitely awaiting in-flight responses that will never arrive, leaving
    profile_export.jsonl complete and no aggregates written. Rather than wait,
    the process tree is terminated once the record file stops growing and the
    bound elapses; recover_aggregates.py reconstructs the aggregates.

    Returns:
        The process return code, or -1 if the run was reaped.
    """
    records = artifact_dir / "profile_export.jsonl"
    last_size = -1
    idle_since = None
    while True:
        try:
            return process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass
        size = records.stat().st_size if records.exists() else 0
        if size != last_size:
            last_size, idle_since = size, time.monotonic()
            continue
        if idle_since is None:
            idle_since = time.monotonic()
        idle = time.monotonic() - idle_since
        if idle >= AIPERF_FINALIZE_TIMEOUT:
            log.warning(
                "AIPerf idle %.0fs after last record (bound %.0fs); reaping",
                idle, AIPERF_FINALIZE_TIMEOUT,
            )
            process.kill()
            with suppress(Exception):
                process.wait(timeout=30)
            # reap orphaned children of the killed process only; a bare
            # `pkill -f aiperf` would also match this script's own cmdline
            subprocess.run(  # nosec B603 - fixed argv, no shell
                ["pkill", "-P", str(process.pid)], check=False, timeout=15,
            )
            (artifact_dir / "RECOVERED").write_text(
                "AIPerf finalisation did not complete; reaped after "
                "{:.0f}s idle. Aggregates from recover_aggregates.py.\n".format(idle)
            )
            return -1


def run_scenario(scenario: str, artifact_dir: Path) -> int:
    artifact_dir.mkdir(parents=True, exist_ok=False)
    write_run_manifest(artifact_dir, scenario)

    stop_event = threading.Event()
    gpu_thread = None
    if ENABLE_GPU_SCRAPE:
        gpu_thread = threading.Thread(
            target=gpu_scrape_thread,
            args=(stop_event, artifact_dir),
            daemon=True,
        )
        gpu_thread.start()

    command = build_aiperf_command(artifact_dir)
    log.info("[%s] starting AIPerf: %s", scenario, " ".join(command))

    capture_vllm_logs(artifact_dir, "startup")
    run_start = time.strftime("%Y-%m-%dT%H:%M:%S")
    process = subprocess.Popen(command)
    try:
        rc = _finalize_watchdog(process, artifact_dir)
    finally:
        stop_event.set()
        capture_vllm_logs(artifact_dir, "run", since=run_start)
    if gpu_thread is not None:
        gpu_thread.join(timeout=10)
    return rc


def main() -> int:
    if SCENARIO not in ("steady", "burst"):
        log.error("SCENARIO must be 'steady' or 'burst', got %r", SCENARIO)
        return 2
    if not TRACE_FILE.exists():
        log.error("TRACE_FILE not found: %s", TRACE_FILE)
        return 2

    artifact_dir = ARTIFACT_ROOT / "{}_{}".format(SCENARIO, time.strftime("%Y-%m-%d_%H%M"))
    # two runs within the same minute would collide on exist_ok=False; add a suffix
    suffix = 1
    while artifact_dir.exists():
        artifact_dir = ARTIFACT_ROOT / "{}_{}_{}".format(
            SCENARIO, time.strftime("%Y-%m-%d_%H%M"), suffix)
        suffix += 1
    log.info("Starting %s trace replay of %s", SCENARIO, TRACE_FILE)
    rc = run_scenario(SCENARIO, artifact_dir)
    log.info("AIPerf exited with code %s (artifacts in %s)", rc, artifact_dir)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
