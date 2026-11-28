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

AIPerf runs with the reader patch in aiperf_hooks/ (a race in AIPerf 0.11.0), and
a stall guard stops a run that waits for records that will never arrive. Either
problem makes this script exit with INVALID_RUN_EXIT (#31).

The old jsonl-tail + Pushgateway path was removed: aiperf 0.11 only flushes
profile_export.jsonl at end of run, so tailing gave no live data. The
old implementation was removed; see git history.
"""

import json
import logging
import os
import re
import shutil
import signal
import subprocess  # nosec B404 - used only with list argv and no shell
import threading
import time
from pathlib import Path
from typing import Callable, List, Optional

import requests

from aiperf_patch_status import HOOK_DIR, STATUS_NAME, read_patch_status

# ── Config ────────────────────────────────────────────────────────────────────
INFERENCE_URL   = os.getenv("INFERENCE_URL", "http://localhost:8000")
MODEL           = os.getenv("MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
TRACE_FILE      = Path(os.getenv("TRACE_FILE", "test_trace.csv"))
SCENARIO        = os.getenv("SCENARIO", "steady")   # "steady" | "burst"
ARTIFACT_ROOT   = Path(os.getenv("ARTIFACT_ROOT", "aiperf_results"))
ENDPOINT_TYPE   = os.getenv("ENDPOINT_TYPE", "completions")
TOKENIZER       = os.getenv("TOKENIZER", "")   # optional local tokenizer path/id
TOKENIZER_REVISION = os.getenv("TOKENIZER_REVISION", "")   # optional Hub commit id
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
    "vllm:num_requests_waiting",
    "vllm:gpu_cache_usage_perc",
]
GPU_SCRAPE_INTERVAL = 5

# ── Stall guard ───────────────────────────────────────────────────────────────
# AIPerf 0.11.0 waits without any time limit until it has one record per completed
# request. A healthy run finished about 5 ms after the notice; 120 s leaves a wide
# margin. This is deliberately not configurable from the environment.
STALL_GUARD_SECONDS = 120
GUARD_POLL_SECONDS = 1.0
# Exit code for a run that must not be used. run_experiment.sh still logs it.
INVALID_RUN_EXIT = 86
GUARD_REPORT_NAME = "stall_guard.json"
# records_manager lines in aiperf 0.11.0.
WAITING_LINE = re.compile(
    r"records_manager - NOTICE - All requests have completed, please wait for the "
    r"results to be processed \(currently ([\d,]+) of ([\d,]+) records processed\)")
ALL_RECEIVED_LINE = re.compile(
    r"records_manager - INFO - Processed \d+ valid requests and \d+ errors "
    r"\(\d+ total\)")

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
        "--concurrency", os.environ.get("MAX_CONCURRENCY", "100"),
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
    if TOKENIZER_REVISION:
        command += ["--tokenizer-revision", TOKENIZER_REVISION]
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
    except Exception as exc:
        log.warning("Prometheus query failed (%s): %s", query, exc)
    return None


# ── GPU scrape thread ─────────────────────────────────────────────────────────

def gpu_scrape_thread(stop, artifact_dir):
    artifact_dir.mkdir(parents=True, exist_ok=True)
    path = artifact_dir / "gpu_telemetry.jsonl"
    with path.open("a") as f:
        while not stop.is_set():
            snap = {"timestamp": int(time.time())}
            ok = True
            for q in GPU_QUERIES:
                v = query_prometheus(q)
                if v is None:
                    ok = False
                    log.warning("GPU scrape missing: %s", q)
                    break
                snap[q] = v
            if ok:
                f.write(json.dumps(snap) + "\n")
                f.flush()
            stop.wait(GPU_SCRAPE_INTERVAL)


# ── Stall guard ───────────────────────────────────────────────────────────────

class _Tail:
    """Returns the complete lines added to a growing file since the last call."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.offset = 0
        self.partial = b""

    def new_lines(self) -> List[bytes]:
        try:
            with self.path.open("rb") as handle:
                handle.seek(self.offset)
                data = handle.read()
        except FileNotFoundError:
            return []
        self.offset += len(data)
        *lines, self.partial = (self.partial + data).split(b"\n")
        return lines


class StallGuard:
    """Decides when AIPerf is stuck waiting for records.

    It arms on AIPerf's "waiting for results" notice. From then on, the number
    of lines in profile_export.jsonl must grow at least once per limit. It
    disarms when AIPerf reports that it has received all records, because the
    count cannot grow after that and only the export remains.
    """

    def __init__(
        self,
        artifact_dir: Path,
        seconds: float = STALL_GUARD_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.seconds = seconds
        self.clock = clock
        self._log = _Tail(artifact_dir / "logs" / "aiperf.log")
        self._records = _Tail(artifact_dir / "profile_export.jsonl")
        self._last_growth: Optional[float] = None
        self.records_on_disk = 0
        self.notice_seen = False
        self.all_received_seen = False
        self.processed_at_notice: Optional[int] = None
        self.expected_at_notice: Optional[int] = None
        self.fired = False
        self.fired_utc: Optional[str] = None
        self.records_on_disk_at_fire: Optional[int] = None

    def observe(self) -> None:
        """Read what AIPerf has written since the last call."""
        for raw in self._log.new_lines():
            line = raw.decode("utf-8", errors="replace")
            waiting = WAITING_LINE.search(line)
            if waiting and not self.notice_seen:
                self.notice_seen = True
                self.processed_at_notice, self.expected_at_notice = (
                    int(value.replace(",", "")) for value in waiting.groups())
                self._last_growth = self.clock()
            if ALL_RECEIVED_LINE.search(line):
                self.all_received_seen = True
        added = len(self._records.new_lines())
        if added:
            self.records_on_disk += added
            if self.notice_seen:
                self._last_growth = self.clock()

    def should_fire(self) -> bool:
        """Return True once, when the stall limit has been reached."""
        self.observe()
        if self.fired or not self.notice_seen or self.all_received_seen:
            return False
        if self.clock() - self._last_growth < self.seconds:
            return False
        self.fired = True
        self.fired_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self.records_on_disk_at_fire = self.records_on_disk
        return True

    def report(self) -> dict:
        """Everything the guard saw, for the run's record."""
        return {
            "stall_guard_seconds": self.seconds,
            "guard_fired": self.fired,
            "notice_seen": self.notice_seen,
            "all_records_received_seen": self.all_received_seen,
            "records_processed_at_notice": self.processed_at_notice,
            "records_expected_at_notice": self.expected_at_notice,
            "fired_utc": self.fired_utc,
            "records_on_disk_at_fire": self.records_on_disk_at_fire,
            "records_on_disk_at_exit": self.records_on_disk,
        }


def wait_with_guard(
    process, guard: StallGuard, poll_seconds: float = GUARD_POLL_SECONDS,
) -> int:
    """Wait for AIPerf, interrupting it once if it stalls.

    Exactly one SIGINT goes to the AIPerf main process, not to its group: AIPerf
    saves its records on the first one and force-quits on a second. There is no
    SIGKILL, so this keeps waiting while AIPerf writes its results.
    """
    while True:
        try:
            code = process.wait(timeout=poll_seconds)
        except subprocess.TimeoutExpired:
            if guard.should_fire():
                log.error(
                    "STALL GUARD: no new record for %s s after AIPerf's completion "
                    "notice (%s of %s processed). Sending one SIGINT. THIS RUN IS "
                    "INVALID.", guard.seconds, guard.processed_at_notice,
                    guard.expected_at_notice)
                process.send_signal(signal.SIGINT)
            continue
        guard.observe()
        return code


def invalid_reasons(code: int, guard: StallGuard, patch: dict) -> List[str]:
    """Reasons this script rejects a run. Request errors are judged separately."""
    reasons = []
    if guard.fired:
        reasons.append("the stall guard fired")
    if code == 0 and patch["active"] != "true":
        reasons.append("reader patch not verified: {}".format(patch["problem"]))
    if code == 0 and not guard.all_received_seen:
        reasons.append(
            "AIPerf's all-records line was never seen, so the stall guard's "
            "coverage of this run is unverified")
    return reasons


# ── AIPerf environment ────────────────────────────────────────────────────────

def aiperf_environment(artifact_dir: Path) -> dict:
    """Environment for the AIPerf command only: loads the reader patch."""
    env = dict(os.environ)
    inherited = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(HOOK_DIR) + (os.pathsep + inherited if inherited else ""))
    env["AIPERF_PATCH_STATUS_FILE"] = str(artifact_dir.resolve() / STATUS_NAME)
    return env


# ── Scenario runner ───────────────────────────────────────────────────────────

def count_trace_rows(trace_file: Path) -> int:
    with trace_file.open() as f:
        return max(sum(1 for line in f if line.strip()) - 1, 0)  # minus header


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
        "tokenizer_revision": TOKENIZER_REVISION,
        "enable_gpu_scrape": ENABLE_GPU_SCRAPE,
        "otel_url": OTEL_URL,
    }
    with (artifact_dir / "run_manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)


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

    guard = StallGuard(artifact_dir)
    process = subprocess.Popen(  # nosec B603 - list argv, no shell; executable from shutil.which, args from module config
        command, env=aiperf_environment(artifact_dir))
    try:
        rc = wait_with_guard(process, guard)
    finally:
        stop_event.set()
    if gpu_thread is not None:
        gpu_thread.join(timeout=10)

    patch = read_patch_status(artifact_dir / STATUS_NAME)
    reasons = invalid_reasons(rc, guard, patch)
    with (artifact_dir / GUARD_REPORT_NAME).open("w") as f:
        json.dump({**guard.report(), "aiperf_exit_code": rc,
                   "reader_patch_active": patch["active"],
                   "invalid_reasons": reasons}, f, indent=2)
    for reason in reasons:
        log.error("[%s] RUN INVALID: %s", scenario, reason)
    return INVALID_RUN_EXIT if reasons else rc


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
