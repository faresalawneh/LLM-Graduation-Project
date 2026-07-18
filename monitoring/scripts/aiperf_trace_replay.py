#!/usr/bin/env python3
"""
AIPerf BurstGPT trace replay with live Pushgateway export.

Alternative benchmark path to burstgpt_real_replay.py: instead of firing
requests ourselves, AIPerf (>= 0.8.0) replays the BurstGPT trace natively via
--custom-dataset-type burst_gpt_trace --fixed-schedule, while a tail thread
reads profile_export.jsonl and pushes the same per-request metrics
(ttft_ms / request_latency_ms / throughput_tokens_per_sec) to the Pushgateway
under the same scenario job names (llm_bench_steady / llm_bench_burst), so the
existing Grafana dashboards work unchanged.

AIPerf also auto-collects vLLM server metrics (333ms interval) into
server_metrics_export.{json,csv,jsonl} inside the artifact dir.

Note: --endpoint-type chat works with burst_gpt_trace (verified with
aiperf 0.11.0 against Qwen2.5-0.5B-Instruct); completions was not needed.
"""

import json
import logging
import os
import shutil
import subprocess
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

# Live metrics: aiperf streams OTel metrics DURING the run to the otel-collector
# (monitoring/docker-compose.yml), which renames them to ttft_ms /
# request_latency_ms / throughput_tokens_per_sec for Prometheus/Grafana.
# Requires the otel extra: pip install "aiperf[otel]". Set OTEL_URL="" to disable.
OTEL_URL        = os.getenv("OTEL_URL", "http://localhost:4318")
# Legacy Pushgateway path: tails profile_export.jsonl and pushes per-request
# metrics. aiperf 0.11 only flushes the jsonl at end of run, so this gives no
# live data anymore — kept as a fallback, off by default.
ENABLE_TAIL_PUSH = os.getenv("ENABLE_TAIL_PUSH", "0") == "1"
SLICE_DURATION  = os.getenv("SLICE_DURATION", "5")

PUSHGATEWAY_URL = os.getenv("PUSHGATEWAY_URL", "http://localhost:9091") + "/metrics/job/{job}"
PROMETHEUS_URL  = os.getenv("PROMETHEUS_URL", "http://localhost:9090") + "/api/v1/query"

# Laptop has no DCGM exporter, so GPU scraping is off unless explicitly enabled.
ENABLE_GPU_SCRAPE = os.getenv("ENABLE_GPU_SCRAPE", "0") == "1"
GPU_QUERIES = [
    'DCGM_FI_DEV_SM_CLOCK{gpu="1"}',
    'DCGM_FI_DEV_FB_USED{gpu="1"}',
    "vllm:num_requests_waiting",
    "vllm:gpu_cache_usage_perc",
]
GPU_SCRAPE_INTERVAL = 5

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

def push_metric(name: str, value: float, scenario: str) -> None:
    job = "llm_bench_{}".format(scenario)
    url = PUSHGATEWAY_URL.format(job=job)
    payload = '# TYPE {} gauge\n{}{{scenario="{}"}} {}\n'.format(name, name, scenario, value)
    try:
        requests.post(url, data=payload,
                      headers={"Content-Type": "text/plain"}, timeout=5)
    except Exception as e:
        log.warning("Pushgateway error: %s", e)


def extract_metric_value(record: dict, dotted_path: str):
    current = record
    for part in dotted_path.split('.'):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    if isinstance(current, dict):
        if "value" in current:
            current = current["value"]
        elif "avg" in current:
            current = current["avg"]
    try:
        return float(current)
    except (TypeError, ValueError):
        return None


def push_request_metrics(record: dict, scenario: str) -> None:
    # Schema verified against aiperf 0.11.0: per-request records still carry
    # metrics.<name>.value dicts in ms, same as the paths aiperf_with_push.py uses.
    ttft_ms = extract_metric_value(record, "metrics.time_to_first_token.value")
    latency_ms = extract_metric_value(record, "metrics.request_latency.value")
    tps = extract_metric_value(record, "metrics.output_token_throughput_per_user.value")

    if ttft_ms is None or latency_ms is None or tps is None:
        return

    push_metric("ttft_ms", ttft_ms, scenario)
    push_metric("request_latency_ms", latency_ms, scenario)
    push_metric("throughput_tokens_per_sec", tps, scenario)
    log.info(
        "[%s] pushed ttft_ms=%.2f request_latency_ms=%.2f throughput_tokens_per_sec=%.2f",
        scenario, ttft_ms, latency_ms, tps,
    )


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


# ── JSONL tail thread ─────────────────────────────────────────────────────────

def tail_jsonl_and_push(stop_event: threading.Event, artifact_dir: Path, scenario: str) -> None:
    export_path = artifact_dir / "profile_export.jsonl"
    log.info("[%s] waiting for %s", scenario, export_path)

    while not stop_event.is_set() and not export_path.exists():
        time.sleep(0.5)

    if stop_event.is_set():
        return

    log.info("[%s] tailing %s", scenario, export_path)
    with export_path.open("r") as handle:
        while not stop_event.is_set():
            line = handle.readline()
            if not line:
                time.sleep(0.2)
                continue

            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                log.warning("[%s] skipping malformed JSON line", scenario)
                continue

            push_request_metrics(record, scenario)


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
        "pushgateway_url": PUSHGATEWAY_URL.format(job="llm_bench_" + scenario),
        "enable_gpu_scrape": ENABLE_GPU_SCRAPE,
        "otel_url": OTEL_URL,
        "enable_tail_push": ENABLE_TAIL_PUSH,
    }
    with (artifact_dir / "run_manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)


def run_scenario(scenario: str, artifact_dir: Path) -> int:
    # artifact_dir is freshly timestamped per run, so profile_export.jsonl
    # cannot exist yet and the tail thread always reads it from the beginning.
    artifact_dir.mkdir(parents=True, exist_ok=False)
    write_run_manifest(artifact_dir, scenario)

    stop_event = threading.Event()
    tail_thread = None
    if ENABLE_TAIL_PUSH:
        tail_thread = threading.Thread(
            target=tail_jsonl_and_push,
            args=(stop_event, artifact_dir, scenario),
            daemon=True,
        )
        tail_thread.start()

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

    process = subprocess.Popen(command)
    try:
        rc = process.wait()
    finally:
        if tail_thread is not None:
            # give the tail thread a moment to drain the last records
            time.sleep(2)
        stop_event.set()
    if tail_thread is not None:
        tail_thread.join(timeout=10)
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
