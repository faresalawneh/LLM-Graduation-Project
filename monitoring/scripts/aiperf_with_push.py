#!/usr/bin/env python3

import json
import logging
import subprocess
import threading
import time
from pathlib import Path
from typing import List
import shutil

import requests


VLLM_MODEL = "facebook/opt-125m"
VLLM_URL = "http://localhost:8000"
PUSHGATEWAY_URL = "http://localhost:9091/metrics/job/llm_bench"
DEFAULT_ARTIFACT_ROOT = Path("/media/works/aiperf_results")
DEFAULT_CONCURRENCY = 10
DEFAULT_REQUEST_COUNT = 500
DEFAULT_ISL = 200
DEFAULT_OSL = 100
DEFAULT_EXTRA_INPUTS = "ignore_eos:true"
STEADY_PAUSE_SEC = 15
RUN_ONLY = "burst"
PROMETHEUS_QUERY_URL = "http://localhost:9090/api/v1/query"
GPU_TELEMETRY_FILE_NAME = "gpu_telemetry.jsonl"
GPU_QUERIES = [
    "DCGM_FI_DEV_SM_CLOCK{gpu=\"1\"}",
    "DCGM_FI_DEV_FB_USED{gpu=\"1\"}",
    "vllm:num_requests_waiting",
    "vllm:gpu_cache_usage_perc",
]
GPU_SCRAPE_INTERVAL_SEC = 5

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("aiperf_with_push")


def build_aiperf_command(artifact_dir: Path, concurrency: int, request_count: int) -> List[str]:
    executable = shutil.which("aiperf") or "/media/works/llm-env/bin/aiperf"
    return [
        executable,
        "profile",
        "--model",
        VLLM_MODEL,
        "--url",
        VLLM_URL,
        "--endpoint-type",
        "completions",
        "--streaming",
        "--concurrency",
        str(concurrency),
        "--request-count",
        str(request_count),
        "--isl",
        str(DEFAULT_ISL),
        "--osl",
        str(DEFAULT_OSL),
        "--extra-inputs",
        DEFAULT_EXTRA_INPUTS,
        "--artifact-dir",
        str(artifact_dir),
    ]


def push_metric(name: str, value: float, scenario: str) -> None:
    payload = "\n".join([
        f'# TYPE {name} gauge',
        f'{name}{{scenario="{scenario}"}} {value}',
        "",
    ])
    requests.post(
        PUSHGATEWAY_URL,
        data=payload,
        headers={"Content-Type": "text/plain"},
        timeout=5,
    )


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
    ttft_ms = extract_metric_value(record, "metrics.time_to_first_token.value")
    latency_ms = extract_metric_value(record, "metrics.request_latency.value")
    tps = extract_metric_value(record, "metrics.output_token_throughput_per_user.value")

    if ttft_ms is None or latency_ms is None or tps is None:
        return

    push_metric("ttft_ms", ttft_ms, scenario)
    push_metric("request_latency_ms", latency_ms, scenario)
    push_metric("throughput_tokens_per_sec", tps, scenario)
    logger.info(
        "[%s] pushed ttft_ms=%.2f request_latency_ms=%.2f throughput_tokens_per_sec=%.2f",
        scenario,
        ttft_ms,
        latency_ms,
        tps,
    )


def query_prometheus_value(metric_query: str):
    response = requests.get(PROMETHEUS_QUERY_URL, params={"query": metric_query}, timeout=5)
    response.raise_for_status()
    payload = response.json()
    if payload.get("status") != "success":
        return None

    result = payload.get("data", {}).get("result", [])
    if not result:
        return None

    value = result[0].get("value", [])
    if len(value) < 2:
        return None

    try:
        return float(value[1])
    except (TypeError, ValueError):
        return None


def gpu_scrape_thread(stop_event: threading.Event, artifact_dir: Path) -> None:
    telemetry_path = artifact_dir / GPU_TELEMETRY_FILE_NAME
    logger.info("GPU scrape thread waiting for %s", artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    with telemetry_path.open("a") as handle:
        while not stop_event.is_set():
            snapshot = {"timestamp": int(time.time())}
            missing_metric = False

            for metric_query in GPU_QUERIES:
                value = query_prometheus_value(metric_query)
                if value is None:
                    missing_metric = True
                    logger.warning("GPU scrape missing metric %s", metric_query)
                    break
                snapshot[metric_query] = value

            if not missing_metric:
                handle.write(json.dumps(snapshot) + "\n")
                handle.flush()
                logger.info("Wrote GPU telemetry snapshot to %s", telemetry_path)

            stop_event.wait(GPU_SCRAPE_INTERVAL_SEC)


def tail_jsonl_and_push(stop_event: threading.Event, artifact_dir: Path, scenario: str) -> None:
    export_path = artifact_dir / "profile_export.jsonl"
    logger.info("[%s] waiting for %s", scenario, export_path)

    while not stop_event.is_set() and not export_path.exists():
        time.sleep(0.5)

    if stop_event.is_set():
        return

    logger.info("[%s] tailing %s", scenario, export_path)
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
                logger.warning("[%s] skipping malformed JSON line", scenario)
                continue

            push_request_metrics(record, scenario)


def run_scenario(
    *,
    scenario: str,
    concurrency: int,
    request_count: int,
    artifact_dir: Path,
) -> int:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    export_path = artifact_dir / "profile_export.jsonl"
    if export_path.exists():
        export_path.unlink()

    stop_event = threading.Event()
    tail_thread = threading.Thread(
        target=tail_jsonl_and_push,
        args=(stop_event, artifact_dir, scenario),
        daemon=True,
    )
    tail_thread.start()

    gpu_thread = threading.Thread(
        target=gpu_scrape_thread,
        args=(stop_event, artifact_dir),
        daemon=True,
    )
    gpu_thread.start()

    command = build_aiperf_command(artifact_dir, concurrency, request_count)
    logger.info("[%s] starting AIPerf: %s", scenario, " ".join(command))

    process = subprocess.Popen(command)
    try:
        return process.wait()
    finally:
        stop_event.set()


def main() -> int:
    steady_artifact_dir = DEFAULT_ARTIFACT_ROOT / "steady"
    burst_artifact_dir = DEFAULT_ARTIFACT_ROOT / "burst"

    steady_rc = 0
    burst_rc = 0

    if RUN_ONLY in ("steady", "both"):
        logger.info("Starting steady scenario")
        steady_rc = run_scenario(
            scenario="steady",
            concurrency=3,
            request_count=2700,
            artifact_dir=steady_artifact_dir,
        )
        logger.info("Steady scenario exited with code %s", steady_rc)

    if RUN_ONLY == "both":
        logger.info("Pausing %ds before burst scenario", STEADY_PAUSE_SEC)
        time.sleep(STEADY_PAUSE_SEC)

    if RUN_ONLY in ("burst", "both"):
        logger.info("Starting burst scenario")
        burst_rc = run_scenario(
            scenario="burst",
            concurrency=50,
            request_count=13500,
            artifact_dir=burst_artifact_dir,
        )
        logger.info("Burst scenario exited with code %s", burst_rc)

    if RUN_ONLY == "steady":
        return steady_rc
    if RUN_ONLY == "burst":
        return burst_rc
    return burst_rc if burst_rc != 0 else steady_rc


if __name__ == "__main__":
    raise SystemExit(main())