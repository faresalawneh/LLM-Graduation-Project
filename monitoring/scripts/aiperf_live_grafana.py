#!/usr/bin/env python3

import logging
import re
import shutil
import subprocess
import threading
import time
from typing import Dict, List

import requests


VLLM_MODEL = "facebook/opt-125m"
VLLM_URL = "http://localhost:8000"
VLLM_METRICS_URL = "http://localhost:8000/metrics"
PUSHGATEWAY_URL = "http://localhost:9091/metrics/job/aiperf_vllm"
ARTIFACT_DIR = "/media/works/aiperf_results/live"

POLL_INTERVAL_SEC = 5

AIPERF_CMD = [
    "aiperf",
    "profile",
    "--model",
    VLLM_MODEL,
    "--url",
    VLLM_URL,
    "--endpoint-type",
    "completions",
    "--streaming",
    "--concurrency",
    "10",
    "--request-count",
    "500",
    "--isl",
    "200",
    "--osl",
    "100",
    "--extra-inputs",
    "ignore_eos:true",
    "--artifact-dir",
    ARTIFACT_DIR,
]

METRIC_NAMES = [
    "vllm:e2e_request_latency_seconds",
    "vllm:num_requests_waiting",
    "vllm:gpu_cache_usage_perc",
]

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("aiperf_live_grafana")


def resolve_aiperf_command() -> List[str]:
    executable = shutil.which("aiperf")
    if executable:
        return [executable, *AIPERF_CMD[1:]]

    fallback = "/media/works/llm-env/bin/aiperf"
    if shutil.which(fallback):
        logger.info("Using fallback AIPerf binary at %s", fallback)
        return [fallback, *AIPERF_CMD[1:]]

    logger.warning("AIPerf binary not found on PATH; attempting to use raw command name")
    return AIPERF_CMD


def fetch_vllm_metrics() -> str:
    response = requests.get(VLLM_METRICS_URL, timeout=5)
    response.raise_for_status()
    return response.text


def extract_selected_metrics(metrics_text: str) -> str:
    selected_lines: List[str] = []
    histogram_prefix = "vllm:e2e_request_latency_seconds"
    wanted_prefixes = tuple(METRIC_NAMES)

    for line in metrics_text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        if stripped.startswith(histogram_prefix + "_sum") or stripped.startswith(histogram_prefix + "_count"):
            selected_lines.append(stripped)
            continue

        if stripped.startswith(wanted_prefixes):
            selected_lines.append(stripped)

    return "\n".join(selected_lines) + ("\n" if selected_lines else "")


def push_metrics(payload: str) -> None:
    if not payload.strip():
        logger.info("No selected vLLM metrics found to push")
        return

    requests.post(
        PUSHGATEWAY_URL,
        data=payload,
        headers={"Content-Type": "text/plain"},
        timeout=5,
    )


def poll_and_push(stop_event: threading.Event) -> None:
    while not stop_event.is_set():
        try:
            metrics_text = fetch_vllm_metrics()
            payload = extract_selected_metrics(metrics_text)
            if payload:
                push_metrics(payload)
                logger.info("Pushed selected vLLM metrics to Pushgateway")
        except requests.RequestException as exc:
            logger.warning("Metrics polling/push failed: %s", exc)

        stop_event.wait(POLL_INTERVAL_SEC)


def main() -> int:
    stop_event = threading.Event()
    poll_thread = threading.Thread(target=poll_and_push, args=(stop_event,), daemon=True)
    poll_thread.start()

    command = resolve_aiperf_command()
    logger.info("Starting AIPerf: %s", " ".join(command))

    process = subprocess.Popen(command)
    try:
        return_code = process.wait()
        logger.info("AIPerf exited with code %s", return_code)
        return return_code
    finally:
        stop_event.set()


if __name__ == "__main__":
    raise SystemExit(main())