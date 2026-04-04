"""
benchmark.py
============
Run on the JUST GPU Server (Ubuntu).
Pipeline:
  1. Convert BurstGPT CSV → JSONL
  2. Start vLLM server
  3. Run AIPerf trace-replay
  4. Push metrics to Pushgateway on Windows machine

Usage:
    pip install vllm aiperf pandas requests
    python benchmark.py
"""

import subprocess
import time
import json
import logging
from pathlib import Path
from typing import Dict

import pandas as pd
import requests

# ──────────────────────────────────────────────────────────────────
# CONFIGURATION — edit these before running
# ──────────────────────────────────────────────────────────────────
VLLM_MODEL          = "facebook/opt-125m"
VLLM_PORT           = 8000

# Path to BurstGPT CSV on the server
BURSTGPT_CSV_PATH   = Path("C:\llm-observability\monitoring\BurstGPT_without_fails_3.csv")  # CHANGE

# Windows machine IP on university network (run: ipconfig → IPv4)
PUSHGATEWAY_URL     = "http://CHANGE_ME:9091"

N_REQUESTS          = 500
AIPERF_CONCURRENCY  = 10

OUTPUT_JSONL        = Path("/tmp/burstgpt_replay.jsonl")
AIPERF_ARTIFACT_DIR = Path("/tmp/aiperf_artifacts")
# ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("benchmark")


def convert_csv_to_jsonl(csv_path: Path, jsonl_path: Path, n: int) -> None:
    """Convert BurstGPT CSV to AIPerf single-turn JSONL."""
    logger.info("Converting CSV → JSONL (n=%d)", n)
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    req_col  = next(c for c in df.columns if "request"  in c.lower() and "token" in c.lower())
    resp_col = next(c for c in df.columns if "response" in c.lower() and "token" in c.lower())

    df = df.dropna(subset=[req_col, resp_col]).head(n)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    with open(jsonl_path, "w") as f:
        for _, row in df.iterrows():
            f.write(json.dumps({
                "text":          "hello " * max(1, int(row[req_col])),
                "output_length": max(1, int(row[resp_col]))
            }) + "\n")

    logger.info("Wrote %d requests → %s", len(df), jsonl_path)


def start_vllm_server() -> subprocess.Popen:
    """Launch vLLM and wait until the health endpoint responds."""
    logger.info("Starting vLLM — model: %s  port: %d", VLLM_MODEL, VLLM_PORT)
    proc = subprocess.Popen(
        ["python", "-m", "vllm.entrypoints.openai.api_server",
         "--model", VLLM_MODEL,
         "--port",  str(VLLM_PORT)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    for attempt in range(60):
        try:
            r = requests.get(f"http://localhost:{VLLM_PORT}/health", timeout=2)
            if r.status_code == 200:
                logger.info("vLLM ready after %ds", attempt * 2)
                return proc
        except Exception:
            pass
        time.sleep(2)
    proc.terminate()
    raise RuntimeError("vLLM did not become ready in 120s")


def run_aiperf(jsonl_path: Path, artifact_dir: Path) -> Dict[str, float]:
    """Run AIPerf and return parsed metrics dict."""
    artifact_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "aiperf", "profile", VLLM_MODEL,
        "--url",                 f"http://localhost:{VLLM_PORT}",
        "--endpoint-type",       "completions",
        "--streaming",
        "--custom-dataset-type", "single-turn",
        "--input-file",          str(jsonl_path),
        "--request-count",       str(N_REQUESTS),
        "--concurrency",         str(AIPERF_CONCURRENCY),
        "--ui-type",             "simple",
        "--output-artifact-dir", str(artifact_dir),
    ]
    logger.info("Running AIPerf…")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error("AIPerf stderr:\n%s", result.stderr)
        raise RuntimeError("AIPerf failed")

    summary_file = artifact_dir / "summary.json"
    if summary_file.exists():
        data = json.loads(summary_file.read_text())
        return {
            "ttft_ms":                   data.get("avg_ttft_ms",      0.0),
            "itl_ms":                    data.get("avg_itl_ms",       0.0),
            "request_latency_ms":        data.get("avg_latency_ms",   0.0),
            "throughput_tokens_per_sec": data.get("output_throughput", 0.0),
        }

    # Fallback: parse stdout
    metrics: Dict[str, float] = {
        "ttft_ms": 0.0, "itl_ms": 0.0,
        "request_latency_ms": 0.0, "throughput_tokens_per_sec": 0.0
    }
    for line in result.stdout.splitlines():
        low = line.lower()
        try:
            val = float(line.split(":")[-1].strip())
        except ValueError:
            continue
        if "ttft"       in low: metrics["ttft_ms"]                   = val
        elif "itl"      in low: metrics["itl_ms"]                    = val
        elif "latency"  in low: metrics["request_latency_ms"]        = val
        elif "throughput" in low and "token" in low:
            metrics["throughput_tokens_per_sec"] = val
    return metrics


def push_to_pushgateway(metrics: Dict[str, float]) -> None:
    """Push AIPerf metrics to Prometheus Pushgateway."""
    payload = "\n".join(
        f"# TYPE {k} gauge\n{k} {v}" for k, v in metrics.items()
    ) + "\n"

    endpoint = PUSHGATEWAY_URL.rstrip("/") + "/metrics/job/aiperf/instance/just_gpu"
    resp = requests.post(
        endpoint,
        data=payload,
        headers={"Content-Type": "text/plain"},
        timeout=10,
    )
    resp.raise_for_status()
    logger.info("Metrics pushed → %s (HTTP %d)", endpoint, resp.status_code)
    for k, v in metrics.items():
        logger.info("  %-35s = %.4f", k, v)


def main() -> None:
    assert "CHANGE_ME" not in PUSHGATEWAY_URL, \
        "Set PUSHGATEWAY_URL to your Windows machine IP first!"
    assert BURSTGPT_CSV_PATH.exists(), \
        f"CSV not found: {BURSTGPT_CSV_PATH}"

    logger.info("=== LLM Benchmark Pipeline ===")
    convert_csv_to_jsonl(BURSTGPT_CSV_PATH, OUTPUT_JSONL, N_REQUESTS)
    vllm_proc = start_vllm_server()

    try:
        metrics = run_aiperf(OUTPUT_JSONL, AIPERF_ARTIFACT_DIR)
        push_to_pushgateway(metrics)
        logger.info("Pipeline complete.")
    except Exception as exc:
        logger.exception("Pipeline failed: %s", exc)
    finally:
        vllm_proc.terminate()
        logger.info("vLLM stopped.")


if __name__ == "__main__":
    main()
