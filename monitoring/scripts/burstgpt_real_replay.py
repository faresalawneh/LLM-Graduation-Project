#!/usr/bin/env python3
"""
BurstGPT Real Timestamp Replay
Replays LLM requests at their original inter-arrival times from the BurstGPT dataset.
Steady window: bucket 21661 (72 requests / 15 min)
Burst window:  bucket 23832 (9483 requests / 15 min)
"""

import csv
import json
import logging
import threading
import time
from pathlib import Path

import requests

# ── Config ────────────────────────────────────────────────────────────────────
VLLM_URL          = "http://localhost:8000/v1/completions"
VLLM_MODEL        = "facebook/opt-125m"
PUSHGATEWAY_URL   = "http://localhost:9091/metrics/job/{job}"
PROMETHEUS_URL    = "http://localhost:9090/api/v1/query"
DATASET_PATH      = Path("/media/works/BurstGPT_without_fails_3.csv")
ARTIFACT_ROOT     = Path("/media/works/aiperf_results")
MAX_TOKENS_CAP    = 1024   # cap outliers from dataset
MAX_CONCURRENCY   = 100    # max simultaneous active requests

# 15-minute bucket IDs from dataset analysis
STEADY_BUCKET     = 21661   # 72 requests / 15 min
BURST_BUCKET      = 23832   # 9483 requests / 15 min
BUCKET_DURATION   = 900     # seconds

GPU_QUERIES = [
    'DCGM_FI_DEV_SM_CLOCK{gpu="1"}',
    'DCGM_FI_DEV_FB_USED{gpu="1"}',
    "vllm:num_requests_waiting",
    "vllm:gpu_cache_usage_perc",
]
GPU_SCRAPE_INTERVAL       = 5
PAUSE_BETWEEN_SCENARIOS   = 15

RUN_ONLY = "both"   # "steady" | "burst" | "both"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("burstgpt_replay")


# ── Dataset loading ────────────────────────────────────────────────────────────

def load_window(bucket_id: int) -> list:
    start_ts = bucket_id * BUCKET_DURATION
    end_ts   = start_ts + BUCKET_DURATION
    rows = []
    with DATASET_PATH.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts = float(row["Timestamp"])
            if start_ts <= ts < end_ts:
                rows.append(row)
    rows.sort(key=lambda r: float(r["Timestamp"]))
    log.info("Loaded %d rows for bucket %d (%.0fs to %.0fs)",
             len(rows), bucket_id, start_ts, end_ts)
    return rows


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


# ── Single request sender ─────────────────────────────────────────────────────

def send_request(row, scenario, results, lock, semaphore):
    prompt_tokens  = int(float(row.get("Request tokens", 50)))
    response_tokens = int(float(row.get("Response tokens", 100)))
    max_tokens     = min(response_tokens, MAX_TOKENS_CAP)
    prompt         = "x " * min(prompt_tokens, 200)

    payload = {
        "model": VLLM_MODEL,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "stream": True,
    }

    with semaphore:
        t0 = time.perf_counter()
    ttft_ms = None
    output_tokens = 0

    try:
        with requests.post(VLLM_URL, json=payload, stream=True, timeout=120) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                line = line.decode("utf-8")
                if line.startswith("data:"):
                    line = line[5:].strip()
                if line == "[DONE]":
                    break
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # First chunk received = first token = TTFT
                if ttft_ms is None:
                    ttft_ms = (time.perf_counter() - t0) * 1000

                token_text = chunk.get("choices", [{}])[0].get("text", "")
                output_tokens += len(token_text.split())

        t1 = time.perf_counter()
        latency_ms = (t1 - t0) * 1000
        duration   = t1 - t0
        throughput = output_tokens / duration if duration > 0 else 0
        ttft_ms    = ttft_ms if ttft_ms is not None else latency_ms

        push_metric("ttft_ms", ttft_ms, scenario)
        push_metric("request_latency_ms", latency_ms, scenario)
        push_metric("throughput_tokens_per_sec", throughput, scenario)

        with lock:
            results.append({
                "latency_ms": latency_ms,
                "throughput": throughput,
                "ttft_ms": ttft_ms,
            })

        log.info("[%s] ttft=%.1fms latency=%.1fms throughput=%.1f tok/s",
                 scenario, ttft_ms, latency_ms, throughput)

    except Exception as e:
        log.warning("[%s] request failed: %s", scenario, e)


# ── Scenario runner ───────────────────────────────────────────────────────────

def run_scenario(scenario, bucket_id, artifact_dir):
    artifact_dir.mkdir(parents=True, exist_ok=True)
    rows = load_window(bucket_id)

    if not rows:
        log.error("No rows found for bucket %d", bucket_id)
        return

    stop = threading.Event()
    gpu_thread = threading.Thread(
        target=gpu_scrape_thread, args=(stop, artifact_dir), daemon=True)
    gpu_thread.start()

    semaphore = threading.Semaphore(MAX_CONCURRENCY)
    results = []
    lock    = threading.Lock()
    threads = []

    base_ts   = float(rows[0]["Timestamp"])
    run_start = time.perf_counter()

    log.info("[%s] replaying %d requests from bucket %d", scenario, len(rows), bucket_id)

    for row in rows:
        target_offset = float(row["Timestamp"]) - base_ts
        elapsed       = time.perf_counter() - run_start
        sleep_for     = target_offset - elapsed
        if sleep_for > 0:
            time.sleep(sleep_for)

        t = threading.Thread(
            target=send_request,
            args=(row, scenario, results, lock, semaphore),
            daemon=True,
        )
        t.start()
        threads.append(t)

    for t in threads:
        t.join(timeout=120)

    stop.set()

    if results:
        avg_latency   = sum(r["latency_ms"] for r in results) / len(results)
        avg_ttft      = sum(r["ttft_ms"] for r in results) / len(results)
        avg_throughput = sum(r["throughput"] for r in results) / len(results)
        sorted_lat    = sorted(r["latency_ms"] for r in results)
        p95_latency   = sorted_lat[int(len(sorted_lat) * 0.95)]

        log.info("[%s] DONE requests=%d avg_ttft=%.1fms avg_latency=%.1fms "
                 "p95_latency=%.1fms avg_throughput=%.1f tok/s",
                 scenario, len(results), avg_ttft, avg_latency,
                 p95_latency, avg_throughput)

        summary = {
            "scenario": scenario,
            "total_requests": len(results),
            "avg_ttft_ms": avg_ttft,
            "avg_latency_ms": avg_latency,
            "p95_latency_ms": p95_latency,
            "avg_throughput_tokens_per_sec": avg_throughput,
        }
        with (artifact_dir / "summary.json").open("w") as f:
            json.dump(summary, f, indent=2)
    else:
        log.warning("[%s] no successful requests", scenario)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if RUN_ONLY in ("steady", "both"):
        log.info("Starting steady scenario")
        run_scenario("steady", STEADY_BUCKET, ARTIFACT_ROOT / "steady_replay")

    if RUN_ONLY == "both":
        log.info("Pausing %ds before burst scenario", PAUSE_BETWEEN_SCENARIOS)
        time.sleep(PAUSE_BETWEEN_SCENARIOS)

    if RUN_ONLY in ("burst", "both"):
        log.info("Starting burst scenario")
        run_scenario("burst", BURST_BUCKET, ARTIFACT_ROOT / "burst_replay")


if __name__ == "__main__":
    main()
