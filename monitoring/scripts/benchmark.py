import asyncio
import time
import csv
import json
import logging
from pathlib import Path
import aiohttp
import requests

# ──────────────────────────────────────────────────────────────────
# CONFIGURATION — edit these before running
# ──────────────────────────────────────────────────────────────────
VLLM_MODEL        = "facebook/opt-125m"
VLLM_PORT         = 8000
BURSTGPT_CSV_PATH = Path("/media/works/BurstGPT_without_fails_3.csv")
PUSHGATEWAY_URL   = "http://localhost:9091"
N_REQUESTS        = 500
REQUEST_RATE      = 10  # req/s
# ──────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("benchmark")
VLLM_URL = f"http://localhost:{VLLM_PORT}/v1/completions"


def load_burstgpt(path, n):
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        req_col  = next(k for k in headers if "request"  in k.lower() and "token" in k.lower())
        resp_col = next(k for k in headers if "response" in k.lower() and "token" in k.lower())
        for row in reader:
            try:
                req  = int(float(row[req_col]))
                resp = int(float(row[resp_col]))
            except Exception:
                continue
            if req + resp <= 2048 and req > 0 and resp > 0:
                rows.append((req, resp))
            if len(rows) >= n:
                break
    logger.info("Loaded %d requests from BurstGPT", len(rows))
    return rows


def push_metric(name, value, job="llm_bench"):
    payload = "# TYPE " + name + " gauge\n" + name + " " + str(value) + "\n"
    requests.post(
        PUSHGATEWAY_URL + "/metrics/job/" + job,
        data=payload,
        headers={"Content-Type": "text/plain"},
        timeout=5,
    )


async def send_request(session, prompt_tokens, max_tokens, idx):
    payload = {
        "model": VLLM_MODEL,
        "prompt": "hello " * prompt_tokens,
        "max_tokens": max_tokens,
        "stream": True,
    }
    t0 = time.perf_counter()
    ttft, tokens = None, 0
    async with session.post(VLLM_URL, json=payload) as resp:
        async for line in resp.content:
            line = line.decode().strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
                text = chunk["choices"][0].get("text", "")
                if text and ttft is None:
                    ttft = (time.perf_counter() - t0) * 1000
                tokens += len(text.split())
            except Exception:
                pass
    latency = (time.perf_counter() - t0) * 1000
    throughput = tokens / (latency / 1000) if latency > 0 else 0
    push_metric("ttft_ms", ttft or 0)
    push_metric("request_latency_ms", latency)
    push_metric("throughput_tokens_per_sec", throughput)
    logger.info("[%d] TTFT=%.1fms  lat=%.1fms  tps=%.1f", idx, ttft or 0, latency, throughput)


async def main():
    assert BURSTGPT_CSV_PATH.exists(), "CSV not found: " + str(BURSTGPT_CSV_PATH)
    rows = load_burstgpt(BURSTGPT_CSV_PATH, N_REQUESTS)
    logger.info("=== Starting benchmark — %d requests @ %d req/s ===", len(rows), REQUEST_RATE)
    async with aiohttp.ClientSession() as session:
        tasks = []
        for i, (req_tok, resp_tok) in enumerate(rows):
            tasks.append(asyncio.create_task(send_request(session, req_tok, resp_tok, i)))
            await asyncio.sleep(1 / REQUEST_RATE)
        await asyncio.gather(*tasks)
    logger.info("=== Benchmark complete ===")


if __name__ == "__main__":
    asyncio.run(main())
