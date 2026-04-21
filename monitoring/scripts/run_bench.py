import asyncio
import time
import csv
import json
import logging
import requests
import aiohttp
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
VLLM_MODEL        = "facebook/opt-125m"
VLLM_URL          = "http://localhost:8000/v1/completions"
BURSTGPT_CSV_PATH = Path("/media/works/BurstGPT_without_fails_3.csv")
N_REQUESTS        = 2000
TARGET_RPS        = 157.0
MAX_MODEL_LEN     = 512
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("benchmark")


def load_burstgpt(path, n):
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                ts   = float(row["Timestamp"])
                req  = min(int(float(row["Request tokens"])),  200)
                resp = min(int(float(row["Response tokens"])), 100)
            except (KeyError, ValueError):
                continue
            if req > 0 and resp > 0 and req + resp <= MAX_MODEL_LEN:
                rows.append((ts, req, resp))
            if len(rows) >= n:
                break
    if not rows:
        raise ValueError("No valid rows loaded from CSV")
    t0 = rows[0][0]
    rows = [(ts - t0, req, resp) for ts, req, resp in rows]
    logger.info("Loaded %d requests, raw duration=%.1fs", len(rows), rows[-1][0])
    return rows


def compute_scale_factor(rows, target_rps):
    t_last = rows[-1][0]
    if t_last == 0:
        return 1.0
    c = len(rows) / (t_last * target_rps)
    logger.info("Scale factor c=%.4f -> scaled duration=%.1fs at %.1f RPS",
                c, t_last / c, target_rps)
    return c


async def send_request(session, prompt_tokens, max_tokens, idx):
    payload = {
        "model": VLLM_MODEL,
        "prompt": "hello " * prompt_tokens,
        "max_tokens": max_tokens,
        "stream": True,
    }
    t0 = time.perf_counter()
    ttft, tokens = None, 0
    try:
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
    except Exception as e:
        logger.warning("[%d] Request failed: %s", idx, e)
        return
    latency = (time.perf_counter() - t0) * 1000
    tps = tokens / (latency / 1000) if latency > 0 else 0
    logger.info("[%d] TTFT=%.1fms  lat=%.1fms  tps=%.1f", idx, ttft or 0, latency, tps)


async def main():
    rows = load_burstgpt(BURSTGPT_CSV_PATH, N_REQUESTS)
    t_last = rows[-1][0]
    c = t_last / 300.0  # compress to 300 seconds
    logger.info("Scale factor c=%.4f -> duration=300s", c)
    logger.info("=== Starting BurstGPT replay at %.1f RPS ===", TARGET_RPS)
    start = time.perf_counter()
    async with aiohttp.ClientSession() as session:
        tasks = []
        for i, (ts_norm, req_tok, resp_tok) in enumerate(rows):
            send_at = ts_norm / c
            now = time.perf_counter() - start
            delay = send_at - now
            if delay > 0:
                await asyncio.sleep(delay)
            tasks.append(asyncio.create_task(
                send_request(session, req_tok, resp_tok, i)
            ))
        await asyncio.gather(*tasks)
    logger.info("=== Benchmark complete ===")


asyncio.run(main())
