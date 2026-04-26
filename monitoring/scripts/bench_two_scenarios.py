import asyncio, csv, json, logging, time
from pathlib import Path
import aiohttp, requests

VLLM_MODEL = "facebook/opt-125m"
VLLM_URL = "http://localhost:8000/v1/completions"
PUSHGATEWAY_URL = "http://localhost:9091"
BURSTGPT_CSV_PATH = Path("/media/works/BurstGPT_without_fails_3.csv")
STEADY_DURATION_SEC = 420
STEADY_RATE = 3
BURST_N_REQUESTS = 500
PAUSE_BETWEEN_SEC = 15
MAX_MODEL_LEN = 512
MAX_CONCURRENT = 50

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("benchmark")

def load_rows(path, n=10000):
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        req_col = next(k for k in headers if "request" in k.lower() and "token" in k.lower())
        resp_col = next(k for k in headers if "response" in k.lower() and "token" in k.lower())
        for row in reader:
            try:
                req = min(int(float(row[req_col])), 200)
                resp = min(int(float(row[resp_col])), 100)
            except:
                continue
            if req > 0 and resp > 0 and req + resp <= MAX_MODEL_LEN:
                rows.append((req, resp))
            if len(rows) >= n:
                break
    logger.info("Loaded %d rows", len(rows))
    return rows

def push_metrics(scenario, ttft, latency, throughput):
    job = "llm_bench"
    payload = "# TYPE ttft_ms gauge\n"
    payload += 'ttft_ms{scenario="' + scenario + '"} ' + str(ttft) + "\n"
    payload += "# TYPE request_latency_ms gauge\n"
    payload += 'request_latency_ms{scenario="' + scenario + '"} ' + str(latency) + "\n"
    payload += "# TYPE throughput_tokens_per_sec gauge\n"
    payload += 'throughput_tokens_per_sec{scenario="' + scenario + '"} ' + str(throughput) + "\n"
    try:
        requests.post(PUSHGATEWAY_URL + "/metrics/job/" + job + "/scenario/" + scenario,
                      data=payload, headers={"Content-Type": "text/plain"}, timeout=5)
    except Exception as e:
        logger.warning("Push failed: %s", e)

async def send_request(session, sem, prompt_tokens, max_tokens, idx, scenario):
    async with sem:
        payload = {"model": VLLM_MODEL, "prompt": "hello " * prompt_tokens,
                   "max_tokens": max_tokens, "stream": True}
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
                    except:
                        pass
        except Exception as e:
            logger.warning("[%s][%d] failed: %s", scenario, idx, e)
            return
        latency = (time.perf_counter() - t0) * 1000
        throughput = tokens / (latency / 1000) if latency > 0 else 0
        push_metrics(scenario, ttft or 0, latency, throughput)
        logger.info("[%s][%d] TTFT=%.1fms lat=%.1fms tps=%.1f", scenario, idx, ttft or 0, latency, throughput)

async def run_steady(session, sem):
    logger.info("=== SCENARIO 1: STEADY ===")
    rows = load_rows(BURSTGPT_CSV_PATH)
    interval = 1.0 / STEADY_RATE
    start = time.perf_counter()
    tasks, idx = [], 0
    while (time.perf_counter() - start) < STEADY_DURATION_SEC:
        req_tok, resp_tok = rows[idx % len(rows)]
        tasks.append(asyncio.create_task(send_request(session, sem, req_tok, resp_tok, idx, "steady")))
        idx += 1
        await asyncio.sleep(interval)
    await asyncio.gather(*tasks)
    logger.info("=== STEADY complete %d requests ===", idx)

async def run_burst(session, sem):
    logger.info("=== SCENARIO 2: BURST ===")
    rows = load_rows(BURSTGPT_CSV_PATH, BURST_N_REQUESTS)
    burst_size = BURST_N_REQUESTS // 5
    pause_between = max((420 / 5) - (burst_size * 0.05), 1)
    idx = 0
    for b in range(5):
        logger.info("  Burst %d/5", b + 1)
        tasks = []
        for i in range(burst_size):
            if idx >= len(rows):
                break
            req_tok, resp_tok = rows[idx]
            tasks.append(asyncio.create_task(send_request(session, sem, req_tok, resp_tok, idx, "burst")))
            idx += 1
            await asyncio.sleep(0.05)
        await asyncio.gather(*tasks)
        if b < 4:
            await asyncio.sleep(pause_between)
    logger.info("=== BURST complete ===")

async def main():
    sem = asyncio.Semaphore(MAX_CONCURRENT)
    async with aiohttp.ClientSession() as session:
        await run_steady(session, sem)
        logger.info("=== PAUSE %ds ===", PAUSE_BETWEEN_SEC)
        await asyncio.sleep(PAUSE_BETWEEN_SEC)
        await run_burst(session, sem)
    logger.info("=== ALL DONE ===")

asyncio.run(main())
