from flask import Flask, jsonify
import requests

app = Flask(__name__)

PROMETHEUS_URL = "http://localhost:9090"


def query_prometheus(promql):
    try:
        resp = requests.get(
            f"{PROMETHEUS_URL}/api/v1/query",
            params={"query": promql},
            timeout=5
        )
        data = resp.json()
        if data["status"] == "success" and data["data"]["result"]:
            return float(data["data"]["result"][0]["value"][1])
        return None
    except Exception:
        return None


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/metrics/summary", methods=["GET"])
def metrics_summary():
    ttft_steady = query_prometheus('ttft_ms{scenario="steady"}')
    ttft_burst = query_prometheus('ttft_ms{scenario="burst"}')
    latency_steady = query_prometheus('request_latency_ms{scenario="steady"}')
    latency_burst = query_prometheus('request_latency_ms{scenario="burst"}')
    throughput_steady = query_prometheus('throughput_tokens_per_sec{scenario="steady"}')
    throughput_burst = query_prometheus('throughput_tokens_per_sec{scenario="burst"}')

    return jsonify({
        "ttft_ms": {"steady": ttft_steady, "burst": ttft_burst},
        "latency_ms": {"steady": latency_steady, "burst": latency_burst},
        "throughput_tokens_per_sec": {"steady": throughput_steady, "burst": throughput_burst}
    })


@app.route("/gpu", methods=["GET"])
def gpu():
    sm_clock = query_prometheus("DCGM_FI_DEV_SM_CLOCK")
    memory_used = query_prometheus("DCGM_FI_DEV_FB_USED")
    temperature = query_prometheus("DCGM_FI_DEV_GPU_TEMP")
    power = query_prometheus("DCGM_FI_DEV_POWER_USAGE")

    return jsonify({
        "sm_clock_mhz": sm_clock,
        "memory_used_mib": memory_used,
        "temperature_c": temperature,
        "power_w": power
    })


@app.route("/vllm", methods=["GET"])
def vllm():
    queue_depth = query_prometheus("vllm:num_requests_waiting")
    running = query_prometheus("vllm:num_requests_running")
    kv_cache = query_prometheus("vllm:gpu_cache_usage_perc")

    return jsonify({
        "requests_waiting": queue_depth,
        "requests_running": running,
        "kv_cache_usage_pct": kv_cache
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)  # nosec
