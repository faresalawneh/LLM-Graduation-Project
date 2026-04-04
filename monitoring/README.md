# LLM Inference Monitoring Pipeline

> **Stack:** vLLM · AIPerf · BurstGPT · Prometheus · Pushgateway · Grafana · DCGM Exporter

---

## Architecture

```
[JUST GPU Server]
  vLLM (opt-125m)
      ↓ AIPerf replay
  AIPerf metrics → HTTP POST → Pushgateway (Windows)
  DCGM Exporter :9400 ←── Prometheus scrapes directly

[Windows Machine — university network]
  Pushgateway  :9091
  Prometheus   :9090  ←── scrapes Pushgateway + DCGM
  Grafana      :3000
```

---

## Folder Structure

```
llm-monitor/
├── docker-compose.yml
├── .env                          ← fill in IPs here
├── prometheus/
│   └── prometheus.yml
├── grafana/
│   └── provisioning/
│       ├── datasources/prometheus.yml
│       └── dashboards/
│           ├── dashboard.yml
│           └── dashboard.json
└── scripts/
    └── benchmark.py              ← run this on JUST server
```

---

## Step-by-Step

### 1 · Windows — fill in `.env`

```
WINDOWS_IP=<your IPv4>          # ipconfig → IPv4 Address
JUST_GPU_SERVER_IP=<server IP>  # on server: hostname -I
GRAFANA_PASSWORD=admin
```

### 2 · Windows — edit `prometheus/prometheus.yml`

Replace `CHANGE_ME` with the actual JUST server IP.

### 3 · Windows — start stack

```bash
docker-compose up -d
# verify: docker ps → prometheus, pushgateway, grafana
```

### 4 · JUST GPU Server — start DCGM Exporter

```bash
docker run -d \
  --name dcgm-exporter \
  --gpus all \
  --restart unless-stopped \
  -p 9400:9400 \
  nvcr.io/nvidia/k8s/dcgm-exporter:latest
```

### 5 · JUST GPU Server — run benchmark

```bash
pip install vllm aiperf pandas requests

# Edit scripts/benchmark.py:
#   BURSTGPT_CSV_PATH → path to your CSV
#   PUSHGATEWAY_URL   → http://<WINDOWS_IP>:9091

python scripts/benchmark.py
```

### 6 · Grafana

- Open: http://localhost:3000  
- Login: `admin` / `admin`  
- Dashboard auto-loaded. If not: Dashboards → Import → `dashboard.json`

---

## Metrics

| Metric | Source |
|--------|--------|
| `ttft_ms` | AIPerf |
| `itl_ms` | AIPerf |
| `request_latency_ms` | AIPerf |
| `throughput_tokens_per_sec` | AIPerf |
| `DCGM_FI_DEV_GPU_UTIL` | DCGM Exporter |
| `DCGM_FI_DEV_FB_USED` | DCGM Exporter |
| `DCGM_FI_DEV_MEM_COPY_UTIL` | DCGM Exporter |

---

## Verify Everything Works

```
Prometheus:   http://localhost:9090  → Status → Targets
              both pushgateway and dcgm_exporter must show UP

Pushgateway:  http://localhost:9091
Grafana:      http://localhost:3000
```
