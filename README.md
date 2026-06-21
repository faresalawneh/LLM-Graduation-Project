# Real-Time Observability Pipeline for LLM Serving

Real-time observability that links LLM serving metrics to GPU hardware telemetry in one view.

## About

This is our Graduation Project (GP1) at Jordan University of Science and Technology, Faculty of Computer and Information Technology.

When an LLM server slows down, it is hard to tell whether the model, the GPU, or the request queue is the cause. Existing tools watch the model or the hardware, rarely both at once. We serve an LLM with vLLM, drive it with realistic traffic from the BurstGPT trace, and collect application metrics (TTFT, latency, throughput) together with GPU metrics (utilisation, memory, SM clock) into a single Grafana view. A FastAPI service exposes the same data over a REST API with JWT auth.

## Tech Stack

| Layer | Tool |
|---|---|
| Inference | vLLM serving `facebook/opt-125m` on dual Tesla P40 |
| Load generation | AIPerf + BurstGPT trace replay |
| Metrics pipeline | Prometheus + Pushgateway |
| GPU telemetry | NVIDIA DCGM Exporter |
| Dashboards | Grafana |
| API | FastAPI (JWT auth, role-based access) |
| CI/CD | GitHub Actions + Docker Hub |

## Repository Structure

```
.
├── api/                          # FastAPI observability service
│   └── app.py
├── monitoring/
│   ├── docker-compose.yml         # Prometheus, Pushgateway, Grafana, DCGM
│   ├── prometheus/prometheus.yml
│   ├── grafana/provisioning/
│   └── scripts/
│       ├── aiperf_with_push.py        # fixed-concurrency benchmark
│       └── burstgpt_real_replay.py    # real trace replay
├── .github/workflows/ci.yml
└── README.md
```

## Getting Started

### Prerequisites
- NVIDIA GPU with drivers, Docker, and Docker Compose
- Python 3.11

### 1. Start vLLM
```bash
CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server \
  --model facebook/opt-125m \
  --dtype float16 \
  --enforce-eager \
  --port 8000
```

### 2. Start the monitoring stack
```bash
cd monitoring
docker compose up -d
```
This brings up Prometheus (9090), Pushgateway (9091), Grafana (3000), and DCGM Exporter (9400).

### 3. Start the API
```bash
docker run -d --name llm-observability-api --network host \
  faresalawneh/llm-observability-api:latest
```
The API runs on port 5000. Swagger UI is at `http://localhost:5000/docs`.

## API Endpoints

| Method | Endpoint | Access | Description |
|---|---|---|---|
| GET | `/health` | Public | Service health check |
| POST | `/auth/token` | Public | Get a JWT (OAuth2 password flow) |
| GET | `/metrics/summary` | Authenticated | TTFT, latency, throughput by scenario |
| GET | `/gpu` | Admin | GPU telemetry from DCGM |
| GET | `/vllm` | Admin | vLLM queue and KV cache metrics |

Two roles are used: `admin` (full access) and `viewer` (no access to `/gpu` and `/vllm`). Default demo credentials are `admin / admin123` and `viewer / viewer123` and should be changed for any real deployment.

## Benchmarking

Both scripts push per-request metrics to Pushgateway and scrape GPU telemetry from Prometheus.

```bash
cd monitoring/scripts

# wipe old metrics first
curl -X PUT http://localhost:9091/api/v1/admin/wipe

# fixed concurrency (steady c=3, burst c=50)
python aiperf_with_push.py

# real BurstGPT timestamp replay
python burstgpt_real_replay.py
```

## Zero-config quick start

```bash
docker run -d -p 9090:9090 faresalawneh/llm-observability-prometheus:latest
docker run -d -p 9091:9091 faresalawneh/llm-observability-pushgateway:latest
docker run -d -p 3000:3000 faresalawneh/llm-observability-grafana:latest
```

The baked Prometheus config points scrape targets at localhost, so this is the quickest way to demo the stack on one machine. For cross-host setups, use Docker Compose with environment overrides instead.

## Running locally vs server

Server:

```bash
cd monitoring
docker compose --profile server up -d
cd monitoring/scripts
curl -X PUT http://localhost:9091/api/v1/admin/wipe
python burstgpt_real_replay.py
```

Local laptop:

```bash
cd monitoring
docker compose --profile local up -d
docker exec ollama ollama pull llama3.2:1b
cd monitoring/scripts
curl -X PUT http://localhost:9091/api/v1/admin/wipe
ENABLE_GPU_SCRAPE=0 INFERENCE_URL=http://localhost:11434/v1/completions MODEL=llama3.2:1b python burstgpt_real_replay.py
```

On the laptop, DCGM and vLLM targets will show as down in Prometheus. That is expected.

## Results

Concurrency sweep from c=1 to c=50 found a clear sweet spot at c=10: 1,811 tokens/s at 73 ms TTFT.

| Metric | Steady (c=3) | Burst (c=50) |
|---|---|---|
| First-token latency (TTFT) | 37 ms | 200 ms |
| End-to-end latency | 993 ms | 1,664 ms |
| Throughput | 217 tok/s | 175 tok/s |
| GPU utilisation | 35% | 85% |
| Queue depth | 0 | 30 |

GPU saturation and a rising queue show up 15 to 30 seconds before first-token latency degrades, which gives an early warning that is not visible from application metrics alone.

## CI/CD

`ci.yml` runs on every push to `main`:
1. `pytest` for the API endpoints
2. `bandit` security scan (zero high-severity findings required)
3. Docker build and push for `faresalawneh/llm-observability-api:latest`, `faresalawneh/llm-observability-grafana:latest`, `faresalawneh/llm-observability-prometheus:latest`, and `faresalawneh/llm-observability-pushgateway:latest`

## Team

- Fares Alawneh (168712)
- Mahmoud Al-waqfi (173032)
- Abdelrahman Tahat (169161)

Supervised by Dr. Tariq Al-omari.

## License

Developed as Graduation Project I at Jordan University of Science and Technology. Intellectual property rights are held by the university under its IP policy.
