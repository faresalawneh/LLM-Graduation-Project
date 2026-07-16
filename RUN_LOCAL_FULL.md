# Running the Full Pipeline Locally (WSL2 + Docker)

Local dev/pre-production environment. Runs the whole stack on the laptop with a
tiny model (Qwen2.5-0.5B) to smoke-test everything before the P40 server.

Two parts:
1. **vLLM** - serves the model (Docker, standalone container)
2. **Monitoring stack** - Prometheus + Pushgateway + Grafana (Docker Compose)

Then the replay script drives traffic and you watch it in Grafana.

---

## Prerequisites

- Windows + WSL2 (Ubuntu)
- NVIDIA GPU, current driver (CUDA >= 12.8 in `nvidia-smi`)
- Docker running in WSL
- NVIDIA Container Toolkit (so Docker sees the GPU)

Verify GPU works in Docker:
```bash
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```
Prints GPU table = good. Errors on `--gpus` = install toolkit (bottom of file).

**Why Docker not a venv:** installing vLLM in a WSL venv fails on this laptop
(DTensor error, UVA not available, missing pyairports - version conflicts from
the fresh CUDA driver). The Docker image ships matching CUDA/torch/transformers,
so it just works. Same image runs on the server.

---

## Step 1 - Start vLLM

```bash
docker rm -f vllm-local 2>/dev/null || true

docker run --gpus all -d --name vllm-local \
  -v ~/hf_cache:/root/.cache/huggingface \
  -p 8000:8000 \
  vllm/vllm-openai:v0.6.3 \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --dtype float16 --max-model-len 2048 \
  --gpu-memory-utilization 0.80 --enforce-eager
```

Watch it boot (first run downloads ~1GB model):
```bash
docker logs -f vllm-local
```
Wait for `Application startup complete`. Ctrl+C to stop watching (container keeps running).

Test:
```bash
curl http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen2.5-0.5B-Instruct","messages":[{"role":"user","content":"Hi"}],"max_tokens":20}'
```
Returns JSON with a reply = vLLM working.

---

## Step 2 - Configure the monitoring stack for local

Two files must be set for local (they default to the server IP).

### `monitoring/prometheus/prometheus.yml`

The `vllm` job must target `host.docker.internal:8000` (the vLLM container from
inside the Prometheus container). Full file:

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: pushgateway
    honor_labels: true
    static_configs:
      - targets:
          - pushgateway:9091

  - job_name: dcgm_exporter
    static_configs:
      - targets:
          - dcgm-exporter:9400
    metrics_path: /metrics
    scrape_interval: 10s

  - job_name: vllm
    static_configs:
      - targets: ['host.docker.internal:8000']
    metrics_path: /metrics
    scrape_interval: 15s
```

Note: locally the `dcgm_exporter` job will just show as DOWN (no DCGM on the
laptop). That is fine - GPU panels stay empty, app panels work.

### `monitoring/docker-compose.yml`

The Prometheus service needs `extra_hosts` so `host.docker.internal` resolves
on Linux Docker:

```yaml
  prometheus:
    image: ${PROMETHEUS_IMAGE:-prom/prometheus:latest}
    container_name: prometheus
    restart: unless-stopped
    extra_hosts:
      - "host.docker.internal:host-gateway"
    volumes:
      - ./prometheus/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    ports:
      - "9090:9090"
    networks:
      - monitoring
```

IMPORTANT: `extra_hosts` and `prometheus:` belong ONLY in docker-compose.yml.
Never put them in prometheus.yml - that breaks Prometheus on startup.

---

## Step 3 - Start the monitoring stack

```bash
cd monitoring
docker compose up -d
docker compose ps          # prometheus, pushgateway, grafana all Up
```

Health checks:
```bash
curl -s localhost:9090/-/healthy          # Prometheus Server is Healthy
docker compose logs prometheus | tail -20 # if it won't stay up, read this
```

If Prometheus keeps dying, it is almost always a YAML indentation error in
prometheus.yml. All three `- job_name:` lines must be at the same indent (2 spaces).

Open the UIs:
- Grafana:    http://localhost:3000  (admin / admin)
- Prometheus: http://localhost:9090/targets  (see which jobs are UP)

---

## Step 4 - Run the replay

The replay drives traffic into vLLM and pushes per-request metrics to Pushgateway.
Convert the Windows path to a WSL path (`C:\...` becomes `/mnt/c/...`).

```bash
DATASET_PATH=/mnt/c/Users/alawn/Documents/GitHub/LLM-Graduation-Project/BurstGPT_without_fails_3.csv \
INFERENCE_URL=http://localhost:8000/v1/completions \
MODEL=Qwen/Qwen2.5-0.5B-Instruct \
python3 burstgpt_real_replay.py
```

If `requests` is missing: `pip install requests`.

Watch Grafana. The "Scenario Replay (Pushgateway)" row fills first (TTFT,
latency, throughput by scenario). The vLLM histogram and engine-state panels
fill once Prometheus is scraping vLLM directly (Step 2).

---

## What "working locally" looks like

- vLLM curl returns text
- `docker compose ps` shows 3 containers Up
- `localhost:9090/targets` shows `pushgateway` UP and `vllm` UP
  (`dcgm_exporter` DOWN is expected on the laptop)
- Grafana Scenario Replay row shows data during a replay
- Grafana engine-state panels (queue, running, KV cache) show data

This is a plumbing smoke test only. Real experiments (correct metrics, real
dataset, RCA data) run on the P40 server.

---

## Stop / clean up

```bash
docker stop vllm-local           # stop vLLM (keeps model cache)
cd monitoring && docker compose down   # stop monitoring stack
# wipe Pushgateway stale data before a fresh replay:
curl -X PUT http://localhost:9091/api/v1/admin/wipe
```

---

## Common issues

| Symptom | Cause | Fix |
|---|---|---|
| `Address already in use` :8000 | old vLLM container/process | `docker rm -f vllm-local` |
| Prometheus won't start | YAML indent error in prometheus.yml | align all `- job_name:` at 2 spaces |
| `lookup prometheus: no such host` in Grafana | Prometheus container down | `docker compose up -d`, check logs |
| vllm target DOWN in Prometheus | missing `extra_hosts` in compose | add `host.docker.internal:host-gateway` |
| GPU panels empty | no DCGM on laptop | expected - GPU metrics are server-only |
| 500 on request / missing module | happens only with venv install | use Docker |
| `--gpus` error | no NVIDIA Container Toolkit | install it (below) |

---

## Installing NVIDIA Container Toolkit (only if `--gpus` fails)

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt update && sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo service docker restart
```

---

## Moving to the P40 server

Same Docker image, change:
- model: `Qwen/Qwen2.5-7B-Instruct`
- `--max-model-len 4096`, `--gpu-memory-utilization 0.92`
- keep `--enforce-eager` (required on Pascal)
- prometheus.yml vllm target back to the server IP, dcgm_exporter comes UP
- re-verify vLLM metric names (`curl :8000/metrics | grep vllm:`) - may differ
